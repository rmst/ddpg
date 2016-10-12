import tensorflow as tf
import ddpg_nets_dm as nets_dm
from replay_memory import ReplayMemory
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('ou_sigma',0.2,'')
flags.DEFINE_float('ou_theta',0.15,'')
flags.DEFINE_integer('warmup',50000,'time without training but only filling the replay memory')
flags.DEFINE_bool('warmq',True,'train Q during warmup time')
flags.DEFINE_float('log',.01,'probability of writing a tensorflow log at each timestep')
flags.DEFINE_integer('bsize',32,'minibatch size')
flags.DEFINE_bool('async',True,'update policy and q function concurrently')

# ...
# TODO: make command line options
tau =.001
discount =.99
pl2 =.0
ql2 =.01
lrp =.0001
lrq =.001
rm_size = 1000000
rm_dtype = 'float32'
threads = 4

# DDPG Agent
# 
class Agent:

  def __init__(self, dimO, dimA):
    dimA = list(dimA)
    dimO = list(dimO)

    nets=nets_dm

    # init replay memory
    self.rm = ReplayMemory(rm_size, dimO, dimA, dtype=np.__dict__[rm_dtype])
    # start tf session
    self.sess = tf.Session(config=tf.ConfigProto(
      inter_op_parallelism_threads=threads,
      log_device_placement=False,
      allow_soft_placement=True))

    # create tf computational graph
    #
    self.theta_p = nets.theta_p(dimO, dimA)
    self.theta_q = nets.theta_q(dimO, dimA)
    self.theta_pt, update_pt = exponential_moving_averages(self.theta_p, tau)
    self.theta_qt, update_qt = exponential_moving_averages(self.theta_q, tau)

    obs = tf.placeholder(tf.float32, [None] + dimO, "obs")
    act_test, sum_p = nets.policy(obs, self.theta_p)

    # explore
    noise_init = tf.zeros([1]+dimA)
    noise_var = tf.Variable(noise_init)
    self.ou_reset = noise_var.assign(noise_init)
    noise = noise_var.assign_sub((FLAGS.ou_theta) * noise_var - tf.random_normal(dimA, stddev=FLAGS.ou_sigma))
    act_expl = act_test + noise

    # test
    q, sum_q = nets.qfunction(obs, act_test, self.theta_q)
    # training
    # policy loss
    meanq = tf.reduce_mean(q, 0)
    wd_p = tf.add_n([pl2 * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
    loss_p = -meanq + wd_p
    # policy optimization
    optim_p = tf.train.AdamOptimizer(learning_rate=lrp)
    grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
    optimize_p = optim_p.apply_gradients(grads_and_vars_p)
    with tf.control_dependencies([optimize_p]):
      train_p = tf.group(update_pt)

    # q optimization
    act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
    rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
    term = tf.placeholder(tf.bool, [FLAGS.bsize], "term")
    obs2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs2")
    # q
    q_train, sum_qq = nets.qfunction(obs, act_train, self.theta_q)
    # q targets
    act2, sum_p2 = nets.policy(obs2, theta=self.theta_pt)
    q2, sum_q2 = nets.qfunction(obs2, act2, theta=self.theta_qt)
    q_target = tf.stop_gradient(tf.select(term,rew,rew + discount*q2))
    # q_target = tf.stop_gradient(rew + discount * q2)
    # q loss
    td_error = q_train - q_target
    ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
    wd_q = tf.add_n([ql2 * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
    loss_q = ms_td_error + wd_q
    # q optimization
    optim_q = tf.train.AdamOptimizer(learning_rate=lrq)
    grads_and_vars_q = optim_q.compute_gradients(loss_q, var_list=self.theta_q)
    optimize_q = optim_q.apply_gradients(grads_and_vars_q)
    with tf.control_dependencies([optimize_q]):
      train_q = tf.group(update_qt)

    # logging
    log_obs = [] if dimO[0]>20 else [tf.histogram_summary("obs/"+str(i),obs[:,i]) for i in range(dimO[0])]
    log_act = [] if dimA[0]>20 else [tf.histogram_summary("act/inf"+str(i),act_test[:,i]) for i in range(dimA[0])]
    log_act2 = [] if dimA[0]>20 else [tf.histogram_summary("act/train"+str(i),act_train[:,i]) for i in range(dimA[0])]
    log_misc = [sum_p, sum_qq, tf.histogram_summary("td_error", td_error)]
    log_grad = [grad_histograms(grads_and_vars_p), grad_histograms(grads_and_vars_q)]
    log_train = log_obs + log_act + log_act2 + log_misc + log_grad

    # initialize tf log writer
    self.writer = tf.train.SummaryWriter(FLAGS.outdir+"/tf", self.sess.graph, flush_secs=20)

    # init replay memory for recording episodes
    max_ep_length = 10000
    self.rm_log = ReplayMemory(max_ep_length,dimO,dimA,rm_dtype) 

    # tf functions
    with self.sess.as_default():
      self._act_test = Fun(obs,act_test)
      self._act_expl = Fun(obs,act_expl)
      self._reset = Fun([],self.ou_reset)
      self._train_q = Fun([obs,act_train,rew,term, obs2],[train_q],log_train,self.writer)
      self._train_p = Fun([obs],[train_p],log_train,self.writer)
      self._train = Fun([obs,act_train,rew,term, obs2],[train_p,train_q],log_train,self.writer)

    # initialize tf variables
    self.saver = tf.train.Saver(max_to_keep=1)
    ckpt = tf.train.latest_checkpoint(FLAGS.outdir+"/tf")
    if ckpt:
      self.saver.restore(self.sess,ckpt)
    else:
      self.sess.run(tf.initialize_all_variables())

    self.sess.graph.finalize()

    self.t = 0  # global training time (number of observations)

  def reset(self, obs):
    self._reset()
    self.observation = obs  # initial observation

  def act(self, test=False):
    obs = np.expand_dims(self.observation, axis=0)
    action = self._act_test(obs) if test else self._act_expl(obs)
    self.action = np.atleast_1d(np.squeeze(action, axis=0)) # TODO: remove this hack
    return self.action

  def observe(self, rew, term, obs2, test=False):

    obs1 = self.observation
    self.observation = obs2

    # train
    if not test:
      self.t = self.t + 1
      self.rm.enqueue(obs1, term, self.action, rew)

      if self.t > FLAGS.warmup:
        self.train()

      elif FLAGS.warmq and self.rm.n > 1000:
        # Train Q on warmup
        obs, act, rew, term, ob2, info = self.rm.minibatch(size=FLAGS.bsize)
        self._train_q(obs,act,rew,term, ob2, log = (np.random.rand() < FLAGS.log), global_step=self.t)

      # save parameters etc.
      # if (self.t+45000) % 50000 == 0: # TODO: correct
      #   s = self.saver.save(self.sess,FLAGS.outdir+"f/tf/c",self.t)
      #   print("DDPG Checkpoint: " + s)

  def train(self):
    obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
    log = (np.random.rand() < FLAGS.log)

    if FLAGS.async:
      self._train(obs,act,rew,ob2,term2, log = log, global_step=self.t)
    else:
      self._train_q(obs,act,rew,ob2,term2, log = log, global_step=self.t)
      self._train_p(obs, log = log, global_step=self.t)

  def write_scalar(self,tag,val):
    s = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=val)])
    self.writer.add_summary(s,self.t)


  def __del__(self):
    self.sess.close()



# Tensorflow utils
#
class Fun:
  """ Creates a python function that maps between inputs and outputs in the computational graph. """
  def __init__(self, inputs, outputs,summary_ops=None,summary_writer=None, session=None ):
    self._inputs = inputs if type(inputs)==list else [inputs]
    self._outputs = outputs
    self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops)==list else summary_ops
    self._session = session or tf.get_default_session()
    self._writer = summary_writer
  def __call__(self, *args, **kwargs):
    """
    Arguments:
      **kwargs: input values
      log: if True write summary_ops to summary_writer
      global_step: global_step for summary_writer
    """
    log = kwargs.get('log',False)

    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg

    out = self._outputs + [self._summary_op] if log else self._outputs
    res = self._session.run(out, feeds)
    
    if log:
      i = kwargs['global_step']
      self._writer.add_summary(res[-1],global_step=i)
      res = res[:-1]

    return res

def grad_histograms(grads_and_vars):
  s = []
  for grad, var in grads_and_vars:
    s.append(tf.histogram_summary(var.op.name + '', var))
    s.append(tf.histogram_summary(var.op.name + '/gradients', grad))
  return tf.merge_summary(s)

def exponential_moving_averages(theta, tau=0.001):
  ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
  update = ema.apply(theta)  # also creates shadow vars
  averages = [ema.average(x) for x in theta]
  return averages, update
