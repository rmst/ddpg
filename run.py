#!/usr/bin/env python
import gym
import numpy as np
import filter_env
import experiment
import ddpg
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('gymkey','','gym key')
flags.DEFINE_string('env','','gym environment')
flags.DEFINE_integer('warmup',20000,'time without training but only filling the replay memory')
flags.DEFINE_integer('test',10000,'time between tests')
# ...
# TODO: make command line options
t_train = 1000000
n_test=20
render=False


class Experiment:
  def run(self):
    self.t_log = 103
    self.t_global = 0

    # create filtered environment
    self.env = filter_env.makeFilteredEnv(gym.make(FLAGS.env))

    self.env.monitor.start(FLAGS.outdir+'/monitor/',video_callable=lambda _: False)
    gym.logger.setLevel(gym.logging.WARNING)

    dimO = self.env.observation_space.shape
    dimA = self.env.action_space.shape

    print('dimO: '+str(dimO) +'  dimA: '+str(dimA))

    self.agent = ddpg.Agent(dimO=dimO,dimA=dimA)

    returns = []
    t_last_test = 0

    # main loop
    while self.t_global < t_train:

      # test
      t_last_test = self.t_global
      R = np.mean([self.run_episode(test=True,render=render) for _ in range(n_test)])
      returns.append((self.t_global, R))
      np.save(FLAGS.outdir+"/returns.npy",returns)
      print('Average return '+str(R)+ ' after '+str(self.t_global)+' timesteps of training')

      # train
      while self.t_global-t_last_test <  FLAGS.test:
        self.run_episode(test=False)

    self.env.monitor.close()
    # upload results
    if FLAGS.gymkey != '':
      gym.upload(FLAGS.outdir+"/monitor",FLAGS.gymkey)

  def run_episode(self,test=True,t_max=250,render=False):
    self.env.monitor.configure(lambda _: test) # TODO: capture video at test time
    observation = self.env.reset()
    self.agent.reset(observation)
    R = 0 # return
    t = 0
    term = False
    while not term:
      if render: self.env.render(mode='human')

      action = self.agent.act(test=test,logging=self.t_global%self.t_log==0)
      #action = self.env.action_space.sample()

      observation, reward, term, info = self.env.step(action)
      term = (t >= t_max) or term

      if not test:

        self.agent.observe(reward,term,observation,test=test)

        if self.t_global > FLAGS.warmup:
          self.agent.train(logging=self.t_global%self.t_log==0)

        self.t_global += 1

      R = R + reward
      t = t + 1

    return R


def main():
  Experiment().run()

if __name__ == '__main__':
  experiment.run(main)
