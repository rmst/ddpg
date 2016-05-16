from gym import core, spaces
import numpy as np

cat = lambda *args: np.vstack(args)
inf = np.inf
pi = np.pi

class DoubleLinkEnv(core.Env):


  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 15
  }

  dt = .01
  dt = .05

  LINK_LENGTH_1 = 1.  # [m]
  LINK_LENGTH_2 = 1.  # [m]
  LINK_MASS_1 = 1.  #: [kg] mass of link 1
  LINK_MASS_2 = 1.  #: [kg] mass of link 2
  friction_coeff=[2.5,2.5]
  
  g = 9.81

  # stateLB=-np.array([pi,pi,50,50])
  # stateUB=np.array([pi,pi,50,50])
  stateLB=-np.array([1,1,1,1,20,20])
  stateUB=np.array([1,1,1,1,20,20])

  actionLB=-np.array([1,1])
  actionUB=np.array([1,1])
  rewardLB=- inf
  rewardUB=0

  def __init__(self,target=(0,2),gravity=9.81):
    """
    Args:
      target: end effector goal position in cartesian coordinates (inside radius of 2 around origin)

    """

    self.observation_space = spaces.Box(self.stateLB, self.stateUB)
    self.action_space = spaces.Box(self.actionLB,self.actionUB)

    self.viewer = None

    self.target = target
    self.g = gravity

  def _reset(self):
    self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
    #self.state = np.array([np.pi/2,0,0,0])
    return self.transformed_state()

  def transformed_state(self):
    q = self.state
    return np.array((np.sin(q[0]),np.cos(q[0]),np.sin(q[1]),np.cos(q[1]),q[2],q[3]))

  def _step(self, a):
    state = self.state
    action= 10. * np.clip(a,self.actionLB,self.actionUB)
    #action = np.array([-30*state[0],10])
    #action = np.array([10,-10*state[1]])

    gravity,coriolis,invM,friction=self.getDynamicsMatrices()

    A = action - coriolis - gravity - friction
    qdd= np.dot(invM,A)
    qd=state[2:] + self.dt * qdd
    q=state[:2] + self.dt * qd
    q = ( q + np.pi) % (2 * np.pi ) - np.pi # wrap angle

    self.state = np.squeeze((q[0],q[1],qd[0],qd[1]))

    reward = -np.mean(np.square(self.getJoints()[1,:]-self.target)) / 8.
    #reward = -np.mean(np.abs(self.getJoints()[1,:]-self.target)) / 8.


    #print reward
    terminal=False

    return (self.transformed_state(), reward, terminal, {})

  def getDynamicsMatrices(self):
    state = self.state

    m1=self.LINK_MASS_1
    m2=self.LINK_MASS_2
    l1=self.LINK_LENGTH_1
    l2=self.LINK_LENGTH_2

    inertias=np.array((m1,m2)) * (np.array((l1,l2)) ** 2 + 1e-05) / 3.0

    lg1=l1 / 2
    lg2=l2 / 2
    q1=state[0]-np.pi/2
    q2=state[1]
    q1d=state[2]
    q2d=state[3]
    c1=np.cos(q1)
    s2=np.sin(q2)
    c2=np.cos(q2)
    c12=np.cos(q1 + q2)
    M11=m1 * lg1 ** 2 + inertias[0] + m2 * (l1 ** 2 + lg2 ** 2 + 2 * l1 * lg2 * c2) + inertias[1]
    M12=m2 * (lg2 ** 2 + l1 * lg2 * c2) + inertias[1]
    M21=M12
    M22=m2 * lg2 ** 2 + inertias[1]

    # invdetM=1. / (M11 * (M22) - M12 * (M21))
    # invM = np.array([[M22,-M21],[-M12,M11]])
    # invM= invM * invdetM
    invM = np.linalg.inv(np.array([[M11,M12],[M21,M22]]))

    gravity=np.array((m1 * self.g * lg1 * c1 + m2 * self.g * (l1 * c1 + lg2 * c12),
                      m2 * self.g * lg2 * c12))

    coriolis=np.array((- m2 * l1 * lg2 * s2 * ((2 * (q1d) * (q2d) + q2d ** 2)),
                         m2 * l1 * lg2 * s2 * (q1d ** 2)))

    friction=np.array((self.friction_coeff[0] * q1d,
                       self.friction_coeff[1] * q2d))

    return gravity,coriolis,invM,friction

  def getJoints(self):
    state = self.state
    q1=state[0]-np.pi/2
    q2=state[1]
    q1d=state[2]
    q2d=state[3]

    xy1 = self.LINK_LENGTH_1 * np.array([np.cos(q1), np.sin(q1)])
    xy2 = xy1 + self.LINK_LENGTH_2 *np.array( [np.cos(q2+q1), np.sin(q2+q1)])
    
    xy1d = self.LINK_LENGTH_1 * np.array([q1d *np.cos(q1), -q1d*np.sin(q1)])
    xy2d = xy1d + self.LINK_LENGTH_2 * np.array([(q2d+q2)*np.cos(q1d+q1), -(q2d+q2)*np.sin(q1d+q1)])
    
    return np.array([xy1, xy2, xy1d, xy2d])

  def _render(self, mode='human', close=False):
    from gym.envs.classic_control import rendering
    if close:
      if self.viewer is not None:
        self.viewer.close()
      return

    s = self.state
    
    if self.viewer is None:
      self.viewer = rendering.Viewer(500,500) if mode=='human' else rendering.Viewer(30,30)
      self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)

    p1 = [-self.LINK_LENGTH_1 * np.cos(s[0]),
           self.LINK_LENGTH_1 * np.sin(s[0])]

    p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
          p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

    xys = np.array([[0,0], p1, p2])[:,::-1]
    thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]

    circ = self.viewer.draw_circle(.1)
    circ.set_color(1., .0, 0)
    circ.add_attr(rendering.Transform(self.target))

    for ((x,y),th) in zip(xys, thetas):
      l,r,t,b = 0, 1, .1, -.1
      jtransform = rendering.Transform(rotation=th, translation=(x,y))
      link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
      link.add_attr(jtransform)
      link.set_color(0,.8, .8)
      circ = self.viewer.draw_circle(.1)
      circ.set_color(.8, .8, 0)
      circ.add_attr(jtransform)

    self.viewer.render()
    if mode == 'rgb_array':
      return self.viewer.get_array()
    elif mode is 'human':
      pass