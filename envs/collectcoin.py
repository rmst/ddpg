from gym import core, spaces
import numpy as np

DEBUG = True

class CollectCoinEnv(core.Env):

  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 15
  }

  goal = [0.9, 0.8];
  coin = [0.2, 0.1];
  hole_y = 0.5
  hole_width = 0.2
  step = 0.05
  step_hole = 0.01
  radius_agent = 0.1
  
  # TODO: check numpy zeros / ones methods
  stateLB = np.zeros(4); # (x,y) of the agent + (x) of the center of the hole + boolean hasCoin
  stateUB = np.ones(4);
  actionLB = -np.ones(2);
  actionUB = np.ones(2)
  rewardLB = -1;
  rewardUB = 1;


  def __init__(self):
    
    self.observation_space = spaces.Box(self.stateLB, self.stateUB)
    self.action_space = spaces.Box(self.actionLB,self.actionUB)

    self.viewer = None

  def _reset(self):
    # corresponds to initstate

    bad = True
    while bad:
      agent = np.random.uniform(low=0, high=1, size=[2])
      bad = agent[2] > self.hole_y - self.radius_agent and agent[2] < self.hole_y + self.radius_agent

    hole_center = np.random.uniform(low=0, high=1, size = [1])
    has_coin = 0
    
    self.state = np.concatenate([agent,hole_center,has_coin]) # TODO: check numpy concatenate method

    return self.getFilteredState()

  def _step(self, action):
    # corrensponds to simulator

    self.state = np.clip(action, self.stateLB, self.stateUB)

    reward = None
    terminal = False

    return (self.getFilteredState(), reward, terminal, {})


  def getFilteredState(self):

    pixels = None # TODO: _render environment

    return pixels


  def _render(self, mode='human', close=False):

    # TODO: COPIED THIS IS FROM ANOTHER ENV -> ADAPT

    from gym.envs.classic_control import rendering
    if close:
      if self.viewer is not None:
        self.viewer.close()
      return

    s = self.state
    
    if self.viewer is None:
      self.viewer = rendering.Viewer(512,512) if mode=='human' else rendering.Viewer(512,512) # TODO: Adapt size
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


