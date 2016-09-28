from gym import core, spaces
import numpy as np

DEBUG = True

class PuddleworldEnv(core.Env):

  metadata = {
    'render.modes': [],
    'video.frames_per_second' : 15
  }

  goal = [1.,1.]
  step = .05

  p1 = [[.1, .75],[.45, .75]]  # Centers of the first puddle
  p2 = [[.45, .4],[.45, .8]]  # Centers of the second puddle
  p3 = [[.8, .2],[.8, .5]]  # Centers of the third puddle
  p4 = [[.7, .75],[.7, .8]]  # Centers of the fourth puddle

  radius = .1

  stateLB = np.array([0., 0])
  stateUB = np.array([1., 1])

  actionLB = -np.array([.05,.05])
  actionUB =  np.array([.05,.05])

  rewardLB = -41
  rewardUB = -1


  def __init__(self):
    self.observation_space = spaces.Box(self.stateLB, self.stateUB) # TODO: rbf features

    self.action_space = spaces.Box(self.actionLB,self.actionUB)

    self.viewer = None

    # rbfs
    self.n_centers = 10.
    self.rbf_centers = np.transpose(np.repeat(np.atleast_2d(np.linspace(0, 1, self.n_centers)),2,axis=0))


  def _reset(self):
    self.state = np.random.uniform(low=0, high=1, size=[2])

    return self.getFilteredState()

  def _step(self, action):
    a2 = np.linalg.norm(action) * self.step # TODO: normalizing correctly?

    noise = np.random.multivariate_normal([0,0],[.0001, 0],[0, .0001])
    s2 = self.state + a2 + noise
    self.state = np.clip(s2, self.stateLB, self.stateUB)

    reward = self.puddlepenalty() - 1

    terminal = np.sum(self.state) >= 1.9

    return (self.getFilteredState(), reward, terminal, {})


  def getFilteredState(self):
    phi = np.exp( - ( (self.stateUB - self.stateLB) / self.n_centers * (self.state - self.rbf_centers))**2)
    phi2 = np.reshape(phi, -1)
    return phi2


  def puddlepenalty(self):
    factor = 400

    if self.state[0] > self.p1[1][0]:
      d1 = np.linalg.norm(np.transpose(self.state) - self.p1[1])
    elif self.state[0] < self.p1[0][0]:
      d1 = np.linalg.norm(np.transpose(self.state) - self.p1[0])
    else:
      d1 = np.abs(self.state[1] - self.p1[0][1])

    if self.state[1] > self.p2[1][1]:
      d2 = np.linalg.norm(np.transpose(self.state) - self.p2[1])
    elif self.state[1] < self.p2[0][1]:
      d2 = np.linalg.norm(np.transpose(self.state) - self.p2[0])
    else:
      d2 = np.abs(self.state[0] - self.p2[0][0])

    if self.state[1] > self.p3[1][1]:
      d3 = np.linalg.norm(np.transpose(self.state) - self.p3[1])
    elif self.state[1] < self.p3[0][1]:
      d3 = np.linalg.norm(np.transpose(self.state) - self.p3[0])
    else:
      d3 = np.abs(self.state[0] - self.p3[0][0])

    if self.state[1] > self.p4[1][1]:
      d4 = np.linalg.norm(np.transpose(self.state) - self.p4[1])
    elif self.state[1] < self.p4[0][1]:
      d4 = np.linalg.norm(np.transpose(self.state) - self.p4[0])
    else:
      d4 = np.abs(self.state[0] - self.p4[0][0])

    min_distance_from_puddle = np.min([d1, d2, d3, d4])

    if min_distance_from_puddle <= self.radius:
      reward = - factor * (self.radius - min_distance_from_puddle)
    else:
      reward = 0

    return reward


if __name__ == '__main__':
	p = PuddleworldEnv()
	print(p._reset())