import numpy as np
import gym

def makeFilteredEnv(env):
  """ crate a new environment class with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  env_type = type(env)

  class FilteredEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      a1 = np.ones_like(acsp.high)
      self.action_space = gym.spaces.Box(-a1,a1)

      self.filter_obs = np.any(obsp.high < 1e10)

      if self.filter_obs:
        o1 = np.ones_like(obsp.high)
        self.observation_space = gym.spaces.Box(-o1, o1)
      

    def step(self,action):
      
      h = acsp.high
      l = acsp.low
      sc = h-l
      c = (h+l)/2.

      ac = np.clip(sc*action+c,l,h)

      obs, reward, term, info = env_type.step(self,ac) # super function

      if self.filter_obs:
        h = obsp.high
        l = obsp.low
        sc = h-l
        c = (h+l)/2.
        obs = (obs-c) / sc


      # reward -= 1 # exploration in the face of uncertainty

      # TODO: remove
      # obs[6] = obs[6] / 40.
      # obs[7] = obs[7] / 40.

      return obs, reward, term, info

  fenv = FilteredEnv()

  print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
  print('True state space: ' + str(obsp.low) + ', ' + str(obsp.high))
  print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
  print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

  return fenv