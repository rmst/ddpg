import numpy as np
import gym

import doublelink
DoubleLinkEnv = doublelink.DoubleLinkEnv

def filterEnv(env):
  """ create a new filtered environment class instance with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  class FilteredEnv(type(env)):
    a1 = np.ones_like(acsp.high)
    o1 = np.ones_like(obsp.high)
    action_space = gym.spaces.Box(-a1,a1)
    observation_space = gym.spaces.Box(-o1, o1)

    def _step(self,action):
      
      h = acsp.high
      l = acsp.low
      sc = h-l
      c = (h+l)/2.

      ac = np.clip(sc*action+c,l,h)

      obs, reward, term, info = env.step(self,ac) # super function

      h = obsp.high
      l = obsp.low
      sc = h-l
      c = (h+l)/2.

      obs = (obs-c) / sc

      return obs, reward, term, info


  fenv = FilteredEnv()
  fenv.__dict__.update(env.__dict__)