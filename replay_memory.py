import numpy as np
import random

# TODO: make this list-based (i.e. variable sized)
class ReplayMemory:
  def __init__(self, size, dimO, dimA, dtype=np.float32):
    self.size = size
    so = np.concatenate(np.atleast_1d(size,dimO),axis=0)
    sa = np.concatenate(np.atleast_1d(size,dimA),axis=0)
    self.observations = np.empty(so, dtype = dtype)
    self.actions = np.empty(sa, dtype = np.float32)
    self.rewards = np.empty(size, dtype = np.float32)
    self.terminals = np.empty(size, dtype = np.bool)
    self.info = np.empty(size,dtype = object)

    self.n = 0
    self.i = -1

  def reset(self):
    self.n = 0
    self.i = -1

  def enqueue(self, observation,terminal,action,reward,info=None):
    
    self.i = (self.i + 1) % self.size
    
    self.observations[self.i, ...] = observation
    self.terminals[self.i] = terminal # tells whether this observation is the last
    
    self.actions[self.i,...] = action
    self.rewards[self.i] = reward
    
    self.info[self.i,...] = info
    
    self.n = min(self.size-1, self.n + 1)
    

  
  def minibatch(self,size):
    # sample uniform random indexes
    # indices = np.zeros(size,dtype=np.int)
    # for k in range(size):
    #   # find random index 
    #   invalid = True
    #   while invalid:
    #     # sample index ignore wrapping over buffer
    #     i = random.randint(0, self.n-2)
    #     # if i-th sample is current one or is terminal: get new index
    #     if i != self.i and not self.terminals[i]:
    #       invalid = False
      
    #   indices[k] = i
    #print i
    #print self.i

    indices = np.random.randint(0, self.n-2, size)
  
    o = self.observations[indices,...]
    a = self.actions[indices]
    r = self.rewards[indices]
    t = self.terminals[indices]
    o2 = self.observations[indices+1,...]
    # t2 = self.terminals[indices+1] # to return t2 instead of t was a mistake
    info = self.info[indices,...]

    return o, a, r, t, o2, info
    
  def __repr__(self):
    indices = range(0,self.n)
    o = self.observations[indices,...]
    a = self.actions[indices]
    r = self.rewards[indices]
    t = self.terminals[indices]
    info = self.info[indices,...]

    s = """
    OBSERVATIONS
    {}

    ACTIONS
    {}

    REWARDS
    {}

    TERMINALS
    {}
    """.format(o,a,r,t)

    return s

# TODO: relocate test
if __name__ == '__main__':
  s = 100
  rm = ReplayMemory(s,1,1)
  
  for i in range(0,100,1):
    rm.enqueue(i,i%3==0,i,i,i)
  
  for i in range(1000):
    o, a, r, t, o2, info = rm.minibatch(10)
    assert all(o == o2-1),"error: o and o2"
    assert all(o != s-1) , "error: o wrap over rm. o = "+str(o) 
    assert all(o2 != 0) , "error: o2 wrap over rm"
