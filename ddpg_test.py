import pytest

def test_ddpg_replay(tmpdir):
  import ddpg
  import numpy as np
  np.set_printoptions(threshold=np.nan)

  ddpg.FLAGS.warmup = 10000
  ddpg.FLAGS.outdir = tmpdir.strpath
  # test replay memory
  a = ddpg.Agent([1],[1])
  a.reset([0])
  T = 10
  actions = []
  for t in range(0,T):
    actions.append(a.act())
    a.observe(t,False,[t+1])

  # print(a.rm)
  # print(a.rm.minibatch(5))
  # assert False



def test_replay_memory():
  from replay_memory import ReplayMemory
  s = 100
  rm = ReplayMemory(s,1,1)
  
  for i in range(0,100,1):
    rm.enqueue(i,i%3==0,i,i,i)
  
  for i in range(1000):
    o, a, r, o2, t2, info = rm.minibatch(10)
    assert all(o == o2-1),"error: o and o2"
    assert all(o != s-1) , "error: o wrap over rm. o = "+str(o) 
    assert all(o2 != 0) , "error: o2 wrap over rm"