#!/usr/bin/env python

import experiment
import argparse
parser = argparse.ArgumentParser(prog='DDPG')
p_hp = parser.add_argument_group('hyperparameters')
# ...

args = experiment.parse(parser)

if args.get('xvfb',False):
  from pyvirtualdisplay import Display
  xvfb=Display(size=(1400, 900),color_depth=24).start()

import ddpg
import gym
import numpy as np
import doublelink
def norm_step(env,a):
  # normalize states and actions
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')

  h = acsp.high
  l = acsp.low
  sc = h-l
  c = (h+l)/2.

  obs, reward, term, info = env.step(np.clip(sc*a+c,l,h))

  h = obsp.high
  l = obsp.low
  sc = h-l
  c = (h+l)/2.

  obs = (obs-c) / sc

  return obs, reward, term, info


class Experiment:

  def run(self,t_train = 1000000,t_warmup=20000,f_test=20,env='Pendulum-v0',render=False,**kwargs):
    self.t_warmup = t_warmup
    self.t_log = 103
    #self.env = gym.make(s.env)
    #from gym.envs.classic_control.dl import DoubleLinkEnv
    self.env = doublelink.DoubleLinkEnv()
    # self.env.monitor.start('./monitor/',video_callable=lambda _: False) TODO: fix on cluster
    dimO = self.env.observation_space.shape
    dimA = self.env.action_space.shape
    print('dimO: '+str(dimO) +'  dimA: '+str(dimA))
    # agent
    self.agent = ddpg.Agent(dimO=dimO,dimA=dimA,**kwargs)

    # main loop
    while self.agent.t < t_train:
      # train
      for i in xrange(f_test): self.run_episode(test=False)

      # test
      R = np.mean([self.run_episode(test=True,render=render) for _ in range(5)])
      print('Average return '+str(R)+ ' after '+str(self.agent.t)+' timesteps of training')


  def run_episode(self,test=True,t_max=200,render=False):
    self.env.monitor.configure(lambda _: False) # TODO: capture video only at test time
    observation = self.env.reset()
    self.agent.reset(observation)
    R = 0 # return
    t = 0
    term = False
    while not term:
      if render: self.env.render(mode='human')

      action = self.agent.act(test=test,logging=self.agent.t%self.t_log==0)

      observation, reward, term, info = norm_step(self.env,action)
      term = (t >= t_max) or term

      self.agent.observe(reward,term,observation,test=test)

      if not test:
        if self.agent.t > self.t_warmup:
          self.agent.train(logging=self.agent.t%self.t_log==0)

      R = R + reward
      t = t + 1

    return R


Experiment().run(**args)
