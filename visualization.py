'''
contains visualization routines using matplotlib
'''
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import cStringIO

def as2d(a):
  if np.ndim(a) == 0:
    return a[np.newaxis,np.newaxis]
  elif np.ndim(a) == 1:
    return a[:,np.newaxis]
  else:
    return a


import tensorflow as tf
from contextlib import contextmanager
@contextmanager
def axes(tag,summary_writer,step):
  """ usage: with axes(...) as ax: ax.plot([1,2,3]) """
  f,ax = plt.subplots()
  f.set_size_inches((5,5))
  f.set_tight_layout(True)

  yield ax

  sio = cStringIO.StringIO()
  f.savefig(sio, format='png',dpi=120)
  val = sio.getvalue()
  s = tf.Summary(value=[tf.Summary.Value(tag=tag+'/'+str(step),image=tf.Summary.Image(encoded_image_string=val))])
  summary_writer.add_summary(s,step)

  sio.close()
  plt.close(f)

class Fig:
  def __init__(self):
    #self.name = name
    self.f,self.ax = plt.subplots()
    self.f.set_size_inches((5,5))
    self.f.set_tight_layout(True)

  def multiplot(self,ys,titles=None):
    if not type(ys) in (list,tuple):
      ys = (ys,)
    
    if type(titles) is str:
      titles = (titles,) 
      
    t2 = []
    ys2 = []
    for j in range(len(ys)):
      ys2.append(as2d(ys[j]))
      d = np.size(ys2[j],1)
      t = titles[j] if titles is not None else 'xyzabcdef'[j]
      if d == 1:
        t2.append(t)
      else:
        t2 = t2 + [ t + '[' + str(i) + ']' for i in range(d)]
      
    self.ax.plot(np.concatenate(ys2,1))
    self.ax.legend(t2)
    return self

  def to_tfsummary(self,tag):
    sio = cStringIO.StringIO()
    self.f.savefig(sio, format='png',dpi=120)
    val = sio.getvalue()
    s = tf.Summary(value=[tf.Summary.Value(tag=tag,
      image=tf.Summary.Image(encoded_image_string=val))])
    sio.close()
    plt.close(self.f)
    return s



# TODO: better density estimations
def hist(a,bins = 50,range=None):
  '''
  a is an array of size (batch x dim)
  for every dim histograms will be seperately computed
  '''
  a = as2d(a)
  if range:
    h = range[1]
    l = range[0]
  else:
    h = a.max()
    l = a.min()
    
  d = np.size(a,1)
  #n = np.size(a,0)
  res = np.empty([bins,d])
  for i in xrange(d):
    res[::-1,i],e = np.histogram(a[:,i],bins,(l,h),density=True)
    
  #print e
  #res = res/n*bins
  #plt.figure()
  plt.imshow(res,extent=[0,d,l,h],aspect='auto',interpolation='none')
  #plt.matshow(res)
  plt.colorbar()
  
def hist_time(a,steps = 20,bins=50):
  '''
  a is an array of size (batch) which will be reshaped (folded)
  and then passed to 'hist'
  '''
  n = np.size(a,0)
  m = n/steps
  res = np.empty([m,steps])
  for i in xrange(steps):
    res[:,i] = a[i*m : (i+1)*m]
    
  hist(res,bins)
    
def plot(ys,titles=None):

  if not type(ys) in (list,tuple):
    ys = (ys,)
  
  if type(titles) is str:
    titles = (titles,) 
    
  t2 = []
  ys2 = []
  for j in range(len(ys)):
    ys2.append(as2d(ys[j]))
    d = np.size(ys2[j],1)
    t = titles[j] if titles is not None else 'xyzabcdef'[j]
    if d == 1:
      t2.append(t)
    else:
      t2 = t2 + [ t + '[' + str(i) + ']' for i in range(d)]
      
  plt.plot(np.concatenate(ys2,1))
  plt.legend(t2)


def hists2d(xs,y,titlex='x',titley='y',bins=50):
  xs = as2d(xs)
  d = np.size(xs,1)
  t = ['p('+titlex+'['+str(i)+'],'+titley+')' for i in range(d)]
  ax = figures(t)  
  for i in range(d):
    ax.next()
    plt.hist2d(xs[:,i],y,bins=bins)
    plt.colorbar()
 

 
# CODE FRAGMENTS:

# # save trajectory
# self.rm_log.enqueue(self.observation,term,self.action,rew)
# if term and np.random.binomial(1,0.1):
#   # plot to tensorboard
#   a = self.rm_log.actions[0:self.rm_log.i , ...]
#   o = self.rm_log.observations[0:self.rm_log.i, ...]
#   s = vis.Fig().multiplot((a,o),('a','o')).to_tfsummary('t'+'{0:08d}'.format(self.t)+'/traj')
#   self.writer.add_summary(s,self.t)
#   self.rm_log.reset()
