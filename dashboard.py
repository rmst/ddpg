#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

from IPython.display import display
from ipywidgets import widgets 
import matplotlib.pyplot as plt

import time
import thread
import numpy as np
import json
import shutil
import cStringIO
import webbrowser
import os
import subprocess
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('exdir',os.getenv('DB_EXDIR',''),'path containing output folders')
flags.DEFINE_boolean('browser',os.getenv('DB_BROWSER','True')=='True','create new jupyter browser tabs')

PORT_DB = "8007"
PORT_IP = "8008"
PORT_TB = "8009"

def main():
  print('Experiment root folder at: ' + FLAGS.exdir)

  free_port(PORT_DB)
  free_port(PORT_IP)
  free_port(PORT_TB)

  subprocess.Popen(['jupyter','notebook', '--no-browser', '--port='+PORT_IP, FLAGS.exdir])

  scriptdir = os.path.dirname(__file__)
  os.environ["DB_EXDIR"] = FLAGS.exdir
  os.environ["DB_BROWSER"] = 'True' if FLAGS.browser else 'False'
  os.system('jupyter notebook --port='+ PORT_DB + ' ' + scriptdir + ' --no-browser')
  if FLAGS.browser:
    webbrowser.open_new_tab('http://localhost:'+ PORT_DB+'/notebooks/dashboard.ipynb')

def free_port(port):
  import signal

  for lsof in ["lsof","/usr/sbin/lsof"]:
    try:        
      out = subprocess.check_output([lsof,"-t","-i:" + port])
      for l in out.splitlines():
        pid = int(l)
        os.kill(pid,signal.SIGTERM)
        # print("Killed process " + str(pid) + " to free port " + port)

      break
    except subprocess.CalledProcessError:
      pid = -1
    except OSError:
      pass

# TODO: remove ?
def load(pattern):
  import glob
  import numpy as np
  data = [np.load(f) for f in glob.glob(FLAGS.exdir+'/'+pattern)]
  return data


def dashboard(max=10):
  main_view = widgets.VBox()
  display(main_view)
  
  def loop():
    views = {}
    while True:
      try:
        views2 = {}
        todisplay = []
        i = 0
        dirs = os.listdir(FLAGS.exdir)
        dirs.sort(reverse=True)
        for e in dirs:
          if i == max: break
          i = i+1
          v = views[e] if e in views else ExpView(e)
          v.update()
          todisplay = todisplay + [v.view]
          views2[e] = v

        main_view.children = todisplay
        views = views2
        time.sleep(1.)
      except Exception as ex:
        print(ex)
        pass

  thread.start_new_thread(loop,())

class ExpView:

  def __init__(self, name):

    self.outdir = FLAGS.exdir + '/' + name

    style_hlink = '<style>.hlink{padding: 5px 10px 5px 10px;display:inline-block;}</style>'
    bname = widgets.HTML(style_hlink +
      '<a class=hlink target="_blank"'+
      'href="http://localhost:'+ PORT_IP +'/tree/'+ name +'"> '+
      name + ' </a> ')

    # self.env = widgets.Button()
    self.run_status = widgets.Button()

    killb = widgets.Button(description='kill')
    delb = widgets.Button(description='delete')
    killb.on_click(lambda _,self=self: exp_kill(self.outdir))

    def delf(_,self=self):
      self.delete()
      exp_delete(self.outdir)
    delb.on_click(delf)

    self.plot = widgets.Image(format='png')
    tbb = widgets.Button(description='tensorboard')
    tbbb = widgets.HTML(style_hlink+'<a class=hlink target="_blank" href="http://localhost:'+ PORT_TB +'"> (open) </a> ')
    
    def ontb(_,self=self):
      free_port(PORT_TB)
      subprocess.Popen(['tensorboard','--port', PORT_TB, '--logdir', self.outdir])
      if FLAGS.browser:
        webbrowser.open_new_tab('http://localhost:'+ PORT_TB)

    tbb.on_click(ontb)

    self.bar = widgets.HBox((bname, self.run_status,tbb,tbbb,killb,delb))

    self.view = widgets.VBox((self.bar,self.plot,widgets.HTML('<br><br>')))
    self.th_stop = False

    def loop_fig(self=self):
      while not self.th_stop:
        try:
          # update plot
          try:
            x = np.load(self.outdir+'/returns.npy')
          except:
            x = np.zeros([1,2])

          f,ax = plt.subplots()
          f.set_size_inches((15,2.5))
          f.set_tight_layout(True)
          ax.plot(x[:,0],x[:,1])
          #ax.plot(i,r)
          sio = cStringIO.StringIO()
          f.savefig(sio, format='png',dpi=60)

          self.plot.value = sio.getvalue()

          sio.close()
          plt.close(f)

        except:
          pass

    self.th = thread.start_new_thread(loop_fig,())

  def update(self):
    try:
      # update labels
      x = xread(self.outdir)
      job = x.get('job',False)
      if job:
        rt = 'job'
        jid = x['job_id']
        try:
          out = subprocess.check_output("squeue --job {} -o %%T".format(jid).split(' '),stderr=subprocess.STDOUT)      
          rs = out[6:]
          if rs == "": rs = "dead"
        except:
          rs = "dead"
      else:
        rt = 'local'
        rs = x['run_status']

      # flags = x.get('__flags') or x.get('flags')
      # self.env.description = flags.get('env','')
      self.run_status.description = rt + ": " + rs
    except:
      pass

  def delete(self):
    self.th_stop = True

def xwrite(path,data):
  with open(path+'/ezex.json','w+') as f:
    json.dump(data,f)

def xread(path):
  with open(path+'/ezex.json') as f:
    return json.load(f)

def exp_kill(outdir):
  ''' try to stop experiment slurm job with destination <outdir> '''
  try:
    x = xread(outdir)
    jid = x['job_id']
    cmd = 'scancel '+str(jid)
    subprocess.check_output(cmd,shell=True)
  except Exception:
    return False


def exp_delete(outdir):
  exp_kill(outdir)
  shutil.rmtree(outdir,ignore_errors=False)



if __name__ == '__main__':
  main()