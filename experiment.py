import time
from datetime import datetime
import subprocess
import shutil
import json
import os
import sys

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '', 'destination folder for results')
flags.DEFINE_boolean('copy',False, 'copy code folder to outdir')
flags.DEFINE_string('tag', '', 'name tag for experiment')
flags.DEFINE_boolean('job',False, 'submit slurm job')
flags.DEFINE_boolean('nvd',False, 'run on Nvidia-Node')
flags.DEFINE_float('autodel', 0., 'auto delete experiments terminating before DEL minutes')
flags.DEFINE_boolean('gdb',False, 'open gdb on error')



def run(main=None):
  argv = sys.argv
  f = flags.FLAGS
  f._parse_flags()

  script = sys.modules['__main__'].__file__
  scriptdir, scriptfile = os.path.split(script)

  if FLAGS.outdir[-1] == '+':
    exdir = FLAGS.outdir[:-1]
    outdir = create(scriptdir, exdir)
    i = argv.index('--outdir')
    argv[i+1] = outdir # TODO: handle --outdir=... case
    FLAGS.outdir = outdir

  elif not os.path.exists(FLAGS.outdir):
    os.mkdir(FLAGS.outdir)

  print("outdir: " + FLAGS.outdir)

  script = (FLAGS.outdir + '/' + scriptfile) if FLAGS.copy else script
  argv[0] = script

  if FLAGS.job:
    argv.remove('--job')
    submit(argv, FLAGS.outdir)
  else:
    main = main or sys.modules['__main__'].main
    Executor(main, FLAGS.outdir).execute()

def create(run_folder,exfolder):
  ''' create unique experiment folder '''
  
  # generate unique name and create folder
  if not os.path.exists(exfolder): os.mkdir(exfolder)

  dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')

  # rf = os.path.basename(run_folder)
  #basename =  dstr+'_'+rf+'_'+ FLAGS.tag
  basename =  '_'.join([dstr,FLAGS.env,FLAGS.tag])

  name = basename
  i = 1
  while name in os.listdir(exfolder):
    name = basename + '_' + str(i)
    i = i+1
    if i > 100:
      raise RuntimeError('Could not create unique experiment folder')
      
  path = os.path.join(exfolder, name)
  os.mkdir(path)

  # copy program to folder
  if FLAGS.copy:
    rcopy(run_folder,path,symlinks=True,ignore='.*')

  return path


def submit(argv, outdir):
  """
  submit a slurm job
  """
  prerun = ''
  python = sys.executable

  run_cmd = ' '.join(['cd', outdir, ';', prerun, python] + argv)

  info = {}
  info['run_cmd'] = run_cmd
  info['flags'] = FLAGS.__flags

  # create slurm script
  jscr = ("#!/bin/bash" + '\n' +
          "#SBATCH -o " + outdir + '/out' + '\n' +
          "#SBATCH --mem-per-cpu=" + "5000" + '\n' +
          "#SBATCH -n 4" + '\n' +
          "#SBATCH -t 24:00:00" + "\n" +
          "#SBATCH --exclusive" + "\n" + 
          ('#SBATCH -C nvd \n' if FLAGS.nvd else '') + "\n" +
          "source ~/.bashrc" + "\n" +
          run_cmd)

  with open(outdir+"/slurmjob","w") as f:
    f.write(jscr)

  cmd = "sbatch " + outdir + "/slurmjob"

  # submit slurm job
  out = subprocess.check_output(cmd,shell=True)
  print("SUBMIT: \n" + out)

  # match job id
  import re
  match = re.search('Submitted batch job (\d*)',out)
  if not match:
    raise RuntimeError('SLURM submit problem')
  jid = match.group(1)

  # write experiment info file
  info['job_id'] = jid
  info['job'] = True
  info['run_status'] = 'pending'
  xwrite(outdir,info)


# register clean up before anybody else does
import atexit, signal
on_exit_do = []
def on_exit():
  if on_exit_do:
    on_exit_do[0]()
atexit.register(on_exit)


class Executor:
  def __init__(self, main, outdir):
    signal.signal(signal.SIGINT, self.on_kill)
    signal.signal(signal.SIGTERM, self.on_kill)
    on_exit_do.append(self.on_exit)

    self.main = main
    self.outdir = outdir

  def on_exit(self):
    elapsed = time.time() - self.t_start
    self.info['end_time'] = time.time()
    xwrite(self.outdir, self.info)
    print('Elapsed seconds: {}\n'.format(elapsed))
    if not self.info.get('job',False) and elapsed <= FLAGS.autodel*60.:
      print('Deleted output folder because runtime < ' + str(FLAGS.autodel) + " minutes")
      shutil.rmtree(self.outdir,ignore_errors=False)

  def on_kill(self,*args):
    self.info['run_status'] = 'aborted'
    print("Experiment aborted")
    sys.exit()

  def execute(self):
    """ execute locally """
    try:
      self.info = xread(self.outdir)
    except:
      self.info = {}

    self.t_start = time.time()

    try:
      self.info['start_time'] = self.t_start
      self.info['run_status'] = 'running'
      xwrite(self.outdir,self.info)

      self.main()

      self.info['run_status'] = 'finished'
    except Exception as e:
      self.on_error(e)

  def on_error(self,e):
    self.info['run_status'] = 'error'

    # construct meaningful traceback
    import traceback, sys, code
    type, value, tb = sys.exc_info()
    tbs = []
    tbm = []
    while tb is not None:
      stb = traceback.extract_tb(tb)
      filename = stb[0][0]
      tdir,fn = os.path.split(filename)
      maindir = os.path.dirname(sys.modules['__main__'].__file__)
      if tdir == maindir:
        tbs.append(tb)
        tbm.append("{} : {} : {} : {}".format(fn,stb[0][1],stb[0][2],stb[0][3]))

      tb = tb.tb_next

    # print custom traceback
    print("\n\n- Experiment error traceback (use --gdb to debug) -\n")
    print("\n".join(tbm)+"\n")
    print("{}: {}\n".format(e.__class__.__name__,e))

    # enter interactive mode (i.e. post mortem)
    if FLAGS.gdb:
      print("\nPost Mortem:")
      for i in reversed(range(len(tbs))):
        print("Level {}: {}".format(i,tbm[i]))
        # pdb.post_mortem(tbs[i])
        frame = tbs[i].tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(banner="", local=ns)
        print("\n")



def xwrite(path,data):
  with open(path+'/ezex.json','w+') as f:
    json.dump(data,f)

def xread(path):
  with open(path+'/ezex.json') as f:
    return json.load(f)


# Util
#

def rcopy(src, dst, symlinks = False, ignore = None):
  import shutil
  ign = shutil.ignore_patterns(ignore)
  copytree(src,dst,symlinks,ign)

def copytree(src, dst, symlinks = False, ignore = None):
  import os
  import shutil
  import stat
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)


try: import cPickle as pickle
except: import pickle
def add(root,val):
  root = os.path.abspath(root)
  m = (os.listdir(root) or ['-1']).sort()[-1]
  n = '{0:08d}'.format(int(m)+1)
  n = root+'/'+n
  with open(n,'wb') as f:
    pickle.dump(val,f)
def lst(root): return os.listdir(root).sort()
def get(path):
  with open(path,'rb') as f:
    return pickle.load(f)
