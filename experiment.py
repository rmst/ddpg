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
    execute(main, FLAGS.outdir)

def create(run_folder,exfolder):
  ''' create unique experiment folder '''
  
  # generate unique name and create folder
  if not os.path.exists(exfolder): os.mkdir(exfolder)

  rf = os.path.basename(run_folder)
  dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
  basename =  dstr+'_'+rf+'_'+ FLAGS.tag
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
  info.update(FLAGS.__dict__)

  # create slurm script
  jscr = ("#!/bin/bash" + '\n' +
          "#SBATCH -o " + outdir + '/out' + '\n' +
          "#SBATCH --mem-per-cpu=" + "5000" + '\n' +
          "#SBATCH -n 4" + '\n' +
          "#SBATCH -t 24:00:00" + "\n" +
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


def execute(fun, outdir):
  # execute locally
  try:
    info = xread(outdir)
  except:
    info = {}

  t_start = time.time()
  try:
    info['start_time'] = t_start
    info['run_status'] = 'running'
    xwrite(outdir,info)
    
    fun()

    info['run_status'] = 'finished'
  except:
    import traceback
    traceback.print_exc()
    info['run_status'] = 'aborted'
    print("aborted")
  finally:
    elapsed = time.time() - t_start
    info['end_time'] = time.time()
    xwrite(outdir,info)
    print('elapsed seconds: ' + str(elapsed))
    if elapsed <= FLAGS.autodel*60.:
      print('delete because runtime < ' + str(FLAGS.autodel) + " minutes")
      shutil.rmtree(outdir,ignore_errors=False)


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
