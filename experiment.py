
import time
from datetime import datetime
import subprocess
import shutil
import json
import os
import sys

def xwrite(path,data):
  """
  write info file

  The info file contains information about the status and parameters of the experiment.
  It can be used to display experiment details in a dashboard.
  """
  with open(path+'/ezex.json','w+') as f:
    json.dump(data,f)

def xread(path):
  with open(path+'/ezex.json') as f:
    return json.load(f)

def _parse_all(parser):
  #import argparse
  args,unknown = parser.parse_known_args()
  d = {}
  d.update(args.__dict__)
  for arg in unknown:
    if arg.startswith('--'):
      opt = arg[2:]
      d[opt] = None
    elif d[opt] is None:
      d[opt] = eval(arg)

  return d

def parse(parser,src=None):
  """
  WARNING: this might involve restart of the script 
  """

  if not src:
    # get filename of caller script
    import inspect
    src = inspect.getfile(sys._getframe(1))

  # experiment arguments
  rn = parser.add_argument_group('run')
  rn.add_argument('--dst',default=os.path.split(src)[0][:-2]+'-runs',help='destination folder for results')
  rn.add_argument('--tag',default='',help='name tag for experiment')
  rn.add_argument('--job',action='store_true',help='submit slurm job')
  rn.add_argument('--nvd',action='store_true',help='run on Nvidia-Node')
  rn.add_argument('-c','--copy',action='store_true',help='copy contents of script folder to dst')
  rn.add_argument('--del',default=0.,type=float,help='auto delete experiments terminating before DEL minutes')
  rn.add_argument('--xvfb',action='store_true',help='run inside xvfb')
  rn.add_argument('--exe',action='count',help='internal command')

  kwargs = _parse_all(parser)
  if kwargs['exe']==2:
    return kwargs
  elif kwargs['exe']==1:
    execute(os.getcwd(),**kwargs) # WARNING: this involves restart of the script 
    sys.exit()
  else:
    submit(src,**kwargs) # WARNING: this involves restart of the script 
    sys.exit()

def create(run_folder,exfolder,tag='',copy=False,**kwargs):
  ''' create unique experiment folder '''
  
  # generate unique name and create folder
  if not os.path.exists(exfolder): os.mkdir(exfolder)

  rf = os.path.basename(run_folder)
  dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
  basename =  dstr+'_'+rf+'_'+tag
  name = basename
  i = 1
  while name in os.listdir(exfolder):
    name = basename + '_' + str(i)
    i = i+1
    if i > 100:
      raise RuntimeError('Could not create unique experiment folder')
      
  path = exfolder+'/'+name
  os.mkdir(path)

  # copy program to folder
  if copy:
    rcopy(run_folder,path,symlinks=True,ignore='.*')

  return path


def submit(src,dst,job=False,prerun='',copy=False,**kwargs):
  """
  create new folder for results in <dst> and then either run <src> locally or submit a slurm job
  """
  src = os.path.abspath(src)
  dst = os.path.abspath(dst)
  folder,script = os.path.split(src)
  path = create(folder,dst,copy=copy,**kwargs)
  script = (path + '/' + script) if copy else src
  python = sys.executable


  # xvfb = 'xvfb-run -n 0 -s "-screen 0 1400x900x24" ' if kwargs['exe']==2 and kwargs['xvfb'] else ''
 

  run_cmd = ('cd '+ path +'; '+
    prerun +' '+python+ ' ' +script+ ' ' + ' '.join(sys.argv[1:]) + ' --exe')

  info = {}
  info['run_cmd'] = run_cmd
  info.update(kwargs)

  if job:
    # create slurm script
    nvd = kwargs.get('nvd',False)
    jscr = ("#!/bin/bash" + '\n' +
            "#SBATCH -o " + path + '/out' + '\n' +
            "#SBATCH --mem-per-cpu=" + "5000" + '\n' +
            "#SBATCH -n 4" + '\n' +
            "#SBATCH -t 24:00:00" + "\n" +
            ('#SBATCH -C nvd \n' if nvd else '') +
            run_cmd)

    with open(path+"/slurmjob","w") as f:
      f.write(jscr)

    cmd = "sbatch " + path + "/slurmjob"

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
    info['run_type'] = 'job'
    info['run_status'] = 'pending'
    xwrite(path,info)

  else:
    info['job_id'] = -1
    info['run_type'] = 'local'
    xwrite(path,info)

    execute(path,**kwargs)


def execute(path,**kwargs):
  # execute locally
  info = xread(path)
  t_start = time.time()
  try:
    run_cmd = info['run_cmd'] + ' --exe'
    info['run_cmd'] = run_cmd
    info['start_time'] = t_start
    info['run_status'] = 'running'
    xwrite(path,info)
    
    print(run_cmd)
    #os.system(cmd)
    #subprocess.check_output(run_cmd,shell=True,stderr=subprocess.STDOUT)
    subprocess.call(run_cmd,shell=True,stdout=sys.stderr)
    info['run_status'] = 'finished'
  except Exception as e:
    print(e)
    info['run_status'] = 'aborted'
    print("aborted")
  finally:
    elapsed = time.time() - t_start
    info['end_time'] = time.time()
    xwrite(path,info)
    print('elapsed seconds: ' + str(elapsed))
    if elapsed <= kwargs.get('del',0)*60:
      print('delete because del ' + str((kwargs.get('del',0))) + " minutes")
      shutil.rmtree(path,ignore_errors=False)

def kill(path):
  ''' try to stop experiment slurm job with destination <path> '''
  try:
    x = xread(path)
    jid = x['job_id']
    cmd = 'scancel '+str(jid)
    subprocess.check_output(cmd,shell=True)
  except Exception:
    return False


def delete(path):
  kill(path)
  shutil.rmtree(path,ignore_errors=False)



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
