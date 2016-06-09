
# recursive copy
def copy(src, dst, symlinks = False, ignore = None):
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


# forcefully free port
def free_port(port):
  import signal
  import subprocess
  import os
  for lsof in ["lsof","/usr/sbin/lsof"]:
    try:        
      out = subprocess.check_output([lsof,"-t","-i:"+str(port)])
      for l in out.splitlines():
        pid = int(l)
        os.kill(pid,signal.SIGTERM)
        print("Killed process " + str(pid) + " to free port " + str(port))

      break
    except subprocess.CalledProcessError:
      pid = -1
    except OSError:
      pass