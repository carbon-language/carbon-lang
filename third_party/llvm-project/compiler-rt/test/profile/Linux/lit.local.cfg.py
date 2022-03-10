import subprocess

def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)


def is_gold_linker_available():

  if not config.gold_executable:
    return False
  try:
    ld_cmd = subprocess.Popen([config.gold_executable, '--help'], stdout = subprocess.PIPE)
    ld_out = ld_cmd.stdout.read().decode()
    ld_cmd.wait()
  except:
    return False

  if not '-plugin' in ld_out:
    return False

  # config.clang is not guaranteed to be just the executable!
  clang_cmd = subprocess.Popen(" ".join([config.clang, '-fuse-ld=gold', '-xc', '-']),
                               shell=True,
                               universal_newlines = True,
                               stdin = subprocess.PIPE,
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE)
  clang_err = clang_cmd.communicate('int main() { return 0; }')[1]

  if not 'invalid linker' in clang_err:
    return True

  return False

root = getRoot(config)

if root.host_os not in ['Linux'] or not is_gold_linker_available():
  config.unsupported = True
