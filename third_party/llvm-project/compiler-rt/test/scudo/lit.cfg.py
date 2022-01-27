# -*- Python -*-

import os

# Setup config name.
config.name = 'Scudo' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Path to the shared library
shared_libscudo = os.path.join(config.compiler_rt_libdir, "libclang_rt.scudo%s.so" % config.target_suffix)
shared_minlibscudo = os.path.join(config.compiler_rt_libdir, "libclang_rt.scudo_minimal%s.so" % config.target_suffix)

# Test suffixes.
config.suffixes = ['.c', '.cpp', '.test']

# C & CXX flags.
c_flags = ([config.target_cflags] +
           ["-pthread",
           "-fPIE",
           "-pie",
           "-O0",
           "-UNDEBUG",
           "-ldl",
           "-Wl,--gc-sections"])

# Android doesn't want -lrt.
if not config.android:
  c_flags += ["-lrt"]

cxx_flags = (c_flags + config.cxx_mode_flags + ["-std=c++11"])

scudo_flags = ["-fsanitize=scudo"]

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(("%clang_scudo ", build_invocation(c_flags + scudo_flags)))
config.substitutions.append(("%clangxx_scudo ", build_invocation(cxx_flags + scudo_flags)))
config.substitutions.append(("%shared_libscudo", shared_libscudo))
config.substitutions.append(("%shared_minlibscudo", shared_minlibscudo))

# Platform-specific default SCUDO_OPTIONS for lit tests.
default_scudo_opts = ''
if config.android:
  # Android defaults to abort_on_error=1, which doesn't work for us.
  default_scudo_opts = 'abort_on_error=0'

# Disable GWP-ASan for scudo internal tests.
if config.gwp_asan:
  config.environment['GWP_ASAN_OPTIONS'] = 'Enabled=0'

if default_scudo_opts:
  config.environment['SCUDO_OPTIONS'] = default_scudo_opts
  default_scudo_opts += ':'
config.substitutions.append(('%env_scudo_opts=',
                             'env SCUDO_OPTIONS=' + default_scudo_opts))

# Hardened Allocator tests are currently supported on Linux only.
if config.host_os not in ['Linux']:
   config.unsupported = True
