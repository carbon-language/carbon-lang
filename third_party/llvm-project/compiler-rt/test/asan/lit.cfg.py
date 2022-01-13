# -*- Python -*-

import os
import platform
import re

import lit.formats

# Get shlex.quote if available (added in 3.3), and fall back to pipes.quote if
# it's not available.
try:
  import shlex
  sh_quote = shlex.quote
except:
  import pipes
  sh_quote = pipes.quote

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

def push_dynamic_library_lookup_path(config, new_path):
  if platform.system() == 'Windows':
    dynamic_library_lookup_var = 'PATH'
  elif platform.system() == 'Darwin':
    dynamic_library_lookup_var = 'DYLD_LIBRARY_PATH'
  else:
    dynamic_library_lookup_var = 'LD_LIBRARY_PATH'

  new_ld_library_path = os.path.pathsep.join(
    (new_path, config.environment.get(dynamic_library_lookup_var, '')))
  config.environment[dynamic_library_lookup_var] = new_ld_library_path

  if platform.system() == 'FreeBSD':
    dynamic_library_lookup_var = 'LD_32_LIBRARY_PATH'
    new_ld_32_library_path = os.path.pathsep.join(
      (new_path, config.environment.get(dynamic_library_lookup_var, '')))
    config.environment[dynamic_library_lookup_var] = new_ld_32_library_path

  if platform.system() == 'SunOS':
    dynamic_library_lookup_var = 'LD_LIBRARY_PATH_32'
    new_ld_library_path_32 = os.path.pathsep.join(
      (new_path, config.environment.get(dynamic_library_lookup_var, '')))
    config.environment[dynamic_library_lookup_var] = new_ld_library_path_32

    dynamic_library_lookup_var = 'LD_LIBRARY_PATH_64'
    new_ld_library_path_64 = os.path.pathsep.join(
      (new_path, config.environment.get(dynamic_library_lookup_var, '')))
    config.environment[dynamic_library_lookup_var] = new_ld_library_path_64

# Setup config name.
config.name = 'AddressSanitizer' + config.name_suffix

# Platform-specific default ASAN_OPTIONS for lit tests.
default_asan_opts = list(config.default_sanitizer_opts)

# On Darwin, leak checking is not enabled by default. Enable for x86_64
# tests to prevent regressions
if config.host_os == 'Darwin' and config.target_arch == 'x86_64':
  default_asan_opts += ['detect_leaks=1']

default_asan_opts_str = ':'.join(default_asan_opts)
if default_asan_opts_str:
  config.environment['ASAN_OPTIONS'] = default_asan_opts_str
  default_asan_opts_str += ':'
config.substitutions.append(('%env_asan_opts=',
                             'env ASAN_OPTIONS=' + default_asan_opts_str))

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

if config.host_os not in ['FreeBSD', 'NetBSD']:
  libdl_flag = "-ldl"
else:
  libdl_flag = ""

# GCC-ASan doesn't link in all the necessary libraries automatically, so
# we have to do it ourselves.
if config.compiler_id == 'GNU':
  extra_link_flags = ["-pthread", "-lstdc++", libdl_flag]
else:
  extra_link_flags = []

# Setup default compiler flags used with -fsanitize=address option.
# FIXME: Review the set of required flags and check if it can be reduced.
target_cflags = [get_required_attr(config, "target_cflags")] + extra_link_flags
target_cxxflags = config.cxx_mode_flags + target_cflags
clang_asan_static_cflags = (["-fsanitize=address",
                            "-mno-omit-leaf-frame-pointer",
                            "-fno-omit-frame-pointer",
                            "-fno-optimize-sibling-calls"] +
                            config.debug_info_flags + target_cflags)
if config.target_arch == 's390x':
  clang_asan_static_cflags.append("-mbackchain")
clang_asan_static_cxxflags = config.cxx_mode_flags + clang_asan_static_cflags

asan_dynamic_flags = []
if config.asan_dynamic:
  asan_dynamic_flags = ["-shared-libasan"]
  if platform.system() == 'Windows':
    # On Windows, we need to simulate "clang-cl /MD" on the clang driver side.
    asan_dynamic_flags += ["-D_MT", "-D_DLL", "-Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames"]
  elif platform.system() == 'FreeBSD':
    # On FreeBSD, we need to add -pthread to ensure pthread functions are available.
    asan_dynamic_flags += ['-pthread']
  config.available_features.add("asan-dynamic-runtime")
else:
  config.available_features.add("asan-static-runtime")
clang_asan_cflags = clang_asan_static_cflags + asan_dynamic_flags
clang_asan_cxxflags = clang_asan_static_cxxflags + asan_dynamic_flags

# Add win32-(static|dynamic)-asan features to mark tests as passing or failing
# in those modes. lit doesn't support logical feature test combinations.
if platform.system() == 'Windows':
  if config.asan_dynamic:
    win_runtime_feature = "win32-dynamic-asan"
  else:
    win_runtime_feature = "win32-static-asan"
  config.available_features.add(win_runtime_feature)

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

config.substitutions.append( ("%clang ", build_invocation(target_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(target_cxxflags)) )
config.substitutions.append( ("%clang_asan ", build_invocation(clang_asan_cflags)) )
config.substitutions.append( ("%clangxx_asan ", build_invocation(clang_asan_cxxflags)) )
if config.asan_dynamic:
  if config.host_os in ['Linux', 'FreeBSD', 'NetBSD', 'SunOS']:
    shared_libasan_path = os.path.join(config.compiler_rt_libdir, "libclang_rt.asan{}.so".format(config.target_suffix))
  elif config.host_os == 'Darwin':
    shared_libasan_path = os.path.join(config.compiler_rt_libdir, 'libclang_rt.asan_{}_dynamic.dylib'.format(config.apple_platform))
  else:
    lit_config.warning('%shared_libasan substitution not set but dynamic ASan is available.')
    shared_libasan_path = None

  if shared_libasan_path is not None:
    config.substitutions.append( ("%shared_libasan", shared_libasan_path) )
  config.substitutions.append( ("%clang_asan_static ", build_invocation(clang_asan_static_cflags)) )
  config.substitutions.append( ("%clangxx_asan_static ", build_invocation(clang_asan_static_cxxflags)) )

# Windows-specific tests might also use the clang-cl.exe driver.
if platform.system() == 'Windows':
  clang_cl_cxxflags = ["-Wno-deprecated-declarations",
                       "-WX",
                       "-D_HAS_EXCEPTIONS=0",
                       "-Zi"] + target_cflags
  clang_cl_asan_cxxflags = ["-fsanitize=address"] + clang_cl_cxxflags
  if config.asan_dynamic:
    clang_cl_asan_cxxflags.append("-MD")

  clang_cl_invocation = build_invocation(clang_cl_cxxflags)
  clang_cl_invocation = clang_cl_invocation.replace("clang.exe","clang-cl.exe")
  config.substitutions.append( ("%clang_cl ", clang_cl_invocation) )

  clang_cl_asan_invocation = build_invocation(clang_cl_asan_cxxflags)
  clang_cl_asan_invocation = clang_cl_asan_invocation.replace("clang.exe","clang-cl.exe")
  config.substitutions.append( ("%clang_cl_asan ", clang_cl_asan_invocation) )

  base_lib = os.path.join(config.compiler_rt_libdir, "clang_rt.asan%%s%s.lib" % config.target_suffix)
  config.substitutions.append( ("%asan_lib", base_lib % "") )
  config.substitutions.append( ("%asan_cxx_lib", base_lib % "_cxx") )
  config.substitutions.append( ("%asan_dll_thunk", base_lib % "_dll_thunk") )

if platform.system() == 'Windows':
  # Don't use -std=c++11 on Windows, as the driver will detect the appropriate
  # default needed to use with the STL.
  config.substitutions.append(("%stdcxx11 ", ""))
else:
  # Some tests uses C++11 features such as lambdas and need to pass -std=c++11.
  config.substitutions.append(("%stdcxx11 ", "-std=c++11 "))

# FIXME: De-hardcode this path.
asan_source_dir = os.path.join(
  get_required_attr(config, "compiler_rt_src_root"), "lib", "asan")
python_exec = sh_quote(get_required_attr(config, "python_executable"))
# Setup path to asan_symbolize.py script.
asan_symbolize = os.path.join(asan_source_dir, "scripts", "asan_symbolize.py")
if not os.path.exists(asan_symbolize):
  lit_config.fatal("Can't find script on path %r" % asan_symbolize)
config.substitutions.append( ("%asan_symbolize", python_exec + " " + asan_symbolize + " ") )
# Setup path to sancov.py script.
sanitizer_common_source_dir = os.path.join(
  get_required_attr(config, "compiler_rt_src_root"), "lib", "sanitizer_common")
sancov = os.path.join(sanitizer_common_source_dir, "scripts", "sancov.py")
if not os.path.exists(sancov):
  lit_config.fatal("Can't find script on path %r" % sancov)
config.substitutions.append( ("%sancov ", python_exec + " " + sancov + " ") )

# Determine kernel bitness
if config.host_arch.find('64') != -1 and not config.android:
  kernel_bits = '64'
else:
  kernel_bits = '32'

config.substitutions.append( ('CHECK-%kernel_bits', ("CHECK-kernel-" + kernel_bits + "-bits")))

config.substitutions.append( ("%libdl", libdl_flag) )

config.available_features.add("asan-" + config.bits + "-bits")

# Fast unwinder doesn't work with Thumb
if not config.arm_thumb:
  config.available_features.add('fast-unwinder-works')

# Turn on leak detection on 64-bit Linux.
leak_detection_android = config.android and 'android-thread-properties-api' in config.available_features and (config.target_arch in ['x86_64', 'i386', 'i686', 'aarch64'])
leak_detection_linux = (config.host_os == 'Linux') and (not config.android) and (config.target_arch in ['x86_64', 'i386', 'riscv64'])
leak_detection_mac = (config.host_os == 'Darwin') and (config.target_arch == 'x86_64')
leak_detection_netbsd = (config.host_os == 'NetBSD') and (config.target_arch in ['x86_64', 'i386'])
if leak_detection_android or leak_detection_linux or leak_detection_mac or leak_detection_netbsd:
  config.available_features.add('leak-detection')

# Set LD_LIBRARY_PATH to pick dynamic runtime up properly.
push_dynamic_library_lookup_path(config, config.compiler_rt_libdir)

# GCC-ASan uses dynamic runtime by default.
if config.compiler_id == 'GNU':
  gcc_dir = os.path.dirname(config.clang)
  libasan_dir = os.path.join(gcc_dir, "..", "lib" + config.bits)
  push_dynamic_library_lookup_path(config, libasan_dir)

# Add the RT libdir to PATH directly so that we can successfully run the gtest
# binary to list its tests.
if config.host_os == 'Windows' and config.asan_dynamic:
  os.environ['PATH'] = os.path.pathsep.join([config.compiler_rt_libdir,
                                             os.environ.get('PATH', '')])

# Default test suffixes.
config.suffixes = ['.c', '.cpp']

if config.host_os == 'Darwin':
  config.suffixes.append('.mm')

if config.host_os == 'Windows':
  config.substitutions.append(('%fPIC', ''))
  config.substitutions.append(('%fPIE', ''))
  config.substitutions.append(('%pie', ''))
else:
  config.substitutions.append(('%fPIC', '-fPIC'))
  config.substitutions.append(('%fPIE', '-fPIE'))
  config.substitutions.append(('%pie', '-pie'))

# Only run the tests on supported OSs.
if config.host_os not in ['Linux', 'Darwin', 'FreeBSD', 'SunOS', 'Windows', 'NetBSD']:
  config.unsupported = True

if not config.parallelism_group:
  config.parallelism_group = 'shadow-memory'

if config.host_os == 'NetBSD':
  config.substitutions.insert(0, ('%run', config.netbsd_noaslr_prefix))
