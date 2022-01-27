# -*- Python -*-

import os
import platform

import lit.formats

# Choose between lit's internal shell pipeline runner and a real shell.  If
# LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
use_lit_shell = os.environ.get("LIT_USE_INTERNAL_SHELL")
if use_lit_shell:
    # 0 is external, "" is default, and everything else is internal.
    execute_external = (use_lit_shell == "0")
else:
    # Otherwise we default to internal on Windows and external elsewhere, as
    # bash on Windows is usually very slow.
    execute_external = (not sys.platform in ['win32'])

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

# Setup config name.
config.name = 'Builtins' + config.name_suffix

# Platform-specific default Builtins_OPTIONS for lit tests.
default_builtins_opts = ''

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Path to the static library
is_msvc = get_required_attr(config, "is_msvc")
if is_msvc:
  base_lib = os.path.join(config.compiler_rt_libdir, "clang_rt.builtins%s.lib "
                          % config.target_suffix)
  config.substitutions.append( ("%librt ", base_lib) )
elif config.host_os  == 'Darwin':
  base_lib = os.path.join(config.compiler_rt_libdir, "libclang_rt.osx.a ")
  config.substitutions.append( ("%librt ", base_lib + ' -lSystem ') )
else:
  base_lib = os.path.join(config.compiler_rt_libdir, "libclang_rt.builtins%s.a"
                          % config.target_suffix)
  if sys.platform in ['win32'] and execute_external:
    # Don't pass dosish path separator to msys bash.exe.
    base_lib = base_lib.replace('\\', '/')
  config.substitutions.append( ("%librt ", base_lib + ' -lc -lm ') )

builtins_source_dir = os.path.join(
  get_required_attr(config, "compiler_rt_src_root"), "lib", "builtins")
if sys.platform in ['win32'] and execute_external:
  # Don't pass dosish path separator to msys bash.exe.
  builtins_source_dir = builtins_source_dir.replace('\\', '/')
builtins_lit_source_dir = get_required_attr(config, "builtins_lit_source_dir")

extra_link_flags = ["-nodefaultlibs"]

target_cflags = [get_required_attr(config, "target_cflags")]
target_cflags += ['-fno-builtin', '-I', builtins_source_dir]
target_cflags += extra_link_flags
target_cxxflags = config.cxx_mode_flags + target_cflags
clang_builtins_static_cflags = ([""] +
                            config.debug_info_flags + target_cflags)
clang_builtins_static_cxxflags = config.cxx_mode_flags + \
                                 clang_builtins_static_cflags

clang_builtins_cflags = clang_builtins_static_cflags
clang_builtins_cxxflags = clang_builtins_static_cxxflags

# FIXME: Right now we don't compile the C99 complex builtins when using
# clang-cl. Fix that.
if not is_msvc:
  config.available_features.add('c99-complex')

builtins_is_msvc = get_required_attr(config, "builtins_is_msvc")
if not builtins_is_msvc:
  config.available_features.add('int128')

clang_wrapper = ""

def build_invocation(compile_flags):
  return " " + " ".join([clang_wrapper, config.clang] + compile_flags) + " "


config.substitutions.append( ("%clang ", build_invocation(target_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(target_cxxflags)) )
config.substitutions.append( ("%clang_builtins ", \
                              build_invocation(clang_builtins_cflags)))
config.substitutions.append( ("%clangxx_builtins ", \
                              build_invocation(clang_builtins_cxxflags)))

# Default test suffixes.
config.suffixes = ['.c', '.cpp']

if not config.emulator:
  config.available_features.add('native-run')

# Add features for available sources
builtins_source_features = config.builtins_lit_source_features.split(';')
# Sanity checks
if not builtins_source_features:
  lit_config.fatal('builtins_source_features cannot be empty')
builtins_source_features_set = set()
builtins_source_feature_duplicates = []
for builtin_source_feature in builtins_source_features:
  if len(builtin_source_feature) == 0:
    lit_config.fatal('builtins_source_feature cannot contain empty features')
  if builtin_source_feature not in builtins_source_features_set:
    builtins_source_features_set.add(builtin_source_feature)
  else:
    builtins_source_feature_duplicates.append(builtin_source_feature)

if len(builtins_source_feature_duplicates) > 0:
  lit_config.fatal(
    'builtins_source_features contains duplicates: {}'.format(
      builtins_source_feature_duplicates)
  )
config.available_features.update(builtins_source_features)
