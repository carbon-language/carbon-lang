# -*- Python -*-

import os

# Setup config name.
config.name = 'HWAddressSanitizer' + getattr(config, 'name_suffix', 'default')

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup default compiler flags used with -fsanitize=memory option.
clang_cflags = [config.target_cflags] + config.debug_info_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags
clang_hwasan_common_cflags = clang_cflags + ["-fsanitize=hwaddress", "-fuse-ld=lld"]

if config.target_arch == 'x86_64' and config.enable_aliases == '1':
  clang_hwasan_common_cflags += ["-fsanitize-hwaddress-experimental-aliasing"]
else:
  config.available_features.add('pointer-tagging')
if config.target_arch == 'x86_64':
  # The callback instrumentation used on x86_64 has a 1/64 chance of choosing a
  # stack tag of 0.  This causes stack tests to become flaky, so we force tags
  # to be generated via calls to __hwasan_generate_tag, which never returns 0.
  # TODO: See if we can remove this once we use the outlined instrumentation.
  clang_hwasan_common_cflags += ["-mllvm", "-hwasan-generate-tags-with-calls=1"]
clang_hwasan_cflags = clang_hwasan_common_cflags + ["-mllvm", "-hwasan-globals",
                                                   "-mllvm", "-hwasan-use-short-granules",
                                                   "-mllvm", "-hwasan-instrument-landing-pads=0",
                                                   "-mllvm", "-hwasan-instrument-personality-functions"]
clang_hwasan_oldrt_cflags = clang_hwasan_common_cflags + ["-mllvm", "-hwasan-use-short-granules=0",
                                                          "-mllvm", "-hwasan-instrument-landing-pads=1",
                                                          "-mllvm", "-hwasan-instrument-personality-functions=0"]

clang_hwasan_cxxflags = config.cxx_mode_flags + clang_hwasan_cflags
clang_hwasan_oldrt_cxxflags = config.cxx_mode_flags + clang_hwasan_oldrt_cflags

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

config.substitutions.append( ("%clangxx ", build_invocation(clang_cxxflags)) )
config.substitutions.append( ("%clang_hwasan ", build_invocation(clang_hwasan_cflags)) )
config.substitutions.append( ("%clang_hwasan_oldrt ", build_invocation(clang_hwasan_oldrt_cflags)) )
config.substitutions.append( ("%clangxx_hwasan ", build_invocation(clang_hwasan_cxxflags)) )
config.substitutions.append( ("%clangxx_hwasan_oldrt ", build_invocation(clang_hwasan_oldrt_cxxflags)) )
config.substitutions.append( ("%compiler_rt_libdir", config.compiler_rt_libdir) )

default_hwasan_opts_str = ':'.join(['disable_allocator_tagging=1', 'random_tags=0', 'fail_without_syscall_abi=0'] + config.default_sanitizer_opts)
if default_hwasan_opts_str:
  config.environment['HWASAN_OPTIONS'] = default_hwasan_opts_str
  default_hwasan_opts_str += ':'
config.substitutions.append(('%env_hwasan_opts=',
                             'env HWASAN_OPTIONS=' + default_hwasan_opts_str))

# Default test suffixes.
config.suffixes = ['.c', '.cpp']

if config.host_os not in ['Linux', 'Android'] or not config.has_lld:
  config.unsupported = True
