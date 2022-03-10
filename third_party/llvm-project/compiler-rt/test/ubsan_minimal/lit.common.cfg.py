# -*- Python -*-

import os

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

# Setup source root.
config.test_source_root = os.path.dirname(__file__)
config.name = 'UBSan-Minimal-' + config.target_arch

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

target_cflags = [get_required_attr(config, "target_cflags")]
clang_ubsan_cflags = ["-fsanitize-minimal-runtime"] + target_cflags
clang_ubsan_cxxflags = config.cxx_mode_flags + clang_ubsan_cflags

# Define %clang and %clangxx substitutions to use in test RUN lines.
config.substitutions.append( ("%clang ", build_invocation(clang_ubsan_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(clang_ubsan_cxxflags)) )

# Default test suffixes.
config.suffixes = ['.c', '.cpp']

# Check that the host supports UndefinedBehaviorSanitizerMinimal tests
if config.host_os not in ['Linux', 'FreeBSD', 'NetBSD', 'Darwin', 'OpenBSD', 'SunOS']: # TODO: Windows
  config.unsupported = True

# Don't target x86_64h if the test machine can't execute x86_64h binaries.
if '-arch x86_64h' in target_cflags and 'x86_64h' not in config.available_features:
  config.unsupported = True

config.available_features.add('arch=' + config.target_arch)
