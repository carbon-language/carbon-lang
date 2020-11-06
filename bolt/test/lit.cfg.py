# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'BOLT'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.cppm', '.m', '.mm', '.cu',
                   '.ll', '.cl', '.s', '.S', '.modulemap', '.test', '.rs']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.bolt_obj_root, 'test')

llvm_config.use_default_substitutions()

config.substitutions.append(('%host_cc', config.host_cc))
config.substitutions.append(('%host_cxx', config.host_cxx))

tool_dirs = [config.llvm_tools_dir,
             config.test_source_root]

linker_tool = llvm_config.use_llvm_tool('ld', required=True)
tools = [
    ToolSubst('llvm-bolt', unresolved='fatal'),
    ToolSubst('perf2bolt', unresolved='fatal'),
    ToolSubst('yaml2obj', unresolved='fatal'),
    ToolSubst('llvm-mc', unresolved='fatal'),
    ToolSubst('linker', command=linker_tool, unresolved='fatal'),
    ToolSubst('link_fdata', command=FindTool('link_fdata.sh'), unresolved='fatal'),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# Propagate path to symbolizer for ASan/MSan.
llvm_config.with_system_environment(
    ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])

config.substitutions.append(('%PATH%', config.environment['PATH']))

# Plugins (loadable modules)
# TODO: This should be supplied by Makefile or autoconf.
if sys.platform in ['win32', 'cygwin']:
    has_plugins = config.enable_shared
else:
    has_plugins = True

if has_plugins and config.llvm_plugin_ext:
    config.available_features.add('plugins')

def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + '-registered-target')
    return features


llvm_config.feature_config(
    [('--assertion-mode', {'ON': 'asserts'}),
     ('--cxxflags', {r'-D_GLIBCXX_DEBUG\b': 'libstdcxx-safe-mode'}),
        ('--targets-built', calculate_arch_features)
     ])

if config.enable_backtrace:
    config.available_features.add('backtrace')

# Check if we should allow outputs to console.
run_console_tests = int(lit_config.params.get('enable_console', '0'))
if run_console_tests != 0:
    config.available_features.add('console')
