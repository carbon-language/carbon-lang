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

tools = [
    ToolSubst('llvm-dwarfdump', unresolved='fatal'),
    ToolSubst('llvm-bolt', unresolved='fatal'),
    ToolSubst('perf2bolt', unresolved='fatal'),
    ToolSubst('yaml2obj', unresolved='fatal'),
    ToolSubst('llvm-mc', unresolved='fatal'),
    ToolSubst('link_fdata', command=FindTool('link_fdata.sh'), unresolved='fatal'),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

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
