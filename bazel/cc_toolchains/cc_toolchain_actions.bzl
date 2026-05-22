# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Useful sets of actions for defining `cc_toolchain_config` features."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES", "ACTION_NAME_GROUPS")

# all_c_compile_actions includes actions that compile C or assembly.
all_c_compile_actions = [
    x
    for x in ACTION_NAME_GROUPS.all_cc_compile_actions
    if x not in ACTION_NAME_GROUPS.all_cpp_compile_actions
]

# preprocessor_compile_actions includes actions that run the preprocessor.
preprocessor_compile_actions = [
    x
    for x in ACTION_NAME_GROUPS.all_cc_compile_actions
    if x not in [ACTION_NAMES.assemble, ACTION_NAMES.cpp_module_codegen]
]
