# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of optimization `cc_toolchain_config` features."""

load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)
load(
    ":cc_toolchain_actions.bzl",
    "all_compile_actions",
    "codegen_compile_actions",
)

# Handle different levels of optimization with individual features so that
# they can be ordered and the defaults can override the minimal settings if
# both are enabled.
minimal_optimization_flags = feature(
    name = "minimal_optimization_flags",
    flag_sets = [flag_set(
        actions = codegen_compile_actions,
        flag_groups = [flag_group(flags = ["-O1"])],
    )],
)
default_optimization_flags = feature(
    name = "default_optimization_flags",
    enabled = True,
    requires = [feature_set(["opt"])],
    flag_sets = [
        flag_set(
            actions = all_compile_actions,
            flag_groups = [flag_group(flags = ["-DNDEBUG"])],
        ),
        flag_set(
            actions = codegen_compile_actions,
            flag_groups = [flag_group(flags = ["-O3"])],
        ),
    ],
)

aarch64_cpu_flags = feature(
    name = "aarch64_cpu_flags",
    enabled = True,
    flag_sets = [flag_set(
        actions = all_compile_actions,
        flag_groups = [flag_group(flags = ["-march=armv8.2-a"])],
    )],
)

x86_64_cpu_flags = feature(
    name = "x86_64_cpu_flags",
    enabled = True,
    flag_sets = [flag_set(
        actions = all_compile_actions,
        flag_groups = [flag_group(flags = ["-march=x86-64-v2"])],
    )],
)

# Note that the order of features is significant in this list and determines the
# relative order of flags from the features listed.
optimization_features = [
    minimal_optimization_flags,
    default_optimization_flags,
]
