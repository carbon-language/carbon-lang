# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting", "int_flag")

filegroup(
    name = "clang_tidy_config",
    srcs = [".clang-tidy"],
    visibility = ["//visibility:public"],
)

# `bazel run //:generate_compile_commands` to produce `compile_commands.json`.
alias(
    name = "generate_compile_commands",
    actual = "@wolfd_bazel_compile_commands//:generate_compile_commands",
)

bool_setting(
    name = "runtimes_build",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

int_flag(
    name = "bootstrap_stage",
    build_setting_default = 0,
    visibility = ["//visibility:public"],
)

# A setting that causes bootstrapping to occur using the `exec` config rather
# than the target config.
#
# The exec config is the more technically correct way of doing bootstrapping
# than the target config. For example it allows bootstrapping with a target that
# isn't compatible with the current execution host. However, in development
# builds, it is likely to force building the entire toolchain twice -- once in
# the target config for running test, and a second time in the exec config for
# the bootstrap. As a consequence, this is disabled by default.
#
# TODO: Add documentation for using the bootstrap flags once stabilized.
bool_flag(
    name = "bootstrap_exec_config",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "bootstrap_with_exec_config",
    flag_values = {"//:bootstrap_exec_config": "True"},
    visibility = ["//visibility:public"],
)
