# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_setting", "int_setting")

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

int_setting(
    name = "bootstrap_stage",
    build_setting_default = 0,
    visibility = ["//visibility:public"],
)
