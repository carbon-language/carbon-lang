# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "string_list_setting")

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

string_list_setting(
    name = "original_platforms",
    build_setting_default = [],
)
