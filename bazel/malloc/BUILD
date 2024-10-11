# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
    name = "is_linux",
    constraint_values = ["@platforms//os:linux"],
)

config_setting(
    name = "opt",
    values = {"compilation_mode": "opt"},
)

selects.config_setting_group(
    name = "linux_opt",
    match_all = [
        ":is_linux",
        ":opt",
    ],
)

# A library that enables TCMalloc for optimized builds on Linux. On other
# platforms and configurations, this falls back on the system malloc.
cc_library(
    name = "tcmalloc_if_linux_opt",
    deps = select({
        ":linux_opt": ["@tcmalloc//tcmalloc"],
        "//conditions:default": ["@bazel_tools//tools/cpp:malloc"],
    }),
)
