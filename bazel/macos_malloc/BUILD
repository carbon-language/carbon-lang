# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:selects.bzl", "selects")

exports_files(["env.bzl"])

selects.config_setting_group(
    name = "macos_asan",
    match_all = [":is_macos", ":macos_asan_build_modes"],
)

config_setting(
    name = "is_macos",
    constraint_values = ["@platforms//os:osx"],
)

selects.config_setting_group(
    name = "macos_asan_build_modes",
    match_any = [":dbg", ":fastbuild"],
)

config_setting(
    name = "dbg",
    values = {"compilation_mode": "dbg"},
)

config_setting(
    name = "fastbuild",
    values = {"compilation_mode": "fastbuild"},
)
