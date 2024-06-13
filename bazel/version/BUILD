# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "int_flag", "string_flag")

package(default_visibility = ["//toolchain/install:__pkg__"])

exports_files(["gen_tmpl.py"])

bool_flag(
    name = "release",
    build_setting_default = False,
)

string_flag(
    name = "pre_release",
    build_setting_default = "dev",
    values = [
        "rc",
        "beta",
        "alpha",
        "nightly",
        "dev",
    ],
)

int_flag(
    name = "rc_number",
    build_setting_default = -1,
)

int_flag(
    name = "beta_number",
    build_setting_default = -1,
)

int_flag(
    name = "alpha_number",
    build_setting_default = -1,
)

string_flag(
    name = "nightly_date",
    build_setting_default = "",
)

# A config setting to observe the value of the `--stamp` command line flag
# within starlark with a macro and `select`. This is a workaround suggested for
# a Bazel issue: https://github.com/bazelbuild/bazel/issues/11164
config_setting(
    name = "internal_stamp_flag_detect",
    values = {"stamp": "1"},
    visibility = ["//visibility:public"],
)
