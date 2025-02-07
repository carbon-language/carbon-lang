# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "int_flag", "string_flag")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//toolchain/install:__pkg__"])

py_binary(
    name = "gen_tmpl",
    srcs = ["gen_tmpl.py"],
)

# Several flags are provided for customizing the exact version used for the
# build of Carbon. Each of these is documented here, but rather than using the
# label-based names in Bazel invocations (`bazel build --//bazel/version:flag`)
# we suggest using the flag aliases provided in the project's `.bazelrc` and we
# document the flags using those aliases. The aliases match the local flag names
# here.
#
# For more details on the versioning scheme used by Carbon, see:
# - https://docs.google.com/document/d/11S5VAPe5Pm_BZPlajWrqDDVr9qc7-7tS2VshqO0wWkk/edit?resourcekey=0-2YFC9Uvl4puuDnWlr2MmYw
#   TODO: Replace with path to the markdown once this lands.
#
# First, we provide a flag to enable a release version: `--release`. It is
# disabled by default, and if enabled it must be the only version flag used.
bool_flag(
    name = "release",
    build_setting_default = False,
)

# A `--pre_release=KIND` flag where `KIND` must be one of:
# - `rc` -- a release candidate version.
#    Example: `--pre_release=rc --rc_number=2`
# - `nightly -- a nightly version.
#    Example: `--pre_release=nightly --nightly_date=2024.06.17`
# - `dev` -- the default, a development build.
#    Example: `--pre_release=dev`
#
# This flag cannot be used along with `--release`, and for all but the `dev`
# kind must be combined with one of the below flags to specify further details
# of the version.
string_flag(
    name = "pre_release",
    build_setting_default = "dev",
    values = [
        "rc",
        "nightly",
        "dev",
    ],
)

# `--rc_number=N` sets the release candidate number to `N`. Requires
# `--pre_release=rc`.
int_flag(
    name = "rc_number",
    build_setting_default = -1,
)

# `--nigthly_date=YYYY.MM.DD` sets the nightly pre-release date. Requires
# `--pre_release=nightly`. The value for this flag must be a string with the
# exact format of `YYYY.MM.DD`.
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
