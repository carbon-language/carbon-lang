# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "int_flag", "string_flag")

package(default_visibility = ["//toolchain/install:__pkg__"])

exports_files(["gen_tmpl.py"])

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
# - `beta` -- a beta version.
# - `alpha` -- an alpha version.
# - `nightly -- a nightly version.
# - `dev` -- the default, a development build.
#
# This flag cannot be used along with `--release`, and for all but the `dev`
# kind must be combined with one of the below flags to specify further details
# of the version.
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

# When `--pre_release=rc` is used, `--rc_number=N` set's the release candidate
# number to `N`.
#
# An error if used without `--pre_release=rc`.
int_flag(
    name = "rc_number",
    build_setting_default = -1,
)

# When `--pre_release=beta` is used, `--beta_number=N` set's the beta
# pre-release number to `N`.
#
# An error if used without `--pre_release=beta`.
int_flag(
    name = "beta_number",
    build_setting_default = -1,
)

# When `--pre_release=alpha` is used, `--alpha_number=N` set's the alpha
# pre-release number to `N`.
#
# An error if used without `--pre_release=alpha`.
int_flag(
    name = "alpha_number",
    build_setting_default = -1,
)

# When `--pre_release=nightly` is used, `--nightly_date=YYYY.MM.DD` set's the
# nightly build date. The value for this flag must be a string with the exact
# format of `YYYY.MM.DD`.
#
# An error if used without `--pre_release=nightly`.
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
