# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(default_visibility = ["//visibility:public"])

# Flag controlling whether the target config is used for the `carbon_*` Bazel
# rules. The default is to use the exec config as that is more correct in cases
# where the target config is not compatible with the exec (cross compiling), and
# for library users of Carbon likely the most efficient as it will provide an
# optimized toolchain.
#
# However, for building the Carbon project itself, this will roughly double the
# build cost by forcing a build in both target and exec config. As a consequence
# we disable the flag in the `.bazelrc` of the project for its builds.
bool_flag(
    name = "use_target_config_carbon_rules",
    build_setting_default = False,
)

config_setting(
    name = "use_target_config_carbon_rules_config",
    flag_values = {":use_target_config_carbon_rules": "True"},
)
