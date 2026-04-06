# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:selects.bzl", "selects")

package(default_visibility = ["//visibility:public"])

constraint_setting(
    name = "bootstrap_stage",
    default_constraint_value = ":stage0",
)

constraint_value(
    name = "stage0",
    constraint_setting = ":bootstrap_stage",
)

constraint_value(
    name = "stage1",
    constraint_setting = ":bootstrap_stage",
)

constraint_value(
    name = "stage2",
    constraint_setting = ":bootstrap_stage",
)

# For use by defs.bzl.
# Matches when asan is enabled on a macOS platform.
selects.config_setting_group(
    name = "macos_asan",
    match_all = [
        ":is_macos",
        ":macos_asan_build_modes",
    ],
)

# For use by defs.bzl.
# Matches macOS platforms.
config_setting(
    name = "is_macos",
    constraint_values = ["@platforms//os:macos"],
)

# For use by defs.bzl.
# Matches build modes where asan is enabled.
selects.config_setting_group(
    name = "macos_asan_build_modes",
    match_any = [
        ":dbg",
        ":fastbuild",
    ],
)

# For use by defs.bzl.
# Matches dbg.
config_setting(
    name = "dbg",
    values = {"compilation_mode": "dbg"},
)

# For use by defs.bzl.
# Matches fastbuild.
config_setting(
    name = "fastbuild",
    values = {"compilation_mode": "fastbuild"},
)

filegroup(
    name = "installed_cc_toolchain_starlark",
    srcs = [
        "carbon_cc_toolchain_config.bzl",
        "cc_toolchain_actions.bzl",
        "cc_toolchain_base_features.bzl",
        "cc_toolchain_config_features.bzl",
        "cc_toolchain_cpp_features.bzl",
        "cc_toolchain_debugging.bzl",
        "cc_toolchain_features.bzl",
        "cc_toolchain_linking.bzl",
        "cc_toolchain_modules.bzl",
        "cc_toolchain_optimization.bzl",
        "cc_toolchain_sanitizer_features.bzl",
        "cc_toolchain_tools.bzl",

        # TODO: Remove this once we can remove the use of it from Carbon
        # toolchain rules.
        "cc_toolchain_carbon_project_features.bzl",
    ],
)
