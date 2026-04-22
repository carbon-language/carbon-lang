# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load(":carbon_cc_toolchain_config.bzl", "gen_cc_toolchain_paths_with_stage")

package(default_visibility = ["//visibility:public"])

exports_files(["carbon_cc_toolchain_config.bzl"])

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

gen_cc_toolchain_paths_with_stage(
    name = "gen_cc_tools_paths",
    stage = 0,
)

# Test that the default toolchain's Make variables expand correctly.
py_test(
    name = "cc_tools_test",
    srcs = ["cc_tools_test.py"],
    args = ["$(location :gen_cc_tools_paths)"],
    data = [":gen_cc_tools_paths"],
    deps = [":cc_tools_test_lib"],
)

# Library containing the test logic, used by tests in other packages.
py_library(
    name = "cc_tools_test_lib",
    srcs = ["cc_tools_test.py"],
    visibility = ["//visibility:public"],
    deps = ["@bazel_tools//tools/python/runfiles"],
)
