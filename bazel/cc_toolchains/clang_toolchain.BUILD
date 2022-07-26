# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is symlinked into a configured Clang toolchain repository as the
# root `BUILD` file for that repository.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_toolchain", "cc_toolchain_suite")
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain_suite(
    name = "bazel_cc_toolchain",
    toolchains = {
        "darwin": ":cc-compiler-darwin",
        "darwin_arm64": ":cc-compiler-darwin-arm64",
        "k8": ":cc-compiler-k8",
        "x64_windows": ":cc-compiler-x64-windows",
    },
)

cc_toolchain(
    name = "cc-compiler-k8",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local-k8",
    toolchain_identifier = "local-k8",
)

cc_toolchain_config(
    name = "local-k8",
    target_cpu = "k8",
)

cc_toolchain(
    name = "cc-compiler-darwin",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local-darwin",
    toolchain_identifier = "local-darwin",
)

cc_toolchain_config(
    name = "local-darwin",
    target_cpu = "darwin",
)

cc_toolchain(
    name = "cc-compiler-darwin-arm64",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local-darwin-arm64",
    toolchain_identifier = "local-darwin-arm64",
)

cc_toolchain_config(
    name = "local-darwin-arm64",
    target_cpu = "darwin_arm64",
)

cc_toolchain(
    name = "cc-compiler-x64-windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local-x64-windows",
    toolchain_identifier = "local-x64-windows",
)

cc_toolchain_config(
    name = "local-x64-windows",
    target_cpu = "x64_windows",
)
