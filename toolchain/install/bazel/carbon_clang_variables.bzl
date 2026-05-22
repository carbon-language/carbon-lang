# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting detected Carbon toolchain configuration variables."""

load(
    "@carbon_toolchain_config//:carbon_detected_variables.bzl",
    _clang_include_dirs = "clang_include_dirs",
    _clang_resource_dir = "clang_resource_dir",
    _clang_sysroot = "clang_sysroot",
)

clang_include_dirs = _clang_include_dirs
clang_resource_dir = _clang_resource_dir
clang_sysroot = _clang_sysroot
