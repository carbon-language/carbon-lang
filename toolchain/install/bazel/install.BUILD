# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel:carbon_cc_toolchain_config.bzl", "carbon_cc_toolchain_suite")

package(default_visibility = ["//visibility:public"])

carbon_cc_toolchain_suite(
    name = "carbon_cc_toolchain",
    configs = [
        ("linux", "aarch64"),
        ("linux", "x86_64"),
        ("freebsd", "x86_64"),
        ("macos", "arm64"),
        ("macos", "x86_64"),
    ],
)
