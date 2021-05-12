# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "cpp_refactoring",
    srcs = ["main.cpp"],
    deps = [
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:tooling",
    ],
)
