# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/testing:golden_test.bzl", "golden_test")

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "check",
    hdrs = ["check.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cpp"],
    deps = [
        ":check",
        "@llvm-project//llvm:gtest_main",
    ],
)
