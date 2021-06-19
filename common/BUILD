# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "indirect_value",
    srcs = ["indirect_value.h"],
)

cc_test(
    name = "indirect_value_test",
    srcs = ["indirect_value_test.cpp"],
    deps = [
        ":indirect_value",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)
