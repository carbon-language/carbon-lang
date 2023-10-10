# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index_base",
    hdrs = ["index_base.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "pretty_stack_trace_function",
    hdrs = ["pretty_stack_trace_function.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "struct_reflection",
    hdrs = ["struct_reflection.h"],
)

cc_test(
    name = "struct_reflection_test",
    size = "small",
    srcs = ["struct_reflection_test.cpp"],
    deps = [
        ":struct_reflection",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
