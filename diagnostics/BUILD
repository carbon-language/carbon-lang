# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "diagnostic_emitter",
    srcs = ["diagnostic_emitter.cpp"],
    hdrs = ["diagnostic_emitter.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "null_diagnostics",
    hdrs = ["null_diagnostics.h"],
    deps = [
        ":diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mocks",
    testonly = 1,
    hdrs = ["mocks.h"],
    deps = [
        ":diagnostic_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gmock",
    ],
)

cc_test(
    name = "diagnostic_emitter_test",
    srcs = ["diagnostic_emitter_test.cpp"],
    deps = [
        ":diagnostic_emitter",
        ":mocks",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gmock",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)
