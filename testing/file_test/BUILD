# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "file_test_base",
    testonly = 1,
    srcs = ["file_test_base.cpp"],
    hdrs = ["file_test_base.h"],
    deps = [
        "//common:check",
        "//testing/util:test_raw_ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "file_test_base_test",
    srcs = ["file_test_base_test.cpp"],
    tests = glob(["testdata/**"]),
    deps = [
        ":file_test_base",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
