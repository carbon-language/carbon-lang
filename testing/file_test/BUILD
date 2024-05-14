# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "autoupdate",
    testonly = 1,
    srcs = ["autoupdate.cpp"],
    hdrs = [
        "autoupdate.h",
        "line.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "@abseil-cpp//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@re2",
    ],
)

cc_library(
    name = "file_test_base",
    testonly = 1,
    srcs = ["file_test_base.cpp"],
    hdrs = ["file_test_base.h"],
    deps = [
        ":autoupdate",
        "//common:check",
        "//common:error",
        "//common:init_llvm",
        "//common:ostream",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "file_test_base_test",
    size = "small",
    srcs = ["file_test_base_test.cpp"],
    tests = glob(["testdata/**"]),
    deps = [
        ":file_test_base",
        "//common:ostream",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
