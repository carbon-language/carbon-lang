# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

file_test(
    name = "file_test",
    size = "small",
    srcs = ["file_test.cpp"],
    tests = [
        "//toolchain/check:testdata",
        "//toolchain/codegen:testdata",
        "//toolchain/diagnostics:testdata",
        "//toolchain/driver:testdata",
        "//toolchain/lex:testdata",
        "//toolchain/lower:testdata",
        "//toolchain/parse:testdata",
    ],
    deps = [
        "//common:all_llvm_targets",
        "//testing/file_test:file_test_base",
        "//toolchain/driver",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "yaml_test_helpers",
    testonly = 1,
    srcs = ["yaml_test_helpers.cpp"],
    hdrs = ["yaml_test_helpers.h"],
    deps = [
        "//common:error",
        "//common:ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "yaml_test_helpers_test",
    size = "small",
    srcs = ["yaml_test_helpers_test.cpp"],
    deps = [
        ":yaml_test_helpers",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
