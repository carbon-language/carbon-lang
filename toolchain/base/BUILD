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
    name = "value_store",
    hdrs = ["value_store.h"],
    deps = [
        ":index_base",
        ":yaml",
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "value_store_test",
    size = "small",
    srcs = ["value_store_test.cpp"],
    deps = [
        ":value_store",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/testing:yaml_test_helpers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "yaml",
    hdrs = ["yaml.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)
