# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "array_stack",
    hdrs = ["array_stack.h"],
    deps = [
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "array_stack_test",
    size = "small",
    srcs = ["array_stack_test.cpp"],
    deps = [
        ":array_stack",
        ":value_store",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "index_base",
    hdrs = ["index_base.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "kind_switch",
    hdrs = ["kind_switch.h"],
    deps = [
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
        "//common:hashing",
        "//common:ostream",
        "//common:set",
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
        "@googletest//:gtest",
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
