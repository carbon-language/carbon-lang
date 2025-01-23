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
    name = "mem_usage",
    hdrs = ["mem_usage.h"],
    deps = [
        ":yaml",
        "//common:map",
        "//common:set",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "timings",
    hdrs = ["timings.h"],
    deps = [
        ":yaml",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_ids",
    hdrs = ["value_ids.h"],
    deps = [
        ":index_base",
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_store",
    hdrs = ["value_store.h"],
    deps = [
        ":mem_usage",
        ":yaml",
        "//common:check",
        "//common:hashtable_key_context",
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
        ":value_ids",
        ":value_store",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "int",
    srcs = ["int.cpp"],
    hdrs = ["int.h"],
    deps = [
        ":index_base",
        ":mem_usage",
        ":value_store",
        ":yaml",
        "//common:check",
        "//common:hashtable_key_context",
        "//common:ostream",
        "//common:set",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "int_test",
    size = "small",
    srcs = ["int_test.cpp"],
    deps = [
        ":int",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "shared_value_stores",
    hdrs = ["shared_value_stores.h"],
    deps = [
        ":int",
        ":mem_usage",
        ":value_ids",
        ":value_store",
        ":yaml",
    ],
)

cc_test(
    name = "shared_value_stores_test",
    size = "small",
    srcs = ["shared_value_stores_test.cpp"],
    deps = [
        ":shared_value_stores",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
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
