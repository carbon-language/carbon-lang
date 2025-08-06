# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")
load("llvm_tools.bzl", "LLVM_MAIN_TOOLS", "generate_llvm_tools_def")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "block_value_store",
    hdrs = ["block_value_store.h"],
    deps = [
        ":mem_usage",
        ":value_store",
        ":yaml",
        "//common:check",
        "//common:hashing",
        "//common:set",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "canonical_value_store",
    hdrs = ["canonical_value_store.h"],
    deps = [
        ":mem_usage",
        ":value_store",
        ":value_store_types",
        ":yaml",
        "//common:hashtable_key_context",
        "//common:set",
    ],
)

cc_test(
    name = "canonical_value_store_test",
    size = "small",
    srcs = ["canonical_value_store_test.cpp"],
    deps = [
        ":canonical_value_store",
        ":value_ids",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "clang_invocation",
    srcs = ["clang_invocation.cpp"],
    hdrs = ["clang_invocation.h"],
    deps = [
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "fixed_size_value_store",
    hdrs = ["fixed_size_value_store.h"],
    deps = [
        ":mem_usage",
        ":value_store_types",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "for_each_macro",
    hdrs = ["for_each_macro.h"],
    deps = [
        "@llvm-project//llvm:Support",
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
        ":for_each_macro",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "kind_switch_test",
    size = "small",
    srcs = ["kind_switch_test.cpp"],
    deps = [
        ":kind_switch",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
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
    name = "relational_value_store",
    hdrs = ["relational_value_store.h"],
    deps = [
        ":value_store_types",
        "//common:check",
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
        ":value_store_types",
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
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "value_store_types",
    hdrs = ["value_store_types.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "int",
    srcs = ["int.cpp"],
    hdrs = ["int.h"],
    deps = [
        ":canonical_value_store",
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

generate_llvm_tools_def(
    name = "llvm_tools_def",
    out = "llvm_tools.def",
)

config_setting(
    name = "is_macos",
    constraint_values = ["@platforms//os:macos"],
)

cc_library(
    name = "llvm_tools",
    srcs = ["llvm_tools.cpp"],
    hdrs = ["llvm_tools.h"],
    linkopts = select({
        # TODO: This should be moved upstream to LLVM's tool libraries that
        # require it.
        ":is_macos": [
            "-framework",
            "CoreFoundation",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":llvm_tools_def",
        "//common:command_line",
        "//common:enum_base",
        "@llvm-project//llvm:Support",
    ] + [info.lib for info in LLVM_MAIN_TOOLS.values()],
)

cc_library(
    name = "shared_value_stores",
    hdrs = ["shared_value_stores.h"],
    deps = [
        ":canonical_value_store",
        ":int",
        ":mem_usage",
        ":value_ids",
        ":value_store",
        ":yaml",
        "@llvm-project//llvm:Support",
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
