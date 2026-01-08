# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("llvm_tools.bzl", "LLVM_MAIN_TOOLS", "generate_llvm_tools_def")
load("runtime_sources.bzl", "generate_runtime_sources_cc_library")

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
        ":install_paths",
        "//common:check",
        "//common:string_helpers",
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
        ":value_store",
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
    name = "id_tag",
    hdrs = ["id_tag.h"],
    deps = [
        "//common:check",
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

# A library for computing install paths for the toolchain. Note that this
# library does *not* include the data itself, as that would form a dependency
# cycle. Each part of the toolchain should add the narrow data file groups to
# their data dependencies, and then use this library to locate them.
cc_library(
    name = "install_paths",
    srcs = ["install_paths.cpp"],
    hdrs = ["install_paths.h"],
    deps = [
        ":llvm_tools",
        "//common:check",
        "//common:error",
        "//common:filesystem",
        "@bazel_tools//tools/cpp/runfiles",
        "@llvm-project//clang:basic",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "test_binary",
    testonly = 1,
    srcs = ["test_binary.cpp"],
    data = ["//toolchain/install:install_data"],
)

cc_test(
    name = "install_paths_test",
    size = "small",
    srcs = ["install_paths_test.cpp"],
    data = [
        ":test_binary",
        "//toolchain/install:install_data",
    ],
    deps = [
        ":install_paths",
        "//common:check",
        "//common:error_test_helpers",
        "//common:filesystem",
        "//common:ostream",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "install_paths_test_helpers",
    testonly = 1,
    srcs = ["install_paths_test_helpers.cpp"],
    hdrs = ["install_paths_test_helpers.h"],
    deps = [
        ":install_paths",
        "//testing/base:global_exe_path",
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
        ":value_store",
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
        ":id_tag",
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

generate_runtime_sources_cc_library(name = "runtime_sources")

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
