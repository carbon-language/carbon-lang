# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "builtin_kind",
    srcs = ["builtin_kind.cpp"],
    hdrs = ["builtin_kind.h"],
    textual_hdrs = ["builtin_kind.def"],
    deps = ["//common:enum_base"],
)

cc_library(
    name = "ids",
    hdrs = ["ids.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/base:index_base",
        "//toolchain/sem_ir:builtin_kind",
    ],
)

cc_library(
    name = "node_kind",
    srcs = ["node_kind.cpp"],
    hdrs = [
        "node_kind.h",
        "typed_nodes.h",
    ],
    textual_hdrs = ["node_kind.def"],
    deps = [
        "//common:enum_base",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:builtin_kind",
        "//toolchain/sem_ir:ids",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "node",
    srcs = ["node.cpp"],
    hdrs = ["node.h"],
    deps = [
        ":builtin_kind",
        ":node_kind",
        "//common:check",
        "//common:ostream",
        "//common:struct_reflection",
        "//toolchain/base:index_base",
        "//toolchain/parse:tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "file",
    srcs = ["file.cpp"],
    hdrs = ["file.h"],
    deps = [
        ":builtin_kind",
        ":node",
        ":node_kind",
        ":value_stores",
        "//common:check",
        "//toolchain/base:value_store",
        "//toolchain/base:yaml",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "formatter",
    srcs = ["formatter.cpp"],
    hdrs = ["formatter.h"],
    deps = [
        ":file",
        ":node_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "entry_point",
    srcs = ["entry_point.cpp"],
    hdrs = ["entry_point.h"],
    deps = [
        ":file",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_stores",
    srcs = ["value_stores.cpp"],
    hdrs = ["value_stores.h"],
    deps = [
        ":node",
        "//toolchain/base:value_store",
        "//toolchain/base:yaml",
        "//toolchain/lex:token_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "typed_nodes_test",
    size = "small",
    srcs = ["typed_nodes_test.cpp"],
    deps = [
        ":node",
        ":node_kind",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "yaml_test",
    size = "small",
    srcs = ["yaml_test.cpp"],
    deps = [
        "//common:ostream",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/driver",
        "//toolchain/testing:yaml_test_helpers",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
