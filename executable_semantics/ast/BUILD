# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "declaration",
    srcs = ["declaration.cpp"],
    hdrs = [
        "abstract_syntax_tree.h",
        "declaration.h",
    ],
    deps = [
        ":function_definition",
        ":member",
        ":pattern",
        ":struct_definition",
        "//common:ostream",
        "//executable_semantics/interpreter:address",
        "//executable_semantics/interpreter:containers",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "expression",
    srcs = ["expression.cpp"],
    hdrs = ["expression.h"],
    deps = [
        "//common:indirect_value",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "//executable_semantics/syntax:paren_contents",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "expression_test",
    srcs = ["expression_test.cpp"],
    deps = [
        ":expression",
        "//executable_semantics/syntax:paren_contents",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

cc_library(
    name = "function_definition",
    srcs = ["function_definition.cpp"],
    hdrs = ["function_definition.h"],
    deps = [
        ":expression",
        ":statement",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "member",
    srcs = ["member.cpp"],
    hdrs = ["member.h"],
    deps = [
        ":pattern",
        "//common:ostream",
    ],
)

cc_library(
    name = "pattern",
    srcs = ["pattern.cpp"],
    hdrs = ["pattern.h"],
    deps = [
        ":expression",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "pattern_test",
    srcs = ["pattern_test.cpp"],
    deps = [
        ":pattern",
        "//executable_semantics/syntax:paren_contents",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

cc_library(
    name = "statement",
    srcs = ["statement.cpp"],
    hdrs = ["statement.h"],
    deps = [
        ":expression",
        ":pattern",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "struct_definition",
    hdrs = ["struct_definition.h"],
    deps = [
        ":member",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)
