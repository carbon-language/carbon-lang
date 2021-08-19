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
        ":class_definition",
        ":function_definition",
        ":member",
        ":pattern",
        "//common:ostream",
        "//executable_semantics/common:ptr",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "expression",
    srcs = ["expression.cpp"],
    hdrs = ["expression.h"],
    deps = [
        ":paren_contents",
        "//common:indirect_value",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "expression_test",
    srcs = ["expression_test.cpp"],
    deps = [
        ":expression",
        ":paren_contents",
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
    name = "paren_contents",
    hdrs = ["paren_contents.h"],
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
        ":paren_contents",
        ":pattern",
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
    name = "class_definition",
    hdrs = ["class_definition.h"],
    deps = [
        ":member",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)
