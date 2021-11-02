# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "ast",
    hdrs = ["ast.h"],
    deps = [
        ":declaration",
        ":library_name",
    ],
)

cc_library(
    name = "declaration",
    srcs = ["declaration.cpp"],
    hdrs = [
        "declaration.h",
    ],
    deps = [
        ":member",
        ":pattern",
        ":source_location",
        ":statement",
        "//common:ostream",
        "//executable_semantics/common:nonnull",
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
    name = "member",
    srcs = ["member.cpp"],
    hdrs = ["member.h"],
    deps = [
        ":pattern",
        ":source_location",
        "//common:ostream",
    ],
)

cc_library(
    name = "library_name",
    hdrs = ["library_name.h"],
)

cc_library(
    name = "paren_contents",
    hdrs = ["paren_contents.h"],
    deps = [
        ":source_location",
        "//executable_semantics/common:error",
    ],
)

cc_library(
    name = "pattern",
    srcs = ["pattern.cpp"],
    hdrs = ["pattern.h"],
    deps = [
        ":expression",
        ":source_location",
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
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = ["//common:ostream"],
)

cc_library(
    name = "statement",
    srcs = ["statement.cpp"],
    hdrs = ["statement.h"],
    deps = [
        ":expression",
        ":pattern",
        ":source_location",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
    ],
)
