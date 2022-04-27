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
        "//executable_semantics/common:nonnull",
    ],
)

cc_library(
    name = "ast_node",
    srcs = ["ast_node.cpp"],
    hdrs = [
        "ast_node.h",
        "ast_rtti.h",
    ],
    deps = [
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

genrule(
    name = "ast_rtti",
    srcs = ["ast_rtti.txt"],
    outs = ["ast_rtti.h"],
    cmd = "./$(location //executable_semantics:gen_rtti)" +
          " $(location ast_rtti.txt) > \"$@\"",
    tools = ["//executable_semantics:gen_rtti"],
)

cc_library(
    name = "ast_test_matchers",
    testonly = 1,
    srcs = [
        "ast_test_matchers_internal.cpp",
        "ast_test_matchers_internal.h",
    ],
    hdrs = ["ast_test_matchers.h"],
    deps = [
        ":ast",
        ":ast_node",
        ":declaration",
        ":expression",
        ":statement",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_test_matchers_test",
    srcs = ["ast_test_matchers_test.cpp"],
    deps = [
        ":ast_test_matchers",
        ":declaration",
        ":expression",
        ":pattern",
        ":statement",
        "//common:gtest_main",
        "//executable_semantics/common:arena",
    ],
)

cc_library(
    name = "impl_binding",
    hdrs = [
        "impl_binding.h",
    ],
    deps = [
        ":ast_node",
        ":pattern",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "declaration",
    srcs = ["declaration.cpp"],
    hdrs = [
        "declaration.h",
    ],
    deps = [
        ":ast_node",
        ":impl_binding",
        ":pattern",
        ":return_term",
        ":statement",
        ":static_scope",
        ":value_category",
        "//common:ostream",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "return_term",
    hdrs = ["return_term.h"],
    deps = [
        ":expression",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "expression",
    srcs = ["expression.cpp"],
    hdrs = ["expression.h"],
    deps = [
        ":ast_node",
        ":paren_contents",
        ":static_scope",
        ":value_category",
        "//common:indirect_value",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error_builders",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "expression_test",
    srcs = ["expression_test.cpp"],
    deps = [
        ":expression",
        ":paren_contents",
        "//common:gtest_main",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
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
        "//executable_semantics/common:error_builders",
        "//executable_semantics/common:source_location",
    ],
)

cc_library(
    name = "pattern",
    srcs = ["pattern.cpp"],
    hdrs = ["pattern.h"],
    deps = [
        ":ast_node",
        ":expression",
        ":static_scope",
        ":value_category",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error_builders",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "pattern_test",
    srcs = ["pattern_test.cpp"],
    deps = [
        ":expression",
        ":paren_contents",
        ":pattern",
        "//common:gtest_main",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "static_scope",
    srcs = ["static_scope.cpp"],
    hdrs = ["static_scope.h"],
    deps = [
        ":ast_node",
        ":value_category",
        "//common:check",
        "//common:error",
        "//executable_semantics/common:error_builders",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "statement",
    srcs = ["statement.cpp"],
    hdrs = ["statement.h"],
    deps = [
        ":ast_node",
        ":expression",
        ":pattern",
        ":return_term",
        ":static_scope",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_category",
    hdrs = ["value_category.h"],
)
