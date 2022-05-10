# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//explorer:__subpackages__"])

cc_library(
    name = "ast",
    hdrs = ["ast.h"],
    deps = [
        ":declaration",
        ":library_name",
        "//explorer/common:nonnull",
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
        ":source_location",
        "@llvm-project//llvm:Support",
    ],
)

genrule(
    name = "ast_rtti",
    srcs = ["ast_rtti.txt"],
    outs = ["ast_rtti.h"],
    cmd = "./$(location //explorer:gen_rtti)" +
          " $(location ast_rtti.txt) > \"$@\"",
    tools = ["//explorer:gen_rtti"],
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
        "//explorer/common:arena",
        "@com_google_googletest//:gtest_main",
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
        ":source_location",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//explorer/common:nonnull",
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
        ":source_location",
        ":statement",
        ":static_scope",
        ":value_category",
        "//common:ostream",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "return_term",
    hdrs = ["return_term.h"],
    deps = [
        ":expression",
        ":source_location",
        "//common:check",
        "//common:ostream",
        "//explorer/common:nonnull",
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
        ":source_location",
        ":static_scope",
        ":value_category",
        "//common:indirect_value",
        "//common:ostream",
        "//explorer/common:arena",
        "//explorer/common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "expression_test",
    srcs = ["expression_test.cpp"],
    deps = [
        ":expression",
        ":paren_contents",
        "//explorer/common:arena",
        "@com_google_googletest//:gtest_main",
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
        ":source_location",
        "//explorer/common:error",
    ],
)

cc_library(
    name = "pattern",
    srcs = ["pattern.cpp"],
    hdrs = ["pattern.h"],
    deps = [
        ":ast_node",
        ":expression",
        ":source_location",
        ":static_scope",
        ":value_category",
        "//common:ostream",
        "//explorer/common:arena",
        "//explorer/common:error",
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
        "//explorer/common:arena",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "static_scope",
    srcs = ["static_scope.cpp"],
    hdrs = ["static_scope.h"],
    deps = [
        ":ast_node",
        ":source_location",
        ":value_category",
        "//common:check",
        "//common:error",
        "//explorer/common:error",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = [
        "//common:ostream",
        "//explorer/common:nonnull",
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
        ":source_location",
        ":static_scope",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//explorer/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_category",
    hdrs = ["value_category.h"],
)
