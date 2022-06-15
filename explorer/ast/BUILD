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
    srcs = ["ast_node.cpp", "ast_rtti.cpp",],
    hdrs = [
        "ast_node.h",
        "ast_rtti.h",
    ],
    deps = [
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

genrule(
    name = "ast_rtti",
    srcs = ["ast_rtti.txt"],
    outs = ["ast_rtti.h", "ast_rtti.cpp"],
    cmd = "./$(location //explorer:gen_rtti)" +
          " $(location ast_rtti.txt)" +
          " $(location ast_rtti.h) $(location ast_rtti.cpp)" +
          " $(rootpath ast_rtti.h)",
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
        ":statement",
        "//common:gtest_main",
        "//explorer/common:arena",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "impl_binding",
    hdrs = [
        "impl_binding.h",
    ],
    deps = [
        ":ast_node",
        ":expression",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
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
        ":expression",
        ":return_term",
        ":statement",
        ":static_scope",
        ":value_category",
        "//common:ostream",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
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
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "expression",
    srcs = ["expression.cpp", "pattern.cpp"],
    hdrs = ["expression.h", "pattern.h"],
    deps = [
        ":ast_node",
        ":paren_contents",
        ":static_scope",
        ":value_category",
        "//common:indirect_value",
        "//common:ostream",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "//explorer/common:source_location",
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
        "//explorer/common:arena",
        "@com_google_googletest//:gtest",
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
        "//explorer/common:error_builders",
        "//explorer/common:source_location",
    ],
)

cc_test(
    name = "pattern_test",
    srcs = ["pattern_test.cpp"],
    deps = [
        ":expression",
        ":paren_contents",
        "//common:gtest_main",
        "//explorer/common:arena",
        "@com_google_googletest//:gtest",
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
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
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
        ":return_term",
        ":static_scope",
        ":value_category",
        "//common:check",
        "//common:ostream",
        "//explorer/common:arena",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_category",
    hdrs = ["value_category.h"],
)
