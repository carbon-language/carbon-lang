# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//explorer:__subpackages__"])

AST_HDRS = [
    "address.h",
    "ast.h",
    "ast_node.h",
    "bindings.h",
    "clone_context.h",
    "declaration.h",
    "element.h",
    "element_path.h",
    "expression.h",
    "impl_binding.h",
    "pattern.h",
    "return_term.h",
    "statement.h",
    "value.h",
    "value_node.h",
    "value_transform.h",
]

genrule(
    name = "ast_rtti",
    srcs = ["ast_rtti.txt"] + AST_HDRS,
    outs = [
        "ast_rtti.h",
        "ast_rtti.cpp",
    ],
    cmd = "./$(location //explorer:gen_rtti)" +
          " $(location ast_rtti.txt)" +
          " $(location ast_rtti.h) $(location ast_rtti.cpp)" +
          " $(rootpath ast_rtti.h)" +
          "".join([" $(rootpath " + f + ")" for f in AST_HDRS]),
    tools = ["//explorer:gen_rtti"],
)

cc_library(
    name = "ast",
    srcs = [
        "ast_node.cpp",
        "ast_rtti.cpp",
        "bindings.cpp",
        "clone_context.cpp",
        "declaration.cpp",
        "element.cpp",
        "expression.cpp",
        "impl_binding.cpp",
        "pattern.cpp",
        "statement.cpp",
        "value.cpp",
    ],
    hdrs = AST_HDRS + ["ast_rtti.h"],
    textual_hdrs = [
        "value_kinds.def",
    ],
    deps = [
        ":library_name",
        ":paren_contents",
        ":value_category",
        "//common:check",
        "//common:error",
        "//common:indirect_value",
        "//common:ostream",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
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
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_test_matchers_test",
    srcs = ["ast_test_matchers_test.cpp"],
    deps = [
        ":ast",
        ":ast_test_matchers",
        "//common:gtest_main",
        "//explorer/common:arena",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "expression_test",
    srcs = ["expression_test.cpp"],
    deps = [
        ":ast",
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
    name = "element_test",
    srcs = ["element_test.cpp"],
    deps = [
        ":ast",
        ":paren_contents",
        "//common:gtest_main",
        "//explorer/common:arena",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "pattern_test",
    srcs = ["pattern_test.cpp"],
    deps = [
        ":ast",
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
        ":ast",
        "//common:check",
        "//common:error",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "value_category",
    hdrs = ["value_category.h"],
)
