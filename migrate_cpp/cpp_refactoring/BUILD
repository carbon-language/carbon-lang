# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "cpp_refactoring",
    srcs = ["main.cpp"],
    deps = [
        ":fn_inserter",
        ":for_range",
        ":matcher",
        ":var_decl",
        "@llvm-project//clang:tooling",
    ],
)

cc_library(
    name = "matcher",
    srcs = ["matcher.cpp"],
    hdrs = [
        "matcher.h",
        "matcher_manager.h",
    ],
    deps = [
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:tooling_core",
    ],
)

cc_library(
    name = "matcher_test_base",
    testonly = 1,
    hdrs = ["matcher_test_base.h"],
    deps = [
        ":matcher",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:tooling",
        "@llvm-project//clang:tooling_core",
    ],
)

# Individual matchers

cc_library(
    name = "fn_inserter",
    srcs = ["fn_inserter.cpp"],
    hdrs = ["fn_inserter.h"],
    deps = [
        ":matcher",
        "@llvm-project//clang:ast_matchers",
    ],
)

cc_test(
    name = "fn_inserter_test",
    srcs = ["fn_inserter_test.cpp"],
    deps = [
        ":fn_inserter",
        ":matcher_test_base",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:tooling",
    ],
)

cc_library(
    name = "for_range",
    srcs = ["for_range.cpp"],
    hdrs = ["for_range.h"],
    deps = [
        ":matcher",
        "@llvm-project//clang:ast_matchers",
    ],
)

cc_test(
    name = "for_range_test",
    srcs = ["for_range_test.cpp"],
    deps = [
        ":for_range",
        ":matcher_test_base",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:tooling",
    ],
)

cc_library(
    name = "var_decl",
    srcs = ["var_decl.cpp"],
    hdrs = ["var_decl.h"],
    deps = [
        ":matcher",
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:type_nodes_gen",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "var_decl_test",
    srcs = ["var_decl_test.cpp"],
    deps = [
        ":matcher_test_base",
        ":var_decl",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:tooling",
    ],
)
