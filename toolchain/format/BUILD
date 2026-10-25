# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "format",
    srcs = [
        "format.cpp",
        "formatter.cpp",
        "formatter.h",
        "token_info.cpp",
        "token_info.h",
        "whitespace_manager.cpp",
        "whitespace_manager.h",
    ],
    hdrs = [
        "format.h",
        "style.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/base:fixed_size_value_store",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "formatter_test",
    size = "small",
    srcs = ["formatter_test.cpp"],
    deps = [
        ":format",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:compile_helper",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
