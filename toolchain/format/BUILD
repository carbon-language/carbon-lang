# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "token_info",
    srcs = ["token_info.cpp"],
    hdrs = ["token_info.h"],
    deps = [
        "//toolchain/base:fixed_size_value_store",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "token_info_test",
    size = "small",
    srcs = ["token_info_test.cpp"],
    deps = [
        ":token_info",
        "//testing/base:gtest_main",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/testing:compile_helper",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "line_wrapper",
    srcs = ["line_wrapper.cpp"],
    hdrs = ["line_wrapper.h"],
    deps = [
        ":token_info",
        "//common:hashing",
        "//common:set",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "line_wrapper_test",
    size = "small",
    srcs = ["line_wrapper_test.cpp"],
    deps = [
        ":token_info",
        ":line_wrapper",
        "//testing/base:gtest_main",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:compile_helper",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "format",
    srcs = [
        "format.cpp",
        "formatter.cpp",
        "formatter.h",
    ],
    hdrs = ["format.h"],
    deps = [
        ":line_wrapper",
        ":token_info",
        "//common:ostream",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
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
    ],
)

cc_fuzz_test(
    name = "format_fuzzer",
    size = "small",
    srcs = ["format_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":format",
        "//common:check",
        "//common:raw_string_ostream",
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)
