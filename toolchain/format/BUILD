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
    name = "style",
    hdrs = ["style.h"],
)

cc_library(
    name = "token_info",
    srcs = ["token_info.cpp"],
    hdrs = ["token_info.h"],
    deps = [
        ":style",
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
        ":style",
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
    name = "comment",
    srcs = ["comment.cpp"],
    hdrs = ["comment.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "comment_test",
    size = "small",
    srcs = ["comment_test.cpp"],
    deps = [
        ":comment",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "cpp_snippet",
    srcs = ["cpp_snippet.cpp"],
    hdrs = ["cpp_snippet.h"],
    deps = [
        ":style",
        "@llvm-project//clang:format",
        "@llvm-project//clang:tooling_core",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "cpp_snippet_test",
    size = "small",
    srcs = ["cpp_snippet_test.cpp"],
    deps = [
        ":cpp_snippet",
        ":style",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "line_wrapper",
    srcs = ["line_wrapper.cpp"],
    hdrs = ["line_wrapper.h"],
    deps = [
        ":style",
        ":token_info",
        "//common:hashing",
        "//common:map",
        "//common:set",
        "//toolchain/lex:token_index",
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
        ":line_wrapper",
        ":style",
        ":token_info",
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
    name = "whitespace_manager",
    srcs = ["whitespace_manager.cpp"],
    hdrs = ["whitespace_manager.h"],
    deps = [
        ":style",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
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
        ":comment",
        ":cpp_snippet",
        ":line_wrapper",
        ":style",
        ":token_info",
        ":whitespace_manager",
        "//common:check",
        "//common:map",
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
        ":style",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:compile_helper",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
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
