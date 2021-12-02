# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "parse_node_kind",
    srcs = ["parse_node_kind.cpp"],
    hdrs = ["parse_node_kind.h"],
    textual_hdrs = ["parse_node_kind.def"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_test(
    name = "parse_node_kind_test",
    srcs = ["parse_node_kind_test.cpp"],
    deps = [
        ":parse_node_kind",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "parse_tree",
    srcs = [
        "parse_tree.cpp",
        "parser_impl.cpp",
        "parser_impl.h",
    ],
    hdrs = ["parse_tree.h"],
    deps = [
        ":parse_node_kind",
        ":precedence",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:token_kind",
        "//toolchain/lexer:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "parse_test_helpers",
    testonly = 1,
    hdrs = ["parse_test_helpers.h"],
    deps = [
        ":parse_node_kind",
        ":parse_tree",
        "//toolchain/lexer:tokenized_buffer",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "parse_tree_test",
    srcs = ["parse_tree_test.cpp"],
    deps = [
        ":parse_node_kind",
        ":parse_test_helpers",
        ":parse_tree",
        "//common:ostream",
        "//toolchain/common:yaml_test_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lexer:tokenized_buffer",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "parse_tree_fuzzer",
    srcs = ["parse_tree_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":parse_tree",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lexer:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "precedence",
    srcs = ["precedence.cpp"],
    hdrs = ["precedence.h"],
    deps = [
        "//toolchain/lexer:token_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "precedence_test",
    srcs = ["precedence_test.cpp"],
    deps = [
        ":precedence",
        "//toolchain/lexer:token_kind",
        "@com_google_googletest//:gtest_main",
    ],
)
