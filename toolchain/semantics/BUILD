# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "semantics",
    srcs = ["semantics.cpp"],
    hdrs = ["semantics.h"],
    deps = [
        "//common:check",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/parser:parse_tree_node_location_translator",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantic_analyzer",
    srcs = ["semantic_analyzer.cpp"],
    hdrs = ["semantic_analyzer.h"],
    deps = [
        ":semantics",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "//toolchain/parser:parse_tree_node_location_translator",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "semantic_analyzer_test",
    size = "small",
    srcs = ["semantic_analyzer_test.cpp"],
    deps = [
        ":semantic_analyzer",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/source:source_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)
