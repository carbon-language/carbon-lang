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
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_analyzer",
    srcs = ["semantics_analyzer.cpp"],
    hdrs = ["semantics_analyzer.h"],
    deps = [
        "//common:check",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "semantics_analyzer_test",
    size = "small",
    srcs = ["semantics_test.cpp"],
    deps = [
        ":semantics_analyzer",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/source:source_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)
