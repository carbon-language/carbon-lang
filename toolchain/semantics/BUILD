# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nodes",
    hdrs = [
        "declared_name.h",
        "function.h",
    ],
    deps = [
        "//common:ostream",
        "//toolchain/parser:parse_tree",
    ],
)

cc_library(
    name = "semantics_ir",
    srcs = ["semantics_ir.cpp"],
    hdrs = ["semantics_ir.h"],
    deps = [
        ":nodes",
        "//common:check",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir_factory",
    srcs = ["semantics_ir_factory.cpp"],
    hdrs = ["semantics_ir_factory.h"],
    deps = [
        ":nodes",
        ":semantics_ir",
        "//common:check",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir_test_helpers",
    testonly = 1,
    hdrs = ["semantics_ir_test_helpers.h"],
    deps = [
        ":semantics_ir",
        "//common:check",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "semantics_ir_factory_test",
    size = "small",
    srcs = ["semantics_ir_factory_test.cpp"],
    deps = [
        ":semantics_ir_factory",
        ":semantics_ir_test_helpers",
        "//common:gtest_main",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)
