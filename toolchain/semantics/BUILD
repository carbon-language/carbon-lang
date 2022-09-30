# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nodes",
    hdrs = [
        "node_kind.h",
        "node_ref.h",
        "nodes/binary_operator.h",
        "nodes/function.h",
        "nodes/integer_literal.h",
        "nodes/return.h",
        "nodes/set_name.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "parse_subtree_consumer",
    srcs = ["parse_subtree_consumer.cpp"],
    hdrs = ["parse_subtree_consumer.h"],
    deps = [
        "//common:check",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir",
    srcs = ["semantics_ir.cpp"],
    hdrs = [
        "node_store.h",
        "semantics_ir.h",
    ],
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
        ":parse_subtree_consumer",
        ":semantics_ir",
        "//common:check",
        "//toolchain/lexer:token_kind",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir_test_helpers",
    testonly = 1,
    srcs = ["semantics_ir_for_test.cpp"],
    hdrs = [
        "nodes/binary_operator_test_matchers.h",
        "nodes/function_test_matchers.h",
        "nodes/integer_literal_test_matchers.h",
        "nodes/return_test_matchers.h",
        "nodes/set_name_test_matchers.h",
        "semantics_ir_for_test.h",
        "semantics_ir_test_helpers.h",
    ],
    deps = [
        ":nodes",
        ":semantics_ir",
        "//common:check",
        "//common:ostream",
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
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
