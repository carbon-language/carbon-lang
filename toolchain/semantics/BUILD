# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "semantics_builtin_kind",
    srcs = ["semantics_builtin_kind.cpp"],
    hdrs = ["semantics_builtin_kind.h"],
    textual_hdrs = ["semantics_builtin_kind.def"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_node_kind",
    srcs = ["semantics_node_kind.cpp"],
    hdrs = ["semantics_node_kind.h"],
    textual_hdrs = ["semantics_node_kind.def"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir",
    srcs = [
        "semantics_parse_tree_handler.cpp",
        "semantics_ir.cpp",
        "semantics_node.cpp",
    ],
    hdrs = [
        "semantics_parse_tree_handler.h",
        "semantics_ir.h",
        "semantics_node.h",
    ],
    deps = [
        ":semantics_builtin_kind",
        ":semantics_node_kind",
        "//common:check",
        "//common:ostream",
        "//toolchain/lexer:token_kind",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)
