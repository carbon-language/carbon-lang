# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_tree_sitter//tree_sitter:tree_sitter.bzl", "tree_sitter_cc_library")

tree_sitter_cc_library(
    name = "treesitter",
    srcs = ["src/scanner.c"],
    grammar = ["grammar.js"],
)

cc_binary(
    name = "test_runner",
    srcs = ["test_runner.cpp"],
    deps = [
        ":treesitter",
    ],
)

cc_test(
    name = "explorer_tests",
    srcs = ["test_runner.cpp"],
    args = ["$(locations //explorer:treesitter_testdata)"],
    data = ["//explorer:treesitter_testdata"],
    deps = [
        ":treesitter",
    ],
)
