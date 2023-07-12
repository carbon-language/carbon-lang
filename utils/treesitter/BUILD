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

tests = glob(
    ["testdata/**/*.carbon"],
    exclude = [
        "testdata/**/fail_*",

        # multiline strings
        "testdata/explorer/string/*",
    ],
)

cc_test(
    name = "tests",
    srcs = ["test_runner.cpp"],
    args = ["$(location {0})".format(x) for x in tests],
    data = tests,
    deps = [
        ":treesitter",
    ],
)
