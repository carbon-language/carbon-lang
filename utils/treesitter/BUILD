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
    srcs = ["test.cpp"],
    deps = [
        ":treesitter",
    ],
)

tests = glob(
    ["testdata/**/*.carbon"],
    exclude = [
        "testdata/**/fail_*",
        "testdata/function/declaration/addr.carbon",
        "testdata/generics/generic_params/template_addr.carbon",
        "testdata/operators/recover_infix_uneven_space_before.carbon",
        "testdata/operators/recover_postfix_space_surrounding.carbon",
    ],
)

cc_test(
    name = "tests",
    srcs = ["test.cpp"],
    args = ["$(location {0})".format(x) for x in tests],
    data = tests,
    deps = [
        ":treesitter",
    ],
)
