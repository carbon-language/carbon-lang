# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_tree_sitter//tree_sitter:tree_sitter.bzl", "tree_sitter_cc_library")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")

tree_sitter_cc_library(
    name = "treesitter_carbon",
    srcs = ["src/scanner.c"],
    grammar = ["grammar.js"],
)

cc_binary(
    name = "treesitter_carbon_tester",
    srcs = ["test.cpp"],
    deps = [
        ":treesitter_carbon",
    ],
)

glob_sh_run(
    args = [
        "$(location :treesitter_carbon_tester)",
    ],
    data = [":treesitter_carbon_tester"],
    file_exts = ["carbon"],
)
