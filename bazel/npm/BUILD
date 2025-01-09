# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Empty; only defs.bzl is needed.

load("@npm//:defs.bzl", "npm_link_all_packages")
load("@aspect_bazel_lib//lib:directory_path.bzl", "directory_path")
load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary", "js_test")

npm_link_all_packages(
    name = "node_modules",
)

directory_path(
    name = "tree_sitter_entry_point",
    directory = ":node_modules/tree-sitter-cli/dir",
    path = "bin/tree-sitter-cli",
)

js_binary(
    name = "tree-sitter",
    entry_point = ":tree_sitter_entry_point",
)
