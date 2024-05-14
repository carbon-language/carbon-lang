# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")
load("@rules_tree_sitter//tree_sitter:tree_sitter.bzl", "tree_sitter_cc_library")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//installers:__subpackages__",
])

tree_sitter_cc_library(
    name = "treesitter",
    srcs = ["src/scanner.c"],
    grammar = ["grammar.js"],
)

cc_binary(
    name = "test_runner",
    testonly = 1,
    srcs = ["test_runner.cpp"],
    deps = [
        ":treesitter",
    ],
)

cc_test(
    name = "explorer_tests",
    size = "small",
    srcs = ["test_runner.cpp"],
    args = ["$(locations //explorer:treesitter_testdata)"],
    data = ["//explorer:treesitter_testdata"],
    deps = [
        ":treesitter",
    ],
)

filegroup(
    name = "string_testdata",
    srcs = glob(
        ["testdata/string/*.carbon"],
        exclude = ["testdata/string/fail_*.carbon"],
    ),
)

filegroup(
    name = "string_fail_testdata",
    srcs = glob(["testdata/string/fail_*.carbon"]),
)

cc_test(
    name = "string_tests",
    size = "small",
    srcs = ["test_runner.cpp"],
    args = ["$(locations :string_testdata)"],
    data = [":string_testdata"],
    deps = [
        ":treesitter",
    ],
)

cc_test(
    name = "string_fail_tests",
    size = "small",
    srcs = ["test_runner.cpp"],
    args = ["$(locations :string_fail_testdata)"],
    data = [":string_fail_testdata"],
    env = {
        "FAIL_TESTS": "1",
    },
    deps = [
        ":treesitter",
    ],
)
