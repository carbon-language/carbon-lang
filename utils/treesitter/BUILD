# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//bazel/check_deps:__pkg__"])

genrule(
    name = "call_tree_sitter",
    srcs = [
        ":grammar.js",
        "//bazel/npm:node_modules/tree-sitter-cli/dir",
    ],
    outs = [
        "src/scanner.c",
        "src/tree_sitter/parser.h",
    ],
    cmd = " ".join([
        "$(NODE_PATH)",
        "./$(execpath //bazel/npm:node_modules/tree-sitter-cli/dir)/bin/tree-sitter-cli",
        "--help",
    ]),
    toolchains = ["@nodejs_toolchains//:resolved_toolchain"],
    tools = ["@nodejs_toolchains//:resolved_toolchain"],
)

cc_library(
    name = "treesitter",
    srcs = [
        "src/parser.c",
        "src/scanner.c",
    ],
    hdrs = ["src/tree_sitter/parser.h"],
    copts = ["-Wno-missing-prototypes"],
    deps = ["@tree-sitter-bazel//:tree-sitter"],
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
