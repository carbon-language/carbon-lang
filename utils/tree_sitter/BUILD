# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//bazel/check_deps:__pkg__"])

# See README.md for instructions on tree-sitter setup and use. These rules are
# manual because the tree-sitter invocation is non-hermetic, and most developers
# won't have it installed; we don't want to break "bazel test //..." if we can
# avoid it.
#
# We use tree-sitter non-hermetically for two key reasons:
#
# - The main way of hermetically using npms in bazel, `aspect_rules_js`, uses
#   declare_symlink; we disallow that for important compatibility reasons.
# - When generated, src/parser.c is over 500 KB, which is larger than we want to
#   check in. It should also be expected to grow if the grammar becomes more complete.

# Convenience target for running all tests, including manual tests.
test_suite(
    name = "tests",
    tags = ["manual"],
    tests = [
        ":explorer_tests",
        ":string_fail_tests",
        ":string_tests",
    ],
)

# Call tree-sitter to generate parser files.
genrule(
    name = "parser_files",
    srcs = ["grammar.js"],
    outs = [
        "src/parser.c",
        "src/tree_sitter/parser.h",
    ],
    cmd = "tree-sitter generate $(location grammar.js) &&\n" +
          "cp src/parser.c $(location src/parser.c) &&\n" +
          "cp src/tree_sitter/parser.h $(location src/tree_sitter/parser.h)",
    tags = ["manual"],
)

cc_library(
    name = "parser",
    srcs = [
        "src/scanner.c",
        ":src/parser.c",
    ],
    hdrs = [":src/tree_sitter/parser.h"],
    copts = ["-Wno-missing-prototypes"],
    tags = ["manual"],
    deps = ["@tree-sitter-bazel//:tree-sitter"],
)

cc_binary(
    name = "test_runner",
    testonly = 1,
    srcs = ["test_runner.cpp"],
    tags = ["manual"],
    deps = [
        ":parser",
        "//testing/base:file_helpers",
    ],
)

cc_test(
    name = "explorer_tests",
    size = "small",
    srcs = ["test_runner.cpp"],
    args = ["$(locations //explorer:tree_sitter_testdata)"],
    data = ["//explorer:tree_sitter_testdata"],
    tags = ["manual"],
    deps = [
        ":parser",
        "//testing/base:file_helpers",
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
    tags = ["manual"],
    deps = [
        ":parser",
        "//testing/base:file_helpers",
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
    tags = ["manual"],
    deps = [
        ":parser",
        "//testing/base:file_helpers",
    ],
)
