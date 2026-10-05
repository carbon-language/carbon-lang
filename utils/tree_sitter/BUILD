# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:make_include_copts.bzl", "make_include_copts")

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
        ":string_fail_tests",
        ":string_tests",
        ":toolchain_testdata_tests",
    ],
)

# Call tree-sitter to generate parser files.
genrule(
    name = "parser_files",
    srcs = ["grammar.js"],
    outs = [
        "parser.c",
        "tree_sitter/parser.h",
    ],
    cmd = "tree-sitter generate $(location grammar.js) -o $(RULEDIR)",
    tags = ["manual"],
)

cc_library(
    name = "parser",
    srcs = [
        "src/scanner.c",
        ":parser.c",
    ],
    hdrs = [":tree_sitter/parser.h"],
    copts = make_include_copts(["."]) + ["-Wno-missing-prototypes"],
    tags = ["manual"],
    deps = ["@tree-sitter-bazel//:tree-sitter"],
)

cc_shared_library(
    name = "parser_shared",
    shared_lib_name = "carbon.so",
    tags = ["manual"],
    deps = [":parser"],
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

# Parses the toolchain's testdata. Files and split files named `fail_*` are
# skipped, as are non-Carbon split files; everything else is expected to parse
# without errors.
#
# Some testdata is excluded because it isn't expected to parse:
# - lex and format tests only tokenize their input.
# - language_server tests don't follow the `fail_*` naming convention for files
#   that are not expected to parse.
cc_test(
    name = "toolchain_testdata_tests",
    size = "small",
    srcs = ["test_runner.cpp"],
    args = [
        "$(locations //toolchain/check:testdata)",
        "$(locations //toolchain/codegen:testdata)",
        "$(locations //toolchain/driver:testdata)",
        "$(locations //toolchain/lower:testdata)",
        "$(locations //toolchain/parse:testdata)",
    ],
    data = [
        "//toolchain/check:testdata",
        "//toolchain/codegen:testdata",
        "//toolchain/driver:testdata",
        "//toolchain/lower:testdata",
        "//toolchain/parse:testdata",
    ],
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
