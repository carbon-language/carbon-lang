# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "parse_node_kind",
    srcs = ["parse_node_kind.cpp"],
    hdrs = ["parse_node_kind.h"],
    textual_hdrs = ["parse_node_kind.def"],
    deps = [
        "//common:check",
        "//common:enum_base",
    ],
)

cc_library(
    name = "parser_state",
    srcs = ["parser_state.cpp"],
    hdrs = ["parser_state.h"],
    textual_hdrs = ["parser_state.def"],
    deps = ["//common:enum_base"],
)

cc_library(
    name = "parse_tree",
    srcs = [
        "parse_tree.cpp",
        "parser_context.cpp",
        "parser_context.h",
    ] +
    # Glob handler files to avoid missing anyway.
    glob([
        "parser_handle_*.cpp",
    ]),
    hdrs = ["parse_tree.h"],
    deps = [
        ":parse_node_kind",
        ":parser_state",
        ":precedence",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:token_kind",
        "//toolchain/lexer:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "parse_tree_test",
    size = "small",
    srcs = ["parse_tree_test.cpp"],
    deps = [
        ":parse_node_kind",
        ":parse_tree",
        "//common:gtest_main",
        "//common:ostream",
        "//toolchain/common:yaml_test_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lexer:tokenized_buffer",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "parse_tree_fuzzer",
    size = "small",
    srcs = ["parse_tree_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":parse_tree",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lexer:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "parse_tree_node_location_translator",
    hdrs = ["parse_tree_node_location_translator.h"],
    deps = [":parse_tree"],
)

cc_library(
    name = "precedence",
    srcs = ["precedence.cpp"],
    hdrs = ["precedence.h"],
    deps = [
        "//common:check",
        "//toolchain/lexer:token_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "precedence_test",
    size = "small",
    srcs = ["precedence_test.cpp"],
    deps = [
        ":precedence",
        "//common:gtest_main",
        "//toolchain/lexer:token_kind",
        "@com_google_googletest//:gtest",
    ],
)

file_test(
    name = "parse_tree_file_test",
    srcs = ["parse_tree_file_test.cpp"],
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        "//testing/file_test:file_test_base",
        "//toolchain/driver",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "dump",
        "parse-tree",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "-v",
        "dump",
        "parse-tree",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)
