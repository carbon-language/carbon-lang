# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "node_kind",
    srcs = ["node_kind.cpp"],
    hdrs = ["node_kind.h"],
    textual_hdrs = ["node_kind.def"],
    deps = [
        "//common:check",
        "//common:enum_base",
        "//toolchain/lex:token_kind",
    ],
)

cc_library(
    name = "state",
    srcs = ["state.cpp"],
    hdrs = ["state.h"],
    textual_hdrs = ["state.def"],
    deps = ["//common:enum_base"],
)

cc_library(
    name = "tree",
    srcs = [
        "context.cpp",
        "context.h",
        "tree.cpp",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "handle_*.cpp",
    ]),
    hdrs = ["tree.h"],
    deps = [
        ":node_kind",
        ":precedence",
        ":state",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "tree_test",
    size = "small",
    srcs = ["tree_test.cpp"],
    deps = [
        ":node_kind",
        ":tree",
        "//common:ostream",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:yaml_test_helpers",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "parse_fuzzer",
    size = "small",
    srcs = ["parse_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":tree",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tree_node_location_translator",
    hdrs = ["tree_node_location_translator.h"],
    deps = [":tree"],
)

cc_library(
    name = "precedence",
    srcs = ["precedence.cpp"],
    hdrs = ["precedence.h"],
    deps = [
        "//common:check",
        "//toolchain/lex:token_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "precedence_test",
    size = "small",
    srcs = ["precedence_test.cpp"],
    deps = [
        ":precedence",
        "//testing/base:gtest_main",
        "//toolchain/lex:token_kind",
        "@com_google_googletest//:gtest",
    ],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "compile",
        "--phase=parse",
        "--dump-parse-tree",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "-v",
        "compile",
        "--phase=parse",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)
