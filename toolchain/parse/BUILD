# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

manifest(
    name = "testdata.txt",
    srcs = [":testdata"],
)

cc_library(
    name = "node_category",
    srcs = ["node_category.cpp"],
    hdrs = ["node_category.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "node_kind",
    srcs = ["node_kind.cpp"],
    hdrs = [
        "node_ids.h",
        "node_kind.h",
        "typed_nodes.h",
    ],
    textual_hdrs = ["node_kind.def"],
    deps = [
        ":node_category",
        "//common:check",
        "//common:enum_base",
        "//common:ostream",
        "//toolchain/base:index_base",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "typed_nodes_test",
    size = "small",
    srcs = ["typed_nodes_test.cpp"],
    deps = [
        ":node_kind",
        ":parse",
        ":tree",
        "//testing/base:gtest_main",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:compile_helper",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cpp"],
    hdrs = ["context.h"],
    deps = [
        ":node_kind",
        ":precedence",
        ":state",
        ":tree",
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:format_providers",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "parse",
    srcs = ["parse.cpp"] +
    # Glob handler files to avoid missing any.
    glob([
        "handle_*.cpp",
    ]),
    hdrs = [
        "handle.h",
        "parse.h",
    ],
    deps = [
        ":context",
        ":dump",
        ":node_kind",
        ":state",
        ":tree",
        "//common:check",
        "//common:ostream",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:format_providers",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cpp"],
    hdrs = ["dump.h"],
    deps = [
        ":tree",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//toolchain/lex:dump",
    ],
    # Always link dump methods.
    alwayslink = 1,
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
        "extract.cpp",
        "tree.cpp",
        "tree_and_subtrees.cpp",
    ],
    hdrs = [
        "tree.h",
        "tree_and_subtrees.h",
    ],
    deps = [
        ":node_kind",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:struct_reflection",
        "//toolchain/base:value_store",
        "//toolchain/lex:token_index",
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
        ":parse",
        ":tree",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/testing:compile_helper",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "parse_fuzzer",
    size = "small",
    srcs = ["parse_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":parse",
        "//common:check",
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex",
        "@llvm-project//llvm:Support",
    ],
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
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "coverage_test",
    size = "small",
    srcs = ["coverage_test.cpp"],
    args = ["--testdata_manifest=$(location :testdata.txt)"],
    data = [
        ":testdata",
        ":testdata.txt",
    ],
    deps = [
        ":node_kind",
        "//testing/base:gtest_main",
        "//toolchain/testing:coverage_helper",
        "@abseil-cpp//absl/flags:flag",
        "@googletest//:gtest",
    ],
)
