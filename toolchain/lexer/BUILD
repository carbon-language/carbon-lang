# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "token_kind",
    srcs = ["token_kind.cpp"],
    hdrs = ["token_kind.h"],
    textual_hdrs = ["token_registry.def"],
    deps = [
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "token_kind_test",
    size = "small",
    srcs = ["token_kind_test.cpp"],
    deps = [
        ":token_kind",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "character_set",
    hdrs = ["character_set.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "lex_helpers",
    srcs = ["lex_helpers.cpp"],
    hdrs = ["lex_helpers.h"],
    deps = [
        "//toolchain/diagnostics:diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "test_helpers",
    testonly = 1,
    hdrs = ["test_helpers.h"],
    deps = [
        "//common:check",
        "//common:string_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "numeric_literal",
    srcs = ["numeric_literal.cpp"],
    hdrs = ["numeric_literal.h"],
    deps = [
        ":character_set",
        ":lex_helpers",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "numeric_literal_benchmark",
    testonly = 1,
    srcs = ["numeric_literal_benchmark.cpp"],
    deps = [
        ":numeric_literal",
        "//common:check",
        "//toolchain/diagnostics:null_diagnostics",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "numeric_literal_test",
    size = "small",
    srcs = ["numeric_literal_test.cpp"],
    deps = [
        ":numeric_literal",
        ":test_helpers",
        "//common:check",
        "//common:gtest_main",
        "//common:ostream",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "numeric_literal_fuzzer",
    size = "small",
    srcs = ["numeric_literal_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/numeric_literal/*"]),
    deps = [
        ":numeric_literal",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "string_literal",
    srcs = ["string_literal.cpp"],
    hdrs = ["string_literal.h"],
    deps = [
        ":character_set",
        ":lex_helpers",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "string_literal_benchmark",
    testonly = 1,
    srcs = ["string_literal_benchmark.cpp"],
    deps = [
        ":string_literal",
        "//toolchain/diagnostics:null_diagnostics",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "string_literal_test",
    size = "small",
    srcs = ["string_literal_test.cpp"],
    deps = [
        ":string_literal",
        ":test_helpers",
        "//common:check",
        "//common:gtest_main",
        "//common:ostream",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "string_literal_fuzzer",
    size = "small",
    srcs = ["string_literal_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/string_literal/*"]),
    deps = [
        ":string_literal",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tokenized_buffer",
    srcs = ["tokenized_buffer.cpp"],
    hdrs = ["tokenized_buffer.h"],
    deps = [
        ":character_set",
        ":lex_helpers",
        ":numeric_literal",
        ":string_literal",
        ":token_kind",
        "//common:check",
        "//common:ostream",
        "//common:string_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tokenized_buffer_test_helpers",
    testonly = 1,
    hdrs = ["tokenized_buffer_test_helpers.h"],
    deps = [
        ":tokenized_buffer",
        "//common:check",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "tokenized_buffer_test",
    size = "small",
    srcs = ["tokenized_buffer_test.cpp"],
    deps = [
        ":tokenized_buffer",
        ":tokenized_buffer_test_helpers",
        "//common:gtest_main",
        "//toolchain/common:yaml_test_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "tokenized_buffer_fuzzer",
    size = "small",
    srcs = ["tokenized_buffer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/tokenized_buffer/*"]),
    deps = [
        ":tokenized_buffer",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@llvm-project//llvm:Support",
    ],
)
