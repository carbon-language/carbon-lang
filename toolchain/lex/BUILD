# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "token_kind",
    srcs = ["token_kind.cpp"],
    hdrs = ["token_kind.h"],
    textual_hdrs = ["token_kind.def"],
    deps = [
        "//common:check",
        "//common:enum_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "token_kind_test",
    size = "small",
    srcs = ["token_kind_test.cpp"],
    deps = [
        ":token_kind",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "character_set",
    hdrs = ["character_set.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "helpers",
    srcs = ["helpers.cpp"],
    hdrs = ["helpers.h"],
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
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "numeric_literal",
    srcs = ["numeric_literal.cpp"],
    hdrs = ["numeric_literal.h"],
    deps = [
        ":character_set",
        ":helpers",
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:format_providers",
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
        "//testing/base:benchmark_main",
        "//toolchain/diagnostics:null_diagnostics",
        "@google_benchmark//:benchmark",
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
        "//common:ostream",
        "//testing/base:gtest_main",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@googletest//:gtest",
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
        "//testing/fuzzing:libfuzzer_header",
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
        ":helpers",
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
        "//testing/base:benchmark_main",
        "//toolchain/diagnostics:null_diagnostics",
        "@google_benchmark//:benchmark",
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
        "//common:ostream",
        "//testing/base:gtest_main",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@googletest//:gtest",
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
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "lex",
    srcs = ["lex.cpp"],
    hdrs = ["lex.h"],
    deps = [
        ":character_set",
        ":dump",
        ":helpers",
        ":numeric_literal",
        ":string_literal",
        ":token_index",
        ":token_kind",
        ":tokenized_buffer",
        "//common:check",
        "//common:variant_helpers",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cpp"],
    hdrs = ["dump.h"],
    deps = [
        ":tokenized_buffer",
        "//common:ostream",
        "//common:raw_string_ostream",
    ],
    # Always link dump methods.
    alwayslink = 1,
)

cc_library(
    name = "token_index",
    hdrs = ["token_index.h"],
    deps = [
        ":token_kind",
        "//toolchain/base:index_base",
    ],
)

cc_library(
    name = "tokenized_buffer",
    srcs = ["tokenized_buffer.cpp"],
    hdrs = ["tokenized_buffer.h"],
    deps = [
        ":character_set",
        ":helpers",
        ":numeric_literal",
        ":string_literal",
        ":token_index",
        ":token_kind",
        "//common:check",
        "//common:ostream",
        "//common:string_helpers",
        "//toolchain/base:index_base",
        "//toolchain/base:mem_usage",
        "//toolchain/base:shared_value_stores",
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
        ":lex",
        ":tokenized_buffer",
        "//common:check",
        "//toolchain/base:shared_value_stores",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "tokenized_buffer_test",
    size = "small",
    srcs = ["tokenized_buffer_test.cpp"],
    deps = [
        ":lex",
        ":token_kind",
        ":tokenized_buffer",
        ":tokenized_buffer_test_helpers",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:mocks",
        "//toolchain/testing:compile_helper",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "tokenized_buffer_fuzzer",
    size = "small",
    srcs = ["tokenized_buffer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/tokenized_buffer/*"]),
    deps = [
        ":lex",
        "//common:check",
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "tokenized_buffer_benchmark",
    testonly = 1,
    srcs = ["tokenized_buffer_benchmark.cpp"],
    deps = [
        ":lex",
        ":token_kind",
        ":tokenized_buffer",
        "//common:check",
        "//common:raw_string_ostream",
        "//testing/base:benchmark_main",
        "//testing/base:source_gen_lib",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:null_diagnostics",
        "@abseil-cpp//absl/random",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)
