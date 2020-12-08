# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "token_kind",
    srcs = ["token_kind.cpp"],
    hdrs = ["token_kind.h"],
    textual_hdrs = ["token_registry.def"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_test(
    name = "token_kind_test",
    srcs = ["token_kind_test.cpp"],
    deps = [
        ":token_kind",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

cc_library(
    name = "tokenized_buffer",
    srcs = ["tokenized_buffer.cpp"],
    hdrs = ["tokenized_buffer.h"],
    deps = [
        ":token_kind",
        "//diagnostics:diagnostic_emitter",
        "//source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tokenized_buffer_test_helpers",
    testonly = 1,
    hdrs = ["tokenized_buffer_test_helpers.h"],
    deps = [
        ":tokenized_buffer",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gmock",
    ],
)

cc_test(
    name = "tokenized_buffer_test",
    srcs = ["tokenized_buffer_test.cpp"],
    deps = [
        ":tokenized_buffer",
        ":tokenized_buffer_test_helpers",
        "//diagnostics:diagnostic_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gmock",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

cc_fuzz_test(
    name = "tokenized_buffer_fuzzer",
    srcs = ["tokenized_buffer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":tokenized_buffer",
        "//diagnostics:diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)
