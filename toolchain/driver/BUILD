# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//testing/file_test:rules.bzl", "file_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "driver",
    srcs = ["driver.cpp"],
    hdrs = ["driver.h"],
    textual_hdrs = ["flags.def"],
    deps = [
        "//common:command_line",
        "//common:vlog",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:sorting_diagnostic_consumer",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/lower",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:formatter",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

file_test(
    name = "driver_file_test",
    srcs = ["driver_file_test.cpp"],
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        ":driver_file_test_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cpp"],
    deps = [
        ":driver",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/base:yaml_test_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:tokenized_buffer_test_helpers",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "driver_fuzzer",
    size = "small",
    srcs = ["driver_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":driver",
        "//testing/base:test_raw_ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "driver_file_test_base",
    testonly = 1,
    hdrs = ["driver_file_test_base.h"],
    deps = [
        ":driver",
        "//testing/file_test:file_test_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "carbon",
    srcs = ["driver_main.cpp"],
    env = cc_env(),
    deps = [
        ":driver",
        "//common:bazel_working_dir",
        "@llvm-project//llvm:Support",
    ],
)
