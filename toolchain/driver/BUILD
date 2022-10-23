# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

exports_files(["lit.cfg.py"])

cc_library(
    name = "driver",
    srcs = ["driver.cpp"],
    hdrs = ["driver.h"],
    textual_hdrs = ["flags.def"],
    deps = [
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:sorting_diagnostic_consumer",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/semantics:semantics_ir_factory",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cpp"],
    deps = [
        ":driver",
        "//common:gtest_main",
        "//toolchain/common:yaml_test_helpers",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:tokenized_buffer_test_helpers",
        "@com_google_googletest//:gtest",
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
