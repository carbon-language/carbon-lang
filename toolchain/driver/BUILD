# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "driver",
    srcs = ["driver.cpp"],
    hdrs = ["driver.h"],
    textual_hdrs = ["flags.def"],
    deps = [
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "driver_test",
    srcs = ["driver_test.cpp"],
    deps = [
        ":driver",
        "//toolchain/common:yaml_test_helpers",
        "//toolchain/lexer:tokenized_buffer_test_helpers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "driver_fuzzer",
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
    deps = [
        ":driver",
        "@llvm-project//llvm:Support",
    ],
)

# FIXME: No support for LLVM's lit-style command line & FileCheck tests.
#
#lit_test(
#    name = "carbon_test.carbon",
#    data = [
#        ":carbon",
#        "@llvm-project//llvm:FileCheck",
#    ],
#)
