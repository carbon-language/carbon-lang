# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "semantics",
    srcs = ["semantics.cpp"],
    hdrs = ["semantics.h"],
    deps = ["//toolchain/parser:parse_tree"],
)

cc_test(
    name = "semantics_test",
    size = "small",
    srcs = ["semantics_test.cpp"],
    deps = [
        ":semantics",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "//toolchain/source:source_buffer",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
