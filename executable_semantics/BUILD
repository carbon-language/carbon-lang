# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/testing:lit_test.bzl", "lit_test")

cc_binary(
    name = "executable_semantics",
    srcs = ["main.cpp"],
    deps = [
        "//executable_semantics/interpreter:exec_program",
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
)

lit_test(
    name = "executable_semantics_lit_test",
    test_dir = "testdata",
    tools = [":executable_semantics"],
)
