# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/testing:lit_test.bzl", "lit_test")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//executable_semantics:__subpackages__",
])

cc_binary(
    name = "executable_semantics",
    srcs = ["main.cpp"],
    data = ["data/prelude.carbon"],
    deps = [
        "//executable_semantics/common:arena",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/interpreter:exec_program",
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
)

lit_test(
    name = "executable_semantics_lit_test",
    data = [
        "data/prelude.carbon",
        ":executable_semantics",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
    test_dir = "testdata",
)

py_binary(
    name = "gen_rtti",
    srcs = ["gen_rtti.py"],
)
