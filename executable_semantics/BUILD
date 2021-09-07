# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/testing:lit_test.bzl", "lit_test")
load("test_list.bzl", "TEST_LIST")

cc_binary(
    name = "executable_semantics",
    srcs = ["main.cpp"],
    deps = [
        "//executable_semantics/common:tracing_flag",
        "//executable_semantics/interpreter:exec_program",
        "//executable_semantics/syntax",
        "@llvm-project//llvm:Support",
    ],
)

[
    lit_test(
        name = "%s_test" % x,
        args = ["-v"],
        data = [
            "testdata/lit.cfg",
            ":executable_semantics",
            "testdata/%s.carbon" % x,
        ],
        test_path = "executable_semantics/testdata",
    )
    for x in TEST_LIST
]
