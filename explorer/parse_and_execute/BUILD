# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = [
    "//explorer:__pkg__",
    "//explorer/fuzzing:__pkg__",
])

cc_library(
    name = "parse_and_execute",
    srcs = ["parse_and_execute.cpp"],
    hdrs = ["parse_and_execute.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:check",
        "//common:error",
        "//explorer/base:trace_stream",
        "//explorer/interpreter:exec_program",
        "//explorer/interpreter:stack_space",
        "//explorer/syntax",
        "//explorer/syntax:prelude",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "parse_and_execute_test",
    size = "small",
    srcs = ["parse_and_execute_test.cpp"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":parse_and_execute",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
