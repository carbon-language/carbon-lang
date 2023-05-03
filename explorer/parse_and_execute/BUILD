# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "add_file_tests")

package(default_visibility = [
    "//explorer:__pkg__",
    "//explorer/fuzzing:__pkg__",
])

cc_library(
    name = "parse_and_execute",
    srcs = ["parse_and_execute.cpp"],
    hdrs = ["parse_and_execute.h"],
    deps = [
        "//common:check",
        "//common:error",
        "//explorer/common:trace_stream",
        "//explorer/interpreter:exec_program",
        "//explorer/interpreter:stack_space",
        "//explorer/syntax",
        "//explorer/syntax:prelude",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "parse_and_execute_test",
    srcs = ["parse_and_execute_test.cpp"],
    deps = [
        ":parse_and_execute",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

add_file_tests(
    name = "file_test",
    srcs = ["file_test.cpp"],
    # Bazel limits sharding to 50. We have lots of tests, so this should
    # maximize parallelism.
    shard_count = 50,
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        ":parse_and_execute",
        "//testing/file_test:file_test_base",
    ],
)

glob_sh_run(
    args = ["$(location //explorer)"],
    data = ["//explorer"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //explorer)",
        "--parser_debug",
        "--trace_file=-",
    ],
    data = ["//explorer"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)

filegroup(
    name = "carbon_files",
    srcs = glob(["testdata/**/*.carbon"]),
    # Files are used for validating fuzzer completeness.
    visibility = ["//explorer/fuzzing:__pkg__"],
)
