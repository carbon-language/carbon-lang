# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//explorer:__subpackages__",
    "//installers:__subpackages__",
])

filegroup(
    name = "standard_libraries",
    srcs = ["data/prelude.carbon"],
)

cc_library(
    name = "main",
    srcs = ["main.cpp"],
    hdrs = ["main.h"],
    deps = [
        "//common:error",
        "//explorer/common:trace_stream",
        "//explorer/parse_and_execute",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "explorer",
    srcs = ["main_bin.cpp"],
    env = cc_env(),
    deps = [
        ":main",
        "//common:bazel_working_dir",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "file_test",
    srcs = ["file_test.cpp"],
    # Bazel limits sharding to 50. We have lots of tests, so this should
    # maximize parallelism.
    shard_count = 50,
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        "//explorer/parse_and_execute",
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
