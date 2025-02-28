# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Trivial, single-file testing libraries. More complex libraries should get
# their own directory.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

# This does extra initialization on top of googletest's gtest_main in order to
# provide stack traces on unexpected exits, because we normally rely on LLVM
# code for that.
#
# This replaces "@googletest//:gtest_main";
# "@googletest//:gtest" should still be used directly.
cc_library(
    name = "gtest_main",
    testonly = 1,
    srcs = ["gtest_main.cpp"],
    deps = [
        ":global_exe_path",
        "//common:init_llvm",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

# This does extra initialization on top of Google benchmark's main in order to
# provide stack traces and setup LLVM.
#
# This replaces `@google_benchmark//:benchmark_main`;
# `@google_benchmark//:benchmark` should still be used directly.
cc_library(
    name = "benchmark_main",
    testonly = 1,
    srcs = ["benchmark_main.cpp"],
    deps = [
        ":global_exe_path",
        "//common:init_llvm",
        "@abseil-cpp//absl/flags:parse",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "capture_std_streams",
    testonly = 1,
    srcs = ["capture_std_streams.cpp"],
    hdrs = ["capture_std_streams.h"],
    deps = [
        "//common:ostream",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "source_gen_lib",
    testonly = 1,
    srcs = ["source_gen.cpp"],
    hdrs = ["source_gen.h"],
    deps = [
        "//common:check",
        "//common:map",
        "//common:raw_string_ostream",
        "//common:set",
        "//toolchain/lex:token_kind",
        "@abseil-cpp//absl/random",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "source_gen_test",
    size = "small",
    srcs = ["source_gen_test.cpp"],
    deps = [
        ":global_exe_path",
        ":gtest_main",
        ":source_gen_lib",
        "//common:set",
        "//toolchain/driver",
        "//toolchain/install:install_paths_test_helpers",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "source_gen",
    testonly = 1,
    srcs = ["source_gen_main.cpp"],
    deps = [
        ":source_gen_lib",
        "//common:bazel_working_dir",
        "//common:command_line",
        "//common:init_llvm",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "file_helpers",
    testonly = 1,
    srcs = ["file_helpers.cpp"],
    hdrs = ["file_helpers.h"],
    deps = [
        "//common:error",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "global_exe_path",
    testonly = 1,
    srcs = ["global_exe_path.cpp"],
    hdrs = ["global_exe_path.h"],
    deps = [
        "//common:check",
        "//common:exe_path",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "global_exe_path_test",
    size = "small",
    srcs = ["global_exe_path_test.cpp"],
    deps = [
        ":global_exe_path",
        ":gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
