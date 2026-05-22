# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Benchmarking infrastructure as well as comprehensive benchmarks across the
# tolochain. Infrastructure libraries may be depended on by focused benchmarks
# in various other parts of the toolchain.

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")

package(default_visibility = ["//visibility:public"])

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
        ":source_gen_lib",
        "//common:all_llvm_targets",
        "//common:set",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//toolchain/base:install_paths_test_helpers",
        "//toolchain/driver",
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

cc_binary(
    name = "compile_benchmark",
    testonly = 1,
    srcs = ["compile_benchmark.cpp"],
    deps = [
        ":source_gen_lib",
        "//common:all_llvm_targets",
        "//testing/base:benchmark_main",
        "//testing/base:global_exe_path",
        "//toolchain/base:install_paths_test_helpers",
        "//toolchain/driver",
        "//toolchain/driver:clang_runner",
        "//toolchain/testing:compile_helper",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "compile_benchmark_test",
    size = "small",
    srcs = [":compile_benchmark"],
    args = [
        "--benchmark_dry_run",
        # The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=/256$$",
    ],
    env = cc_env(),
)
