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
    # The subprocess benchmarks execute the installed toolchain binaries.
    data = ["//toolchain/install:install_data"],
    deps = [
        ":source_gen_lib",
        "//common:all_llvm_targets",
        "//common:filesystem",
        "//testing/base:benchmark_main",
        "//testing/base:global_exe_path",
        "//toolchain/base:install_paths",
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
        # Dry-run the smallest (256-line) case of every benchmark. The optional
        # `/real_time` suffix is present on the subprocess benchmarks, which use
        # real time. The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=/256(/real_time)?$$",
    ],
    env = cc_env(),
)

cc_binary(
    name = "prelude_benchmark",
    testonly = 1,
    srcs = ["prelude_benchmark.cpp"],
    deps = [
        "//common:all_llvm_targets",
        "//common:check",
        "//testing/base:benchmark_main",
        "//testing/base:global_exe_path",
        "//toolchain/base:install_paths",
        "//toolchain/base:install_paths_test_helpers",
        "//toolchain/base:mem_usage",
        "//toolchain/driver",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "prelude_benchmark_test",
    size = "small",
    srcs = [":prelude_benchmark"],
    # No `--benchmark_filter` is needed (unlike `compile_benchmark_test`): the
    # benchmarks aren't `Range()`-parameterized, so a dry run is already cheap.
    args = ["--benchmark_dry_run"],
    env = cc_env(),
)

cc_binary(
    name = "format_benchmark",
    testonly = 1,
    srcs = ["format_benchmark.cpp"],
    deps = [
        ":source_gen_lib",
        "//common:check",
        "//common:raw_string_ostream",
        "//testing/base:benchmark_main",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/format",
        "//toolchain/lex",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
        "//toolchain/testing:compile_helper",
        "@google_benchmark//:benchmark",
        "@llvm-project//clang:format",
        "@llvm-project//clang:tooling_core",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "format_benchmark_test",
    size = "small",
    srcs = [":format_benchmark"],
    args = [
        "--benchmark_dry_run",
        # The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=/256$$",
    ],
    env = cc_env(),
)
