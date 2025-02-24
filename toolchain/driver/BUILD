# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob([
        "testdata/**/*.carbon",
        "testdata/**/*.cpp",
    ]),
)

cc_library(
    name = "clang_runner",
    srcs = ["clang_runner.cpp"],
    hdrs = ["clang_runner.h"],
    deps = [
        ":tool_runner_base",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/install:install_paths",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:clang-driver",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_test(
    name = "clang_runner_test",
    size = "small",
    srcs = ["clang_runner_test.cpp"],
    env = cc_env(),
    deps = [
        ":clang_runner",
        "//common:all_llvm_targets",
        "//common:check",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_binary(
    name = "compile_benchmark",
    testonly = 1,
    srcs = ["compile_benchmark.cpp"],
    deps = [
        ":driver",
        "//testing/base:benchmark_main",
        "//testing/base:global_exe_path",
        "//testing/base:source_gen_lib",
        "//toolchain/install:install_paths_test_helpers",
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
        "--benchmark_min_time=1x",
        # The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=/256$$",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "clang_subcommand.cpp",
        "clang_subcommand.h",
        "codegen_options.cpp",
        "codegen_options.h",
        "compile_subcommand.cpp",
        "compile_subcommand.h",
        "driver.cpp",
        "driver_env.h",
        "format_subcommand.cpp",
        "format_subcommand.h",
        "language_server_subcommand.cpp",
        "language_server_subcommand.h",
        "link_subcommand.cpp",
        "link_subcommand.h",
        "lld_subcommand.cpp",
        "lld_subcommand.h",
    ],
    hdrs = [
        "driver.h",
        "driver_subcommand.h",
    ],
    data = [
        "//toolchain/install:install_data.no_driver",
    ],
    textual_hdrs = ["flags.def"],
    deps = [
        ":clang_runner",
        ":lld_runner",
        "//common:command_line",
        "//common:error",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//common:version",
        "//common:vlog",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/base:shared_value_stores",
        "//toolchain/base:timings",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:sorting_diagnostic_consumer",
        "//toolchain/format",
        "//toolchain/install:install_paths",
        "//toolchain/language_server",
        "//toolchain/lex",
        "//toolchain/lower",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:dump",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:formatter",
        "//toolchain/sem_ir:inst_namer",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cpp"],
    env = cc_env(),
    deps = [
        ":driver",
        "//common:all_llvm_targets",
        "//common:raw_string_ostream",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/install:install_paths",
        "//toolchain/lex:tokenized_buffer_test_helpers",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "driver_fuzzer",
    size = "small",
    srcs = ["driver_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":driver",
        "//common:exe_path",
        "//common:raw_string_ostream",
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/install:install_paths",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "lld_runner",
    srcs = ["lld_runner.cpp"],
    hdrs = ["lld_runner.h"],
    deps = [
        ":tool_runner_base",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/install:install_paths",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",
        "@llvm-project//lld:MachO",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "lld_runner_test",
    size = "small",
    srcs = ["lld_runner_test.cpp"],
    env = cc_env(),
    deps = [
        ":clang_runner",
        ":lld_runner",
        "//common:all_llvm_targets",
        "//common:check",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "tool_runner_base",
    srcs = ["tool_runner_base.cpp"],
    hdrs = ["tool_runner_base.h"],
    data = [
        "//toolchain/install:install_data.no_driver",
    ],
    deps = [
        "//common:ostream",
        "//common:vlog",
        "//toolchain/install:install_paths",
        "@llvm-project//llvm:Support",
    ],
)
