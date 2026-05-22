# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**/*.carbon",
        "testdata/**/*.cpp",
    ]),
)

cc_library(
    name = "clang_runner",
    srcs = [
        "clang_runner.cpp",
        "clang_runtimes.cpp",
    ],
    hdrs = [
        "clang_runner.h",
        "clang_runtimes.h",
    ],
    deps = [
        ":llvm_runner",
        ":runtimes_cache",
        ":tool_runner_base",
        "//common:check",
        "//common:error",
        "//common:filesystem",
        "//common:latch",
        "//common:ostream",
        "//common:string_helpers",
        "//common:vlog",
        "//third_party/llvm:clang_cc1",
        "//toolchain/base:clang_invocation",
        "//toolchain/base:install_paths",
        "//toolchain/base:kind_switch",
        "//toolchain/base:runtimes_build_info",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:clang-driver",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_test(
    name = "clang_runner_test",
    size = "small",
    srcs = ["clang_runner_test.cpp"],
    data = ["//toolchain/install:install_data"],
    deps = [
        ":clang_runner",
        ":llvm_runner",
        ":runtimes_cache",
        "//common:all_llvm_targets",
        "//common:check",
        "//common:error_test_helpers",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//toolchain/base:install_paths",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_test(
    name = "clang_runtimes_test",
    size = "medium",
    srcs = ["clang_runtimes_test.cpp"],
    data = [
        "//toolchain/install:built_runtimes",
        "//toolchain/install:install_data",
    ],
    deps = [
        ":clang_runner",
        ":llvm_runner",
        ":runtimes_cache",
        "//common:all_llvm_targets",
        "//common:check",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//toolchain/base:install_paths",
        "//toolchain/base:llvm_tools",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "codegen_options",
    srcs = [
        "codegen_options.cpp",
    ],
    hdrs = [
        "codegen_options.h",
    ],
    deps = [
        "//common:command_line",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "build_runtimes_subcommand.cpp",
        "build_runtimes_subcommand.h",
        "clang_subcommand.cpp",
        "clang_subcommand.h",
        "compile_subcommand.cpp",
        "compile_subcommand.h",
        "config_subcommand.cpp",
        "config_subcommand.h",
        "driver.cpp",
        "driver_env.h",
        "driver_subcommand.cpp",
        "format_subcommand.cpp",
        "format_subcommand.h",
        "language_server_subcommand.cpp",
        "language_server_subcommand.h",
        "link_subcommand.cpp",
        "link_subcommand.h",
        "lld_subcommand.cpp",
        "lld_subcommand.h",
        "llvm_subcommand.cpp",
        "llvm_subcommand.h",
    ],
    hdrs = [
        "driver.h",
        "driver_subcommand.h",
    ],
    data = [
        "//toolchain/install:all_data_files",
    ],
    textual_hdrs = ["flags.def"],
    deps = [
        ":clang_runner",
        ":codegen_options",
        ":lld_runner",
        ":llvm_runner",
        ":runtimes_cache",
        "//common:check",
        "//common:command_line",
        "//common:error",
        "//common:filesystem",
        "//common:ostream",
        "//common:pretty_stack_trace_function",
        "//common:raw_string_ostream",
        "//common:version",
        "//common:vlog",
        "//toolchain/base:clang_invocation",
        "//toolchain/base:install_paths",
        "//toolchain/base:llvm_tools",
        "//toolchain/base:shared_value_stores",
        "//toolchain/base:timings",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:emitter",
        "//toolchain/diagnostics:format_providers",
        "//toolchain/diagnostics:sorting_consumer",
        "//toolchain/format",
        "//toolchain/language_server",
        "//toolchain/lex",
        "//toolchain/lower",
        "//toolchain/lower:options",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:typed_insts",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:lex",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cpp"],
    data = ["//toolchain/install:install_data"],
    deps = [
        ":driver",
        "//common:all_llvm_targets",
        "//common:error_test_helpers",
        "//common:filesystem",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//toolchain/base:install_paths",
        "//toolchain/diagnostics:emitter",
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
        "//toolchain/base:install_paths",
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
        "//common:string_helpers",
        "//common:vlog",
        "//toolchain/base:install_paths",
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
    name = "llvm_runner",
    srcs = ["llvm_runner.cpp"],
    hdrs = ["llvm_runner.h"],
    deps = [
        ":tool_runner_base",
        "//common:ostream",
        "//common:string_helpers",
        "//common:vlog",
        "//toolchain/base:install_paths",
        "//toolchain/base:llvm_tools",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",
        "@llvm-project//lld:MachO",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "llvm_runner_test",
    size = "small",
    srcs = ["llvm_runner_test.cpp"],
    deps = [
        ":llvm_runner",
        "//common:all_llvm_targets",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:capture_std_streams",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "runtimes_cache",
    srcs = ["runtimes_cache.cpp"],
    hdrs = ["runtimes_cache.h"],
    deps = [
        "//common:check",
        "//common:error",
        "//common:filesystem",
        "//common:ostream",
        "//common:version",
        "//common:vlog",
        "//toolchain/base:install_paths",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "runtimes_cache_test",
    size = "small",
    srcs = ["runtimes_cache_test.cpp"],
    data = ["//toolchain/install:install_data"],
    deps = [
        ":runtimes_cache",
        "//common:check",
        "//common:error_test_helpers",
        "//common:filesystem",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//common:version",
        "//testing/base:capture_std_streams",
        "//testing/base:file_helpers",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tool_runner_base",
    srcs = ["tool_runner_base.cpp"],
    hdrs = ["tool_runner_base.h"],
    data = [
        "//toolchain/install:all_data_files",
    ],
    deps = [
        "//common:ostream",
        "//common:vlog",
        "//toolchain/base:install_paths",
        "@llvm-project//llvm:Support",
    ],
)
