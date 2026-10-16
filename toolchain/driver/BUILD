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
    name = "driver_env",
    hdrs = ["driver_env.h"],
    deps = [
        ":runtimes_cache",
        "//common:ostream",
        "//toolchain/base:install_paths",
        "//toolchain/diagnostics:emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "driver_subcommand",
    srcs = ["driver_subcommand.cpp"],
    hdrs = ["driver_subcommand.h"],
    deps = [
        ":driver_env",
        "//common:command_line",
        "//common:ostream",
        "//toolchain/base:install_paths",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "compile_options",
    srcs = ["compile_options.cpp"],
    hdrs = ["compile_options.h"],
    deps = [
        ":codegen_options",
        ":driver_env",
        "//common:command_line",
        "//common:error",
        "//common:ostream",
        "//toolchain/base:clang_invocation",
        "//toolchain/check",
        "//toolchain/diagnostics:emitter",
        "//toolchain/lower:options",
        "//toolchain/parse",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "compile_driver",
    srcs = ["compile_driver.cpp"],
    hdrs = ["compile_driver.h"],
    deps = [
        ":codegen_options",
        ":compile_options",
        ":driver_env",
        ":runtimes_cache",
        "//common:error",
        "//common:ostream",
        "//common:pretty_stack_trace_function",
        "//common:vlog",
        "//toolchain/base:clang_invocation",
        "//toolchain/base:install_paths",
        "//toolchain/base:shared_value_stores",
        "//toolchain/base:timings",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:emitter",
        "//toolchain/diagnostics:format_providers",
        "//toolchain/diagnostics:renderer",
        "//toolchain/diagnostics:sorting_consumer",
        "//toolchain/lex",
        "//toolchain/lower",
        "//toolchain/lower:options",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:typed_insts",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "compile_subcommand",
    srcs = ["compile_subcommand.cpp"],
    hdrs = ["compile_subcommand.h"],
    deps = [
        ":compile_driver",
        ":compile_options",
        ":driver_env",
        ":driver_subcommand",
        "//common:command_line",
        "//toolchain/diagnostics:emitter",
        "//toolchain/diagnostics:format_providers",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "format_subcommand",
    srcs = ["format_subcommand.cpp"],
    hdrs = ["format_subcommand.h"],
    deps = [
        ":driver_env",
        ":driver_subcommand",
        "//common:command_line",
        "//common:error",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//toolchain/base:install_paths",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:emitter",
        "//toolchain/format",
        "//toolchain/lex",
        "//toolchain/source:source_buffer",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "language_server_subcommand",
    srcs = ["language_server_subcommand.cpp"],
    hdrs = ["language_server_subcommand.h"],
    deps = [
        ":codegen_options",
        ":driver_env",
        ":driver_subcommand",
        "//common:command_line",
        "//toolchain/diagnostics:emitter",
        "//toolchain/language_server",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "carbon_runtimes",
    srcs = ["carbon_runtimes.cpp"],
    hdrs = ["carbon_runtimes.h"],
    deps = [
        ":codegen_options",
        ":compile_driver",
        ":compile_options",
        ":driver_env",
        ":runtimes_cache",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:string_helpers",
        "//common:vlog",
        "//toolchain/base:install_paths",
        "//toolchain/base:runtimes_build_info",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "link_subcommand",
    srcs = [
        "link_driver.cpp",
        "link_options.cpp",
        "link_subcommand.cpp",
    ],
    hdrs = [
        "link_driver.h",
        "link_options.h",
        "link_subcommand.h",
    ],
    deps = [
        ":carbon_runtimes",
        ":clang_runner",
        ":codegen_options",
        ":driver_env",
        ":driver_subcommand",
        "//common:check",
        "//common:command_line",
        "//common:error",
        "//common:filesystem",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/base:install_paths",
        "//toolchain/base:runtimes_build_info",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "build_subcommand",
    srcs = ["build_subcommand.cpp"],
    hdrs = ["build_subcommand.h"],
    deps = [
        ":codegen_options",
        ":compile_driver",
        ":compile_options",
        ":driver_env",
        ":driver_subcommand",
        ":link_subcommand",
        "//common:command_line",
        "//common:error",
        "//common:filesystem",
        "//common:hashing",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "build_runtimes_subcommand",
    srcs = ["build_runtimes_subcommand.cpp"],
    hdrs = ["build_runtimes_subcommand.h"],
    deps = [
        ":carbon_runtimes",
        ":clang_runner",
        ":codegen_options",
        ":driver_env",
        ":driver_subcommand",
        ":runtimes_cache",
        "//common:command_line",
        "//common:error",
        "//toolchain/diagnostics:emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "config_subcommand",
    srcs = ["config_subcommand.cpp"],
    hdrs = ["config_subcommand.h"],
    deps = [
        ":clang_runner",
        ":codegen_options",
        ":driver_env",
        ":driver_subcommand",
        "//common:check",
        "//common:command_line",
        "//common:filesystem",
        "//common:ostream",
        "//common:version",
        "//toolchain/base:clang_invocation",
        "//toolchain/base:install_paths",
        "//toolchain/diagnostics:emitter",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:lex",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "clang_subcommand",
    srcs = ["clang_subcommand.cpp"],
    hdrs = ["clang_subcommand.h"],
    deps = [
        ":clang_runner",
        ":driver_env",
        ":driver_subcommand",
        "//common:command_line",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "lld_subcommand",
    srcs = ["lld_subcommand.cpp"],
    hdrs = ["lld_subcommand.h"],
    deps = [
        ":driver_env",
        ":driver_subcommand",
        ":lld_runner",
        "//common:command_line",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "llvm_subcommand",
    srcs = ["llvm_subcommand.cpp"],
    hdrs = ["llvm_subcommand.h"],
    deps = [
        ":driver_env",
        ":driver_subcommand",
        ":llvm_runner",
        "//common:check",
        "//common:command_line",
        "//toolchain/base:llvm_tools",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "driver",
    srcs = [
        "driver.cpp",
    ],
    hdrs = [
        "driver.h",
    ],
    data = [
        "//toolchain/install:all_data_files",
    ],
    textual_hdrs = ["flags.def"],
    deps = [
        ":build_runtimes_subcommand",
        ":build_subcommand",
        ":clang_subcommand",
        ":compile_subcommand",
        ":config_subcommand",
        ":driver_env",
        ":driver_subcommand",
        ":format_subcommand",
        ":language_server_subcommand",
        ":link_subcommand",
        ":lld_subcommand",
        ":llvm_subcommand",
        ":runtimes_cache",
        "//common:check",
        "//common:command_line",
        "//common:error",
        "//common:filesystem",
        "//common:ostream",
        "//common:pretty_stack_trace_function",
        "//common:version",
        "//common/terminal:capabilities",
        "//toolchain/diagnostics:renderer",
        "@llvm-project//llvm:Support",
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
        "//toolchain/diagnostics:renderer",
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
