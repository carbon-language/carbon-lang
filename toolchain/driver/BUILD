# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "clang_runner",
    srcs = ["clang_runner.cpp"],
    hdrs = ["clang_runner.h"],
    data = [
        "//toolchain/install:llvm_link_data",
    ],
    deps = [
        "//common:command_line",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/install:install_paths",
        "@llvm-project//clang:basic",
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
    deps = [
        ":clang_runner",
        "//common:all_llvm_targets",
        "//common:check",
        "//common:ostream",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
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
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "compile_benchmark_test",
    srcs = ["compile_benchmark_test.sh"],
    data = [":compile_benchmark"],
)

cc_library(
    name = "driver",
    srcs = ["driver.cpp"],
    hdrs = ["driver.h"],
    data = [
        "//toolchain/install:install_lib_data",
    ],
    textual_hdrs = ["flags.def"],
    deps = [
        ":clang_runner",
        "//common:command_line",
        "//common:version",
        "//common:vlog",
        "//toolchain/base:value_store",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:sorting_diagnostic_consumer",
        "//toolchain/install:install_paths",
        "//toolchain/lex",
        "//toolchain/lower",
        "//toolchain/parse",
        "//toolchain/parse:tree",
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
    deps = [
        ":driver",
        "//common:all_llvm_targets",
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
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
        "//testing/base:test_raw_ostream",
        "//testing/fuzzing:libfuzzer_header",
        "//toolchain/install:install_paths",
        "@llvm-project//llvm:Support",
    ],
)

# This target doesn't include prelude libraries. To get a target that has the
# prelude available, use //toolchain.
cc_binary(
    name = "carbon",
    srcs = ["driver_main.cpp"],
    env = cc_env(),
    deps = [
        ":driver",
        "//common:all_llvm_targets",
        "//common:bazel_working_dir",
        "//common:exe_path",
        "//common:init_llvm",
        "//common:version_stamp",
        "//toolchain/install:install_paths",
        "@llvm-project//llvm:Support",
    ],
)
