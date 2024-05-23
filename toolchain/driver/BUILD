# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@llvm-project//llvm:binary_alias.bzl", "binary_alias")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

# Build rules to construct the install data tree for the Carbon toolchain.
#
# The directory `install` should mirror a Unix-y [FHS] style installation of
# Carbon under some prefix (`/usr`, `/usr/local`, or possibly a home directory
# or a custom prefix). When installed, user facing binaries, which we largely
# expect to be solely the `carbon` driver will be at `install/bin/carbon`. We
# collect data, private executables, and other internals used by the toolchain
# in `install/lib/carbon`. While using a `lib` subdirectory isn't a perfect fit
# for the FHS, having a single location where private data is collected is
# significantly superior to spreading them across the system. This also matches
# similar patterns used by Clang itself and several other language toolchains
# and standard libraries.
#
# Code that needs data files in the toolchain should add them to paths below
# `install/lib/carbon` here and use data dependencies on these paths so that we
# are always testing against a realistic layout.

# Copy Clang and LLVM toolchain files into a synthetic LLVM installation under
# `install/lib/carbon/llvm` so that parts of Clang that expect to find an LLVM
# installation at relative paths work correctly without exposing these as
# user-facing components.
copy_file(
    name = "copy_lld",
    src = "@llvm-project//lld:lld",
    out = "install/lib/carbon/llvm/bin/lld",
    allow_symlink = False,
    is_executable = True,
)

[
    binary_alias(
        name = "install/lib/carbon/llvm/bin/" + bin_name,
        binary = ":copy_lld",
    )
    for bin_name in [
        "ld.lld",
        "ld64.lld",
        "lld-link",
        "wasm-ld",
    ]
]

filegroup(
    name = "llvm_link_data",
    data = ["install/lib/carbon/llvm/bin/" + bin_name for bin_name in [
        "lld",
        "ld.lld",
        "ld64.lld",
        "lld-link",
        "wasm-ld",
    ]],
)

cc_library(
    name = "clang_runner",
    srcs = ["clang_runner.cpp"],
    hdrs = ["clang_runner.h"],
    data = [
        ":llvm_link_data",
    ],
    deps = [
        "//common:command_line",
        "//common:ostream",
        "//common:vlog",
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
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@googletest//:gtest",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "driver",
    srcs = ["driver.cpp"],
    hdrs = ["driver.h"],
    data = ["//core:prelude"],
    textual_hdrs = ["flags.def"],
    deps = [
        ":clang_runner",
        "//common:command_line",
        "//common:vlog",
        "//toolchain/base:value_store",
        "//toolchain/check",
        "//toolchain/codegen",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:sorting_diagnostic_consumer",
        "//toolchain/lex",
        "//toolchain/lower",
        "//toolchain/parse",
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
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/diagnostics:diagnostic_emitter",
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
        "//testing/base:test_raw_ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "carbon",
    srcs = ["driver_main.cpp"],
    env = cc_env(),
    deps = [
        ":driver",
        "//common:all_llvm_targets",
        "//common:bazel_working_dir",
        "//common:init_llvm",
        "@llvm-project//llvm:Support",
    ],
)
