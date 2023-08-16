# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower_to_llvm",
    srcs = ["lower_to_llvm.cpp"],
    hdrs = ["lower_to_llvm.h"],
    deps = [
        ":lowering_context",
        "//toolchain/semantics:semantics_ir",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "lowering_context",
    srcs = [
        "lowering_context.cpp",
        "lowering_function_context.cpp",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "lowering_handle*.cpp",
    ]),
    hdrs = [
        "lowering_context.h",
        "lowering_function_context.h",
    ],
    deps = [
        "//common:check",
        "//common:vlog",
        "//toolchain/semantics:semantics_ir",
        "//toolchain/semantics:semantics_node",
        "//toolchain/semantics:semantics_node_kind",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "lowering_file_test",
    srcs = ["lowering_file_test.cpp"],
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        "//toolchain/driver:driver_file_test_base",
        "@llvm-project//llvm:Support",
    ],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "dump",
        "llvm-ir",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "-v",
        "dump",
        "llvm-ir",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)
