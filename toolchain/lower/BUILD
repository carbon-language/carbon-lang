# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower",
    srcs = ["lower.cpp"],
    hdrs = ["lower.h"],
    deps = [
        ":context",
        "//toolchain/sem_ir:file",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "context",
    srcs = [
        "file_context.cpp",
        "function_context.cpp",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "handle*.cpp",
    ]),
    hdrs = [
        "file_context.h",
        "function_context.h",
    ],
    deps = [
        "//common:check",
        "//common:vlog",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:node",
        "//toolchain/sem_ir:node_kind",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "lower_file_test",
    srcs = ["lower_file_test.cpp"],
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
