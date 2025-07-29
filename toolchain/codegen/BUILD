# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "codegen",
    srcs = ["codegen.cpp"],
    hdrs = ["codegen.h"],
    deps = [
        "//common:check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:file_diagnostics",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
    ],
)
