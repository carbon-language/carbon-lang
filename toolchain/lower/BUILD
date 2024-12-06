# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "lower",
    srcs = ["lower.cpp"],
    hdrs = ["lower.h"],
    deps = [
        ":context",
        "//toolchain/check:sem_ir_diagnostic_converter",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:inst_namer",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "context",
    srcs = [
        "constant.cpp",
        "constant.h",
        "file_context.cpp",
        "function_context.cpp",
        "mangler.cpp",
        "mangler.h",
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
        "//common:map",
        "//common:vlog",
        "//toolchain/base:kind_switch",
        "//toolchain/check:sem_ir_diagnostic_converter",
        "//toolchain/sem_ir:entry_point",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_namer",
        "//toolchain/sem_ir:typed_insts",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)
