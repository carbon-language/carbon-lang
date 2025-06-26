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
    name = "lower",
    srcs = ["lower.cpp"],
    hdrs = ["lower.h"],
    deps = [
        ":context",
        "//toolchain/parse:tree",
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
        "context.cpp",
        "file_context.cpp",
        "function_context.cpp",
        "mangler.cpp",
        "mangler.h",
        "specific_coalescer.cpp",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "handle*.cpp",
    ]),
    hdrs = [
        "context.h",
        "file_context.h",
        "function_context.h",
        "specific_coalescer.h",
    ],
    deps = [
        "//common:check",
        "//common:growing_range",
        "//common:map",
        "//common:pretty_stack_trace_function",
        "//common:raw_string_ostream",
        "//common:vlog",
        "//toolchain/base:fixed_size_value_store",
        "//toolchain/base:kind_switch",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:absolute_node_id",
        "//toolchain/sem_ir:diagnostic_loc_converter",
        "//toolchain/sem_ir:entry_point",
        "//toolchain/sem_ir:expr_info",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:inst_namer",
        "//toolchain/sem_ir:stringify",
        "//toolchain/sem_ir:typed_insts",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:lex",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)
