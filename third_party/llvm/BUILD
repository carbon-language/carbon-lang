# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library")

cc_library(
    name = "clang_cc1",
    srcs = ["clang_cc1.cpp"],
    hdrs = ["clang_cc1.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:check",
        "//toolchain/base:install_paths",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "multiplex_external_sema_source",
    srcs = ["multiplex_external_sema_source.cpp"],
    hdrs = ["multiplex_external_sema_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//clang:sema",
        "@llvm-project//llvm:Support",
    ],
)
