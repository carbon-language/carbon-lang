# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "clang_cc1",
    srcs = ["clang_cc1.cpp"],
    hdrs = ["clang_cc1.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:check",
        "//toolchain/install:install_paths",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:Support",
    ],
)
