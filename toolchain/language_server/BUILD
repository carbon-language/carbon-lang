# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "language_server",
    srcs = [
        "language_server.cpp",
        "server.cpp",
        "server.h",
    ],
    hdrs = ["language_server.h"],
    # Some parameters are unused in clangd headers.
    copts = ["-Wno-unused-parameter"],
    deps = [
        "//common:error",
        "//common:ostream",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
        "@llvm-project//llvm:Support",
    ],
)
