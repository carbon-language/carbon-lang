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
    name = "language_server",
    srcs = ["language_server.cpp"],
    hdrs = ["language_server.h"],
    deps = [
        ":context",
        ":incoming_messages",
        ":outgoing_messages",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//toolchain/diagnostics:diagnostic_emitter",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cpp"],
    hdrs = ["context.h"],
    deps = [
        "//common:check",
        "//common:map",
        "//common:raw_string_ostream",
        "//toolchain/base:shared_value_stores",
        "//toolchain/check",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:file_diagnostics",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "handle",
    srcs = glob(["handle_*"]),
    hdrs = ["handle.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/base:shared_value_stores",
        "//toolchain/lex",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/parse",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "incoming_messages",
    srcs = ["incoming_messages.cpp"],
    hdrs = ["incoming_messages.h"],
    deps = [
        ":context",
        ":handle",
        "//common:check",
        "//common:map",
        "//common:ostream",
        "//common:raw_string_ostream",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "outgoing_messages",
    hdrs = ["outgoing_messages.h"],
    deps = [
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)
