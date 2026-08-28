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
    name = "language_server",
    srcs = ["language_server.cpp"],
    hdrs = ["language_server.h"],
    deps = [
        ":context",
        ":incoming_messages",
        ":outgoing_messages",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//toolchain/base:install_paths",
        "//toolchain/diagnostics:emitter",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "sem_ir_index",
    srcs = ["sem_ir_index.cpp"],
    hdrs = ["sem_ir_index.h"],
    deps = [
        "//common:check",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:typed_insts",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cpp"],
    hdrs = ["context.h"],
    deps = [
        ":sem_ir_index",
        "//common:check",
        "//common:map",
        "//common:raw_string_ostream",
        "//toolchain/base:clang_invocation",
        "//toolchain/base:install_paths",
        "//toolchain/base:shared_value_stores",
        "//toolchain/check",
        "//toolchain/diagnostics:emitter",
        "//toolchain/diagnostics:file_diagnostics",
        "//toolchain/diagnostics:renderer",
        "//toolchain/driver:codegen_options",
        "//toolchain/driver:compile_driver",
        "//toolchain/driver:compile_options",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "position",
    srcs = ["position.cpp"],
    hdrs = ["position.h"],
    deps = [
        ":context",
        ":sem_ir_index",
        "//toolchain/lex:token_index",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:typed_insts",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "handle",
    srcs = glob(["handle_*"]),
    hdrs = ["handle.h"],
    deps = [
        ":context",
        ":position",
        ":sem_ir_index",
        "//common:check",
        "//common:raw_string_ostream",
        "//toolchain/base:shared_value_stores",
        "//toolchain/lex",
        "//toolchain/lex:token_index",
        "//toolchain/lex:token_kind",
        "//toolchain/parse",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:stringify",
        "//toolchain/sem_ir:typed_insts",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
        "@llvm-project//llvm:Support",
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
