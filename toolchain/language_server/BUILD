# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "language_server",
    srcs = ["language_server.cpp"],
    hdrs = ["language_server.h"],
    deps = [
        ":context",
        ":handler_registry",
        ":handlers",
        ":incoming_messages",
        ":outgoing_messages",
        "//common:error",
        "//common:ostream",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    deps = [
        "//common:map",
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
    ],
)

cc_library(
    name = "handler_registry",
    srcs = [
        # Registry instantiation needs to be one-per-file.
        "handler_registry_for_call.cpp",
        "handler_registry_for_notification.cpp",
    ],
    hdrs = ["handler_registry.h"],
    deps = [
        "@llvm-project//clang-tools-extra/clangd:ClangDaemon",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "handlers",
    srcs = glob(["handle_*"]),
    deps = [
        ":context",
        ":handler_registry",
        "//toolchain/base:shared_value_stores",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex",
        "//toolchain/parse",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "incoming_messages",
    srcs = ["incoming_messages.cpp"],
    hdrs = ["incoming_messages.h"],
    deps = [
        ":context",
        ":handler_registry",
        "//common:check",
        "//common:map",
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
