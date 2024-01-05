# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//installers:__subpackages__",
    "//language_server:__subpackages__",
])

cc_binary(
    name = "language_server",
    srcs = [
        "language_server.cpp",
        "language_server.h",
        "main.cpp",
    ],
    # Some parameters are unused in clangd headers.
    copts = ["-Wno-unused-parameter"],
    deps = [
        "//common:error",
        "//toolchain/base:value_store",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/lex",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/source:source_buffer",
        "@llvm-project//clang-tools-extra/clangd:clangd_library",
        "@llvm-project//llvm:Support",
    ],
)
