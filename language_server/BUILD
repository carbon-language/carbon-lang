# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = [
    "//language_server:__subpackages__",
])

cc_binary(
    name = "language_server",
    srcs = [
        "language_server.cpp",
        "language_server.h",
        "main.cpp",
    ],
    deps = [
        "//common:error",
        "//third_party/clangd",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)
