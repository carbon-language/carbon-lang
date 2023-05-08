# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower_to_llvm",
    srcs = ["lower_to_llvm.cpp"],
    hdrs = ["lower_to_llvm.h"],
    deps = [
        ":lowering_context",
        "//toolchain/semantics:semantics_ir",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "lowering_context",
    srcs = [
        "lowering_context.cpp",
        "lowering_handle.cpp",
    ],
    hdrs = ["lowering_context.h"],
    deps = [
        "//common:check",
        "//toolchain/semantics:semantics_ir",
        "//toolchain/semantics:semantics_node",
        "//toolchain/semantics:semantics_node_kind",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
