# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower",
    srcs = ["lower.cpp"],
    hdrs = ["lower.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/semantics:semantics_ir",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
