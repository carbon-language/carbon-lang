# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower_to_llvm",
    srcs = ["lower_to_llvm.cpp"],
    hdrs = ["lower_to_llvm.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/semantics:semantics_ir",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
