# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gen_llvm",
    srcs = ["gen_llvm.cpp"],
    hdrs = ["gen_llvm.h"],
    deps = [
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)
