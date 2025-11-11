# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

carbon_binary(
    name = "re2_playground",
    srcs = ["re2_playground.carbon"],
    flags = ["--clang-arg=-std=c++20"],
    tags = ["manual"],
    deps = [
        "@llvm-project//llvm:LineEditor",
        "@re2",
    ],
)
