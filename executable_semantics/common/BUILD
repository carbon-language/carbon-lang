# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "arena",
    srcs = ["arena.cpp"],
    hdrs = ["arena.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "error",
    hdrs = ["error.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "error_test",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

cc_library(
    name = "tracing_flag",
    srcs = ["tracing_flag.cpp"],
    hdrs = ["tracing_flag.h"],
    visibility = ["//executable_semantics:__subpackages__"],
)
