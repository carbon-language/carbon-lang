# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "source_buffer",
    srcs = ["source_buffer.cpp"],
    hdrs = ["source_buffer.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_test(
    name = "source_buffer_test",
    srcs = ["source_buffer_test.cpp"],
    deps = [
        ":source_buffer",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gmock",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)
