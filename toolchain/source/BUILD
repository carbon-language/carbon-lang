# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "source_buffer",
    srcs = ["source_buffer.cpp"],
    hdrs = ["source_buffer.h"],
    deps = [
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "source_buffer_test",
    srcs = ["source_buffer_test.cpp"],
    deps = [
        ":source_buffer",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
