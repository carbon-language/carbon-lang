# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "file_test_base",
    srcs = ["file_test_base.cpp"],
    hdrs = ["file_test_base.h"],
    deps = [
        "//common:check",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "file_test",
    srcs = ["file_test_base_test.cpp"],
    args = ["$(location :example.carbon)"],
    data = ["example.carbon"],
    deps = [
        ":file_test_base",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)
