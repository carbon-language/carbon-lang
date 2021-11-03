# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "diagnostic_emitter",
    srcs = ["diagnostic_emitter.cpp"],
    hdrs = ["diagnostic_emitter.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "null_diagnostics",
    hdrs = ["null_diagnostics.h"],
    deps = [
        ":diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mocks",
    testonly = 1,
    hdrs = ["mocks.h"],
    deps = [
        ":diagnostic_emitter",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "diagnostic_emitter_test",
    srcs = ["diagnostic_emitter_test.cpp"],
    deps = [
        ":diagnostic_emitter",
        ":mocks",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
