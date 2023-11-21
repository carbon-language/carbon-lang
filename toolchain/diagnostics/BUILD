# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "diagnostic_emitter",
    hdrs = ["diagnostic_emitter.h"],
    deps = [
        ":diagnostic_kind",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "diagnostic_emitter_test",
    size = "small",
    srcs = ["diagnostic_emitter_test.cpp"],
    deps = [
        ":diagnostic_emitter",
        ":mocks",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "diagnostic_kind",
    srcs = ["diagnostic_kind.cpp"],
    hdrs = ["diagnostic_kind.h"],
    textual_hdrs = [
        "diagnostic_kind.def",
    ],
    deps = [
        "//common:enum_base",
    ],
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
    name = "sorting_diagnostic_consumer",
    hdrs = ["sorting_diagnostic_consumer.h"],
    deps = [
        ":diagnostic_emitter",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "sorting_diagnostic_consumer_test",
    size = "small",
    srcs = ["sorting_diagnostic_consumer_test.cpp"],
    deps = [
        ":diagnostic_emitter",
        ":mocks",
        ":sorting_diagnostic_consumer",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mocks",
    testonly = 1,
    srcs = ["mocks.cpp"],
    hdrs = ["mocks.h"],
    deps = [
        ":diagnostic_emitter",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
