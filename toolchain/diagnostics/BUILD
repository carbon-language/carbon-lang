# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

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
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "diagnostic_kind",
    srcs = ["diagnostic_kind.cpp"],
    hdrs = ["diagnostic_kind.h"],
    textual_hdrs = [
        "diagnostic_registry.def",
    ],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "diagnostic_kind_test",
    size = "small",
    srcs = ["diagnostic_kind_test.cpp"],
    deps = [
        ":diagnostic_kind",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
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
        "//common:gtest_main",
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
