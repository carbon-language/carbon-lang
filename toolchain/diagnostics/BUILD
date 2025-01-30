# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "diagnostic_emitter",
    srcs = [
        "diagnostic.cpp",
        "diagnostic_consumer.cpp",
    ],
    hdrs = [
        "diagnostic.h",
        "diagnostic_consumer.h",
        "diagnostic_emitter.h",
    ],
    deps = [
        ":diagnostic_kind",
        ":format_providers",
        "//common:check",
        "//common:ostream",
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
        "@googletest//:gtest",
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

manifest(
    name = "all_testdata.txt",
    srcs = ["//toolchain/testing:all_testdata"],
)

cc_test(
    name = "coverage_test",
    size = "small",
    srcs = ["coverage_test.cpp"],
    args = ["--testdata_manifest=$(location :all_testdata.txt)"],
    data = [
        ":all_testdata.txt",
        "//toolchain/testing:all_testdata",
    ],
    deps = [
        ":diagnostic_kind",
        "//testing/base:gtest_main",
        "//toolchain/testing:coverage_helper",
        "@abseil-cpp//absl/flags:flag",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "file_diagnostics",
    hdrs = ["file_diagnostics.h"],
    deps = [
        ":diagnostic_emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "format_providers",
    srcs = ["format_providers.cpp"],
    hdrs = ["format_providers.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "format_providers_test",
    size = "small",
    srcs = ["format_providers_test.cpp"],
    deps = [
        ":diagnostic_emitter",
        ":format_providers",
        ":mocks",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
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
        "//testing/base:gtest_main",
        "@googletest//:gtest",
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
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
