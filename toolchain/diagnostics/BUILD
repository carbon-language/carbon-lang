# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "emitter",
    srcs = [
        "consumer.cpp",
        "diagnostic.cpp",
    ],
    hdrs = [
        "consumer.h",
        "diagnostic.h",
        "emitter.h",
    ],
    deps = [
        ":format_providers",
        ":kind",
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "emitter_test",
    size = "small",
    srcs = ["emitter_test.cpp"],
    deps = [
        ":emitter",
        ":mocks",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "kind",
    srcs = ["kind.cpp"],
    hdrs = ["kind.h"],
    textual_hdrs = [
        "kind.def",
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
        ":kind",
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
        ":emitter",
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
        ":emitter",
        ":format_providers",
        ":mocks",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "null_diagnostics",
    testonly = 1,
    hdrs = ["null_diagnostics.h"],
    deps = [
        ":emitter",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "sorting_consumer",
    hdrs = ["sorting_consumer.h"],
    deps = [
        ":emitter",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "sorting_consumer_test",
    size = "small",
    srcs = ["sorting_consumer_test.cpp"],
    deps = [
        ":emitter",
        ":mocks",
        ":sorting_consumer",
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
        ":emitter",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
