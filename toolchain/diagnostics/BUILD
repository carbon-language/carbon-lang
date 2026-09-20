# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/cc_rules:defs.bzl", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "emitter",
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

# Drawing diagnostics, kept apart from `:emitter` so that naming a `Consumer`
# doesn't pull the terminal library -- and through it `<filesystem>` -- into
# every layer that emits a diagnostic.
cc_library(
    name = "renderer",
    srcs = [
        "renderer.cpp",
        "stream_consumer.cpp",
    ],
    hdrs = [
        "renderer.h",
        "stream_consumer.h",
    ],
    deps = [
        ":emitter",
        ":format_providers",
        ":kind",
        "//common:check",
        "//common:ostream",
        "//common/terminal:buffer",
        "//common/terminal:capabilities",
        "//common/terminal:color",
        "//common/terminal:metrics",
        "//common/terminal:output_buffer_ref",
        "//common/terminal:style",
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

cc_test(
    name = "renderer_test",
    size = "small",
    srcs = ["renderer_test.cpp"],
    deps = [
        ":emitter",
        ":renderer",
        "//common:raw_string_ostream",
        "//common/terminal:capabilities",
        "//common/terminal:metrics",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "renderer_fuzzer",
    size = "small",
    srcs = ["renderer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":emitter",
        ":renderer",
        "//common:check",
        "//common/terminal:capabilities",
        "//common/terminal:output_buffer_ref",
        "//testing/fuzzing:libfuzzer_header",
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

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**/*.carbon"]),
)
