# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Terminal rendering: what the attached terminal can do, and how to draw styled
# text for it. Used for diagnostic rendering and for command line output.

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "output_buffer_ref",
    hdrs = ["output_buffer_ref.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_test(
    name = "output_buffer_ref_test",
    size = "small",
    srcs = ["output_buffer_ref_test.cpp"],
    deps = [
        ":output_buffer_ref",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "color",
    srcs = ["color.cpp"],
    hdrs = ["color.h"],
    deps = [
        ":output_buffer_ref",
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "color_test",
    size = "small",
    srcs = ["color_test.cpp"],
    deps = [
        ":color",
        "//common:ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "style",
    srcs = ["style.cpp"],
    hdrs = ["style.h"],
    deps = [
        ":color",
        ":output_buffer_ref",
        "//common:check",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "style_test",
    size = "small",
    srcs = ["style_test.cpp"],
    deps = [
        ":style",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "capabilities",
    srcs = ["capabilities.cpp"],
    hdrs = ["capabilities.h"],
    deps = [
        ":color",
        "//common:filesystem",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "capabilities_test",
    size = "small",
    srcs = ["capabilities_test.cpp"],
    deps = [
        ":capabilities",
        "//common:filesystem",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cpp"],
    hdrs = ["metrics.h"],
    deps = [
        ":capabilities",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cpp"],
    deps = [
        ":metrics",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cpp"],
    hdrs = ["buffer.h"],
    deps = [
        ":capabilities",
        ":color",
        ":metrics",
        ":output_buffer_ref",
        ":style",
        "//common:check",
        "//common:filesystem",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "buffer_test",
    size = "small",
    srcs = ["buffer_test.cpp"],
    deps = [
        ":buffer",
        "//common:filesystem",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "terminal_benchmark",
    testonly = 1,
    srcs = ["terminal_benchmark.cpp"],
    deps = [
        ":buffer",
        ":capabilities",
        ":color",
        ":style",
        "//testing/base:benchmark_main",
        "@abseil-cpp//absl/random",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "terminal_benchmark_test",
    size = "small",
    srcs = [":terminal_benchmark"],
    args = ["--benchmark_dry_run"],
)
