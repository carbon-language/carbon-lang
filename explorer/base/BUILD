# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//explorer:__subpackages__"])

cc_library(
    name = "arena",
    hdrs = ["arena.h"],
    deps = [
        ":nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "arena_test",
    testonly = 1,
    alwayslink = 1,
    srcs = ["arena_test.cpp"],
    deps = [
        ":arena",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "decompose",
    hdrs = ["decompose.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "decompose_test",
    testonly = 1,
    alwayslink = 1,
    srcs = ["decompose_test.cpp"],
    deps = [
        ":decompose",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "error_builders",
    hdrs = ["error_builders.h"],
    deps = [
        ":source_location",
        "//common:error",
    ],
)

cc_library(
    name = "error_builders_test",
    testonly = 1,
    alwayslink = 1,
    srcs = ["error_builders_test.cpp"],
    deps = [
        ":error_builders",
        ":source_location",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "nonnull",
    hdrs = ["nonnull.h"],
    deps = [
        "//common:check",
    ],
)

cc_library(
    name = "print_as_id",
    hdrs = ["print_as_id.h"],
    deps = [
        "//common:ostream",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = [
        ":nonnull",
        "//common:ostream",
    ],
)

cc_library(
    name = "trace_stream",
    hdrs = ["trace_stream.h"],
    deps = [
        ":source_location",
        "//common:check",
        "//common:ostream",
        "//explorer/base:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "set_program_phase_raii_test",
    testonly = 1,
    alwayslink = 1,
    srcs = ["set_program_phase_raii_test.cpp"],
    deps = [
        ":trace_stream",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "set_file_context_raii_test",
    testonly = 1,
    alwayslink = 1,
    srcs = ["set_file_context_raii_test.cpp"],
    deps = [
        ":source_location",
        ":trace_stream",
        "@com_google_googletest//:gtest",
    ],
)
