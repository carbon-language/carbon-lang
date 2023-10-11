# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bazel_working_dir",
    hdrs = ["bazel_working_dir.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "command_line",
    srcs = ["command_line.cpp"],
    hdrs = ["command_line.h"],
    deps = [
        ":check",
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "command_line_test",
    srcs = ["command_line_test.cpp"],
    deps = [
        ":command_line",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "check",
    srcs = [
        "check_internal.cpp",
        "check_internal.h",
    ],
    hdrs = ["check.h"],
    deps = [
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cpp"],
    deps = [
        ":check",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "enum_base",
    hdrs = ["enum_base.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "enum_base_test_def",
    testonly = 1,
    textual_hdrs = ["enum_base_test.def"],
)

cc_test(
    name = "enum_base_test",
    srcs = ["enum_base_test.cpp"],
    deps = [
        ":enum_base",
        ":enum_base_test_def",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "error",
    hdrs = ["error.h"],
    deps = [
        ":check",
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "error_test",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "indirect_value",
    hdrs = ["indirect_value.h"],
)

cc_test(
    name = "indirect_value_test",
    srcs = ["indirect_value_test.cpp"],
    deps = [
        ":indirect_value",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ostream",
    hdrs = ["ostream.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "string_helpers",
    srcs = ["string_helpers.cpp"],
    hdrs = ["string_helpers.h"],
    deps = [
        ":check",
        ":error",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "string_helpers_test",
    srcs = ["string_helpers_test.cpp"],
    deps = [
        ":string_helpers",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "struct_reflection",
    hdrs = ["struct_reflection.h"],
)

cc_test(
    name = "struct_reflection_test",
    size = "small",
    srcs = ["struct_reflection_test.cpp"],
    deps = [
        ":struct_reflection",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "vlog",
    srcs = ["vlog_internal.h"],
    hdrs = ["vlog.h"],
    deps = [
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "vlog_test",
    srcs = ["vlog_test.cpp"],
    deps = [
        ":vlog",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)
