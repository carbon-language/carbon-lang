# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bazel_working_dir",
    hdrs = ["bazel_working_dir.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "check",
    srcs = ["check_internal.h"],
    hdrs = ["check.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cpp"],
    deps = [
        ":check",
        "//common:gtest_main",
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
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

# This does extra initialization on top of googletest's gtest_main in order to
# provide stack traces on unexpected exits, because we normally rely on LLVM
# code for that.
#
# This replaces "@com_google_googletest//:gtest_main";
# "@com_google_googletest//:gtest" should still be used directly.
cc_library(
    name = "gtest_main",
    testonly = 1,
    srcs = ["gtest_main.cpp"],
    deps = [
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
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
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "metaprogramming",
    hdrs = ["metaprogramming.h"],
)

cc_library(
    name = "ostream",
    hdrs = ["ostream.h"],
    deps = [
        ":metaprogramming",
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
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
