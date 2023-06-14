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
    srcs = [
        "check_internal.cpp",
        "check_internal.h",
    ],
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
        "//testing/util:gtest_main",
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
    textual_hdrs = ["enum_base_test.def"],
)

cc_test(
    name = "enum_base_test",
    srcs = ["enum_base_test.cpp"],
    deps = [
        ":enum_base",
        ":enum_base_test_def",
        "//testing/util:gtest_main",
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
        "//testing/util:gtest_main",
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
        "//testing/util:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "metaprogramming",
    hdrs = ["metaprogramming.h"],
)

cc_test(
    name = "metaprogramming_test",
    srcs = ["metaprogramming_test.cpp"],
    deps = [
        ":metaprogramming",
        "//testing/util:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
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
        "//testing/util:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "vlog",
    srcs = ["vlog_internal.h"],
    hdrs = ["vlog.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "vlog_test",
    srcs = ["vlog_test.cpp"],
    deps = [
        ":vlog",
        "//testing/util:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
