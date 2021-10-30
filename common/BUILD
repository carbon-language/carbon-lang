# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "indirect_value",
    srcs = ["indirect_value.h"],
)

cc_test(
    name = "indirect_value_test",
    srcs = ["indirect_value_test.cpp"],
    deps = [
        ":indirect_value",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ostream",
    hdrs = ["ostream.h"],
    srcs = ["ostream.cpp"],
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
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "string_helpers_test",
    srcs = ["string_helpers_test.cpp"],
    deps = [
        ":string_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)
