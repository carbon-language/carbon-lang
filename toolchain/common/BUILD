# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

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
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "index_base",
    hdrs = ["index_base.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "yaml_test_helpers",
    testonly = 1,
    srcs = ["yaml_test_helpers.cpp"],
    hdrs = ["yaml_test_helpers.h"],
    deps = [
        "//common:ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
