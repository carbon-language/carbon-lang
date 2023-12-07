# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Trivial, single-file testing libraries. More complex libraries should get
# their own directory.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

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
        "//common:init_llvm",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_raw_ostream",
    testonly = 1,
    hdrs = ["test_raw_ostream.h"],
    deps = [
        "//common:ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "test_raw_ostream_test",
    size = "small",
    srcs = ["test_raw_ostream_test.cpp"],
    deps = [
        ":test_raw_ostream",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)
