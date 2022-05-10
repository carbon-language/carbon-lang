# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//explorer:__subpackages__"])

cc_library(
    name = "arena",
    hdrs = ["arena.h"],
    deps = [
        ":nonnull",
    ],
)

cc_library(
    name = "error",
    hdrs = ["error.h"],
    deps = [
        "//common:check",
        "//common:error",
        "//explorer/ast:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "error_test",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "nonnull",
    hdrs = ["nonnull.h"],
    deps = [
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)
