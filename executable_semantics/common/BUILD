# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "any_printable_ptr",
    hdrs = ["any_printable_ptr.h"],
    deps = [
        ":nonnull",
        "//common:ostream",
    ],
)

cc_test(
    name = "any_printable_ptr_test",
    srcs = ["any_printable_ptr_test.cpp"],
    deps = [
        ":any_printable_ptr",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
    ],
)

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
    ],
)

cc_test(
    name = "error_test",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        "@llvm-project//llvm:gtest",
        "@llvm-project//llvm:gtest_main",
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
