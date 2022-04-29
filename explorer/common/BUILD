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
    name = "error_builders",
    hdrs = ["error_builders.h"],
    deps = [
        ":source_location",
        "//common:error",
    ],
)

cc_test(
    name = "error_builders_test",
    srcs = ["error_builders_test.cpp"],
    deps = [
        ":error_builders",
        ":source_location",
        "//common:gtest_main",
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
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = [
        ":nonnull",
        "//common:ostream",
    ],
)
