# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "autoupdate",
    testonly = 1,
    srcs = ["autoupdate.cpp"],
    hdrs = [
        "autoupdate.h",
        "line.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//common:raw_string_ostream",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@re2",
    ],
)

cc_library(
    name = "file_test_base",
    testonly = 1,
    srcs = [
        "file_test_base.cpp",
        "run_test.cpp",
        "run_test.h",
        "test_file.cpp",
        "test_file.h",
    ],
    hdrs = ["file_test_base.h"],
    deps = [
        ":autoupdate",
        ":manifest",
        "//common:check",
        "//common:error",
        "//common:exe_path",
        "//common:init_llvm",
        "//common:ostream",
        "//common:raw_string_ostream",
        "//testing/base:file_helpers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "file_test_base_test",
    size = "small",
    srcs = ["file_test_base_test.cpp"],
    tests = glob(["testdata/**"]),
    deps = [
        ":file_test_base",
        "//common:ostream",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

# Note this is separate from the implementation; see the .h file.
cc_library(
    name = "manifest",
    testonly = 1,
    hdrs = ["manifest.h"],
    deps = ["@llvm-project//llvm:Support"],
)

# Used through `file_test` in rules.bzl.
cc_library(
    name = "manifest_impl",
    testonly = 1,
    srcs = ["manifest.cpp"],
    deps = [":manifest"],
)
