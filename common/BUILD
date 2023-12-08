# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

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
    size = "small",
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
    size = "small",
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
    size = "small",
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
    size = "small",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "hashing",
    srcs = ["hashing.cpp"],
    hdrs = ["hashing.h"],
    deps = [
        ":check",
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "hashing_test",
    srcs = ["hashing_test.cpp"],
    deps = [
        ":hashing",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "hashing_benchmark",
    testonly = 1,
    srcs = ["hashing_benchmark.cpp"],
    deps = [
        ":check",
        ":hashing",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/random",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "indirect_value",
    hdrs = ["indirect_value.h"],
)

cc_test(
    name = "indirect_value_test",
    size = "small",
    srcs = ["indirect_value_test.cpp"],
    deps = [
        ":indirect_value",
        "//testing/base:gtest_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "init_llvm",
    srcs = ["init_llvm.cpp"],
    hdrs = ["init_llvm.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

# Link against this to cause `:init_llvm` to pull in all LLVM targets.
#
# Be careful when depending on this: it pulls in several hundred megabytes of
# LLVM binary size in -c fastbuild. This should only be depended on by a
# `cc_binary` or `cc_test` target, never a `cc_library`.
cc_library(
    name = "all_llvm_targets",
    srcs = ["all_llvm_targets.cpp"],
    deps = [
        ":init_llvm",
        "@llvm-project//llvm:AllTargetsAsmParsers",
        "@llvm-project//llvm:AllTargetsCodeGens",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
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
    size = "small",
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
    size = "small",
    srcs = ["vlog_test.cpp"],
    deps = [
        ":vlog",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "@com_google_googletest//:gtest",
    ],
)
