# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

py_binary(
    name = "migrate_cpp",
    srcs = ["migrate_cpp.py"],
    data = [
        ":clang_tidy.yaml",
        "//migrate_cpp/cpp_refactoring",
    ],
    python_version = "PY3",
)

cc_library(
    name = "output_segment",
    hdrs = ["output_segment.h"],
    deps = [
        "//common:check",
        "@llvm-project//clang:ast",
    ],
)

cc_library(
    name = "rewriter",
    srcs = ["rewriter.cpp"],
    hdrs = ["rewriter.h"],
    visibility = [
        # For dependency checking. Remove once it has a binary using it.
        "//bazel/check_deps:__pkg__",
    ],
    deps = [
        ":output_segment",
        "//common:check",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:tooling",
        "@llvm-project//clang:tooling_core",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "rewriter_test",
    srcs = ["rewriter_test.cpp"],
    deps = [
        ":rewriter",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:tooling",
    ],
)
