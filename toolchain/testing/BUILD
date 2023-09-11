# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

file_test(
    name = "file_test",
    srcs = ["file_test.cpp"],
    tests = [
        "//toolchain/check:testdata",
        "//toolchain/codegen:testdata",
        "//toolchain/driver:testdata",
        "//toolchain/lex:testdata",
        "//toolchain/lower:testdata",
        "//toolchain/parse:testdata",
    ],
    deps = [
        "//testing/file_test:file_test_base",
        "//toolchain/driver",
        "@llvm-project//llvm:Support",
    ],
)
