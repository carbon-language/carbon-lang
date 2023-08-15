# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//common/testing/file_test:rules.bzl", "file_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "codegen",
    srcs = ["codegen.cpp"],
    hdrs = ["codegen.h"],
    deps = [
        "@llvm-project//llvm:AllTargetsAsmParsers",
        "@llvm-project//llvm:AllTargetsCodeGens",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
    ],
)

file_test(
    name = "codegen_file_test",
    srcs = ["codegen_file_test.cpp"],
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        "//toolchain/driver:driver_file_test_base",
        "@llvm-project//llvm:Support",
    ],
)
