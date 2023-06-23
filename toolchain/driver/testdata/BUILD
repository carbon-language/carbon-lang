# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//testing/lit_test:rules.bzl", "glob_lit_tests")

glob_lit_tests(
    data = [
        "//testing/lit_test:merge_output",
        "//toolchain/driver:carbon",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
    driver = "lit.cfg.py",
    test_file_exts = ["carbon"],
)
