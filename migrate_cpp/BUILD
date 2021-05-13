# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "migrate_cpp",
    srcs = ["migrate_cpp.py"],
    data = [
        ":clang_tidy.yaml",
        "//migrate_cpp/cpp_refactoring",
        "@bootstrap_clang_toolchain//:bin/clang-tidy",
    ],
    python_version = "PY3",
)
