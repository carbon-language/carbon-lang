# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

py_binary(
    name = "migrate_cpp",
    srcs = ["migrate_cpp.py"],
    data = [
        ":clang_tidy.yaml",
        "//migrate_cpp/cpp_refactoring",
    ],
    python_version = "PY3",
)

mypy_test(
    name = "migrate_cpp_mypy_test",
    include_imports = True,
    deps = [":migrate_cpp"],
)
