# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "create_compdb",
    srcs = ["create_compdb.py"],
)

mypy_test(
    name = "create_compdb_mypy_test",
    include_imports = True,
    deps = [":create_compdb"],
)
