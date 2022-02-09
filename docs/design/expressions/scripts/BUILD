# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")
load("@py_deps//:requirements.bzl", "requirement")

py_binary(
    name = "precedence",
    srcs = ["precedence.py"],
    deps = [requirement("graphviz")],
)

py_test(
    name = "precedence_test",
    srcs = ["precedence_test.py"],
    data = ["precedence.dot"],
    deps = [":precedence"],
)

mypy_test(
    name = "precedence_mypy_test",
    include_imports = True,
    deps = [":precedence"],
)
