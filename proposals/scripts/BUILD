# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_binary(
    name = "new_proposal",
    srcs = ["new_proposal.py"],
    python_version = "PY3",
    deps = [":utils"],
)

py_test(
    name = "new_proposal_test",
    size = "small",
    srcs = ["new_proposal_test.py"],
    data = ["template.md"],
    python_version = "PY3",
    deps = [":new_proposal"],
)

py_binary(
    name = "check_proposal_names",
    srcs = ["check_proposal_names.py"],
    python_version = "PY3",
    deps = [":utils"],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [":utils"],
)

py_test(
    name = "check_proposal_names_test",
    size = "small",
    srcs = ["check_proposal_names_test.py"],
    python_version = "PY3",
    deps = [":check_proposal_names"],
)
