# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

py_binary(
    name = "new_proposal",
    srcs = ["new_proposal.py"],
    python_version = "PY3",
)

py_test(
    name = "new_proposal_test",
    srcs = ["new_proposal_test.py"],
    data = ["template.md"],
    python_version = "PY3",
    deps = [":new_proposal"],
)

mypy_test(
    name = "new_proposal_mypy_test",
    include_imports = True,
    deps = [":new_proposal"],
)
