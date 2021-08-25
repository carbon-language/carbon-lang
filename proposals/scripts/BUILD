# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "proposals",
    srcs = ["proposals.py"],
)

py_test(
    name = "proposals_test",
    srcs = ["proposals_test.py"],
    data = ["//proposals:md_files"],
    python_version = "PY3",
    deps = [":proposals"],
)

py_binary(
    name = "new_proposal",
    srcs = ["new_proposal.py"],
    python_version = "PY3",
)

py_test(
    name = "new_proposal_test",
    srcs = ["new_proposal_test.py"],
    data = ["//proposals:md_files"],
    python_version = "PY3",
    deps = [":new_proposal"],
)

# This is a directly runnable script, but should not be run via bazel.
py_library(
    name = "update_proposal_list",
    srcs = ["update_proposal_list.py"],
    deps = [":proposals"],
)
