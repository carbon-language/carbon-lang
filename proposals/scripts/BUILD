# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@mypy_integration//:mypy.bzl", "mypy_test")

py_library(
    name = "proposal_list",
    srcs = ["proposal_list.py"],
    visibility = ["//website/jekyll:__pkg__"],
)

py_test(
    name = "proposal_list_test",
    srcs = ["proposal_list_test.py"],
    data = ["//proposals:website_content"],
    python_version = "PY3",
    deps = [":proposal_list"],
)

py_binary(
    name = "new_proposal",
    srcs = ["new_proposal.py"],
    python_version = "PY3",
)

py_test(
    name = "new_proposal_test",
    srcs = ["new_proposal_test.py"],
    data = ["//proposals:website_content"],
    python_version = "PY3",
    deps = [":new_proposal"],
)

mypy_test(
    name = "new_proposal_mypy_test",
    deps = [":new_proposal"],
)

# This is a directly runnable script, but should not be run via bazel.
py_library(
    name = "update_proposal_list",
    srcs = ["update_proposal_list.py"],
    deps = [":proposal_list"],
)

mypy_test(
    name = "update_proposal_list_mypy_test",
    deps = [":update_proposal_list"],
)
