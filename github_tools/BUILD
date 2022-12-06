# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@py_deps//:requirements.bzl", "requirement")

py_library(
    name = "github_helpers",
    srcs = ["github_helpers.py"],
    deps = [requirement("gql")],
)

py_test(
    name = "github_helpers_test",
    srcs = ["github_helpers_test.py"],
    python_version = "PY3",
    deps = [":github_helpers"],
)

py_binary(
    name = "pr_comments",
    srcs = ["pr_comments.py"],
    python_version = "PY3",
    deps = ["github_helpers"],
)

py_test(
    name = "pr_comments_test",
    srcs = ["pr_comments_test.py"],
    python_version = "PY3",
    deps = [":pr_comments"],
)
