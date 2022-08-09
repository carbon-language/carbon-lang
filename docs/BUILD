__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

load("@py_deps_doctests//:requirements.bzl", "requirement")
load("tests/md_tests.bzl", "generate_md_tests")

py_binary(
    name = "md_to_testdata",
    srcs = ["tests/md_to_testdata.py"],
    deps = [requirement("markdown")],
)

generate_md_tests(
    tool = "md_to_testdata",
    srcs = glob(["**/*.md"]),
)

# run with `bazel test docs/...`
