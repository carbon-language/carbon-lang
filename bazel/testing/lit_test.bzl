# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for a lit test."""

def lit_test(name, test_dir, tools = None, **kwargs):
    """Runs `lit` on test_dir.

    `lit` reference:
      https://llvm.org/docs/CommandGuide/lit.html

    To pass flags to `lit`, use `--test_arg`. For example:
      bazel test :lit_test --test_arg=-v
      bazel test :lit_test --test_arg=--filter=REGEXP

    Args:
      name: Name of the build rule.
      test_dir: The directory with the lit tests.
      tools: An optional list of tools to provide to the tests. These will be
        aliased for execution.
      **kwargs: Any additional parameters for the generated py_test.
    """
    if not tools:
        tools = []
    tools += [
        "@llvm-project//llvm:lit",
        "@llvm-project//llvm:not",
        "@llvm-project//llvm:FileCheck",
    ]
    native.py_test(
        name = name,
        srcs = ["//bazel/testing:lit_test.py"],
        main = "//bazel/testing:lit_test.py",
        data = tools + native.glob([test_dir + "/**"]),
        args = ["--tool=%s" % t for t in tools] + [test_dir, "--"],
        **kwargs
    )
