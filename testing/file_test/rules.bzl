# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_test")

"""Rules for building fuzz tests."""

def file_test(name, tests, data = [], args = [], **kwargs):
    """Generates tests using the file_test base.

    There will be one main test using `name` that can be sharded, and includes
    all files. Additionally, per-file tests will be generated as
    `name.file_path`; these per-file tests will be manual.

    Args:
      name: The base name of the tests.
      tests: The list of test files to use as data, typically a glob.
      data: Passed to cc_test.
      args: Passed to cc_test.
      **kwargs: Passed to cc_test.
    """
    cc_test(
        name = name,
        data = tests + data,
        args = ["--file_tests=" + ",".join([
            "$(location {0})".format(x)
            for x in tests
        ])] + args,
        **kwargs
    )
