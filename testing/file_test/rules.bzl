# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building file tests.

file_test uses the tests_as_input_file rule to transform test dependencies into
a file which can be accessed as a list. This avoids long argument parsing.
"""

load("@rules_cc//cc:defs.bzl", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")

def file_test(
        name,
        tests,
        data = [],
        args = [],
        prebuilt_binary = None,
        **kwargs):
    """Generates tests using the file_test base.

    There will be one main test using `name` that can be sharded, and includes
    all files. Additionally, per-file tests will be generated as
    `name.file_path`; these per-file tests will be manual.

    Args:
      name: The base name of the tests.
      tests: The list of test files to use as data, typically a glob.
      data: Passed to cc_test.
      args: Passed to cc_test.
      prebuilt_binary: If set, specifies a prebuilt test binary to use instead
                       of building a new one.
      **kwargs: Passed to cc_test.
    """

    # Ensure tests are always a filegroup for tests_as_input_file_rule.
    tests_file = "{0}.tests".format(name)
    manifest(
        name = tests_file,
        srcs = tests,
        testonly = 1,
    )
    args = ["--test_targets_file=$(rootpath :{0})".format(tests_file)] + args
    data = [":" + tests_file] + tests + data

    if prebuilt_binary:
        native.sh_test(
            name = name,
            srcs = [prebuilt_binary],
            data = data,
            args = args,
            **kwargs
        )
    else:
        cc_test(
            name = name,
            data = data,
            args = args,
            **kwargs
        )
