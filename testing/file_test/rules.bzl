# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("@bazel_skylib//rules:native_binary.bzl", "native_test")

def add_file_tests(name, srcs, deps, tests, shard_count = 1):
    """Generates tests using the file_test base.

    There will be one main test using `name` that can be sharded, and includes
    all files. Additionally, per-file tests will be generated as
    `name.file_path`; these per-file tests will be manual.

    Args:
      name: The base name of the tests.
      srcs: cc_binary srcs.
      deps: cc_binary deps.
      tests: The list of test files to use as data.
      shard_count: The number of shards to use; defaults to 1.
    """
    bin = "{0}.bin".format(name)

    # Produce a single binary for all test forms.
    native.cc_binary(
        name = bin,
        srcs = srcs,
        deps = deps,
    )

    # There's one main test for files.
    native_test(
        name = name,
        src = bin,
        out = "{0}.copy".format(bin),
        data = tests,
        args = ["$(location {0})".format(x) for x in tests],
        shard_count = shard_count,
        tags = ["manual"],
    )

    # But we also produce per-file tests that can be run directly.
    for test in tests:
        native_test(
            name = "{0}.{1}".format(name, test),
            src = bin,
            out = "{0}.copy.{1}".format(bin, test),
            data = [test],
            args = ["$(location {0})".format(test)],
            #tags = ["manual"],
        )
