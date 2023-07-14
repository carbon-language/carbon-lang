# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("@bazel_skylib//rules:native_binary.bzl", "native_test")

def file_test(name, srcs, deps, tests, shard_count = 1):
    """Generates tests using the file_test base.

    There will be one main test using `name` that can be sharded, and includes
    all files. Additionally, per-file tests will be generated as
    `name.file_path`; these per-file tests will be manual.

    Args:
      name: The base name of the tests.
      srcs: cc_test srcs.
      deps: cc_test deps.
      tests: The list of test files to use as data.
      shard_count: The number of shards to use; defaults to 1.
    """
    subset_name = "{0}.subset".format(name)

    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        data = tests,
        args = ["$(location {0})".format(x) for x in tests],
        shard_count = shard_count,
    )

    native_test(
        name = subset_name,
        src = name,
        out = subset_name,
        data = tests,
        tags = ["manual"],
    )
