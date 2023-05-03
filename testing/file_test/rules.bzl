# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("@bazel_skylib//rules:native_binary.bzl", "native_test")

def add_file_tests(name, srcs, tests, deps, shard_count = 1):
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
    )

    # But we also produce per-file tests that can be run directly.
    for test in tests:
        native_test(
            name = "{0}.{1}".format(name, test),
            src = bin,
            out = "{0}.copy.{1}".format(bin, test),
            data = [test],
            args = ["$(location {0})".format(test)],
            tags = ["manual"],
        )
