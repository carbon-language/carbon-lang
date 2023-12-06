# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building tests from multiple files."""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

def gtest_library(deps = [], **kwargs):
    """Generates a library corresponding to a test suite.

    Args:
        **kwargs: Passed down to the cc_library rule.
    """
    cc_library(
        testonly = 1,
        alwayslink = 1,
        deps = ["@com_google_googletest//:gtest"] + deps,
        **kwargs
    )

def gtest_test(name, tests, deps = [], args = [], **kwargs):
    """Generates a collection of test targets, one for each test in `tests`.

    Also generates a test binary that can be used by other test targets.

    Args:
        name: The name of the test binary.
        tests: A list of (label, suite) pairs for each gtest_library and
               its test suite name. Dependencies on these libraries are added
               to any specified `deps`.
        **kwargs: All other arguments are passed down to the cc_binary rule.
    """
    cc_binary(
        name = name,
        testonly = 1,
        deps = deps + [label for (label, suite) in tests],
        **kwargs
    )
    [
        native.sh_test(
            name = native.package_relative_label(label).name + ".test",
            size = "small",
            srcs = [name],
            args = args + ["--gtest_filter=" + suite + ".*"],
        )
        for (label, suite) in tests
    ]

    # Also add an sh_test for tests associated with none of the listed suites.
    native.sh_test(
        name = name + ".uncategorized",
        size = "small",
        srcs = [name],
        args = args + [
            "--gtest_filter=-" + ":".join([suite for (label, suite) in tests]),
        ],
    )
