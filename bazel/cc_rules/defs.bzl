# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wraps standard cc rules with the `cc_env` addition.

These should generally be used in place of `@rules_cc`.
"""

load(
    "@rules_cc//cc:defs.bzl",
    actual_cc_binary = "cc_binary",
    actual_cc_library = "cc_library",
    actual_cc_test = "cc_test",
)
load("//bazel/cc_toolchains:defs.bzl", "cc_env")

# Expose cc_library directly, for consistency.
cc_library = actual_cc_library

def cc_binary(env = {}, **kwargs):
    """Wraps `cc_binary`, adding `cc_env`."""
    actual_cc_binary(
        env = cc_env() | env,
        **kwargs
    )

def cc_test(env = {}, **kwargs):
    """Wraps `cc_binary`, adding `cc_env`."""
    actual_cc_test(
        env = cc_env() | env,
        **kwargs
    )
