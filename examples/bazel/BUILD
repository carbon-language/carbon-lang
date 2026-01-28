# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Note that this is not a BUILD file that is part of the larger Carbon project, and
# is not built by running `bazel build //examples/bazel/...`. This is its own
# _distinct_ example Bazel project rooted at `examples/bazel`. You will need to
# `cd` into this directory to interact with it.
#
# For more details about how to manually interact with this example, please see
# the adjacent `MODULES.bazel` file.

load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "example",
    srcs = [
        "example.cpp",
        "example.h",
    ],
)
