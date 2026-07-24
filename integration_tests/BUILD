# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This package is for integration tests of the entire Carbon system,
# encompassing things like libraries as well as the toolchain proper. These
# will typically be execution tests, which use the toolchain to build and run a
# test binary, rather than directly checking the toolchain's output. This kind
# of testing should be used sparingly, because it's substantially more costly,
# and not applicable when the toolchain is built as a cross-compiler.

load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

# TODO: Change this to `carbon_test` once that exists.
carbon_binary(
    name = "make_unique_test",
    srcs = ["make_unique_test.carbon"],
)
