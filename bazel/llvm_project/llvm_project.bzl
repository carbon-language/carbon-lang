# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Apply llvm_configure to produce a llvm-project repo."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_project = module_extension(
    implementation = lambda ctx: llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "X86",
        ],
    ),
)
