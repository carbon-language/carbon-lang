# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Supports running a tool from the install filegroup."""

load("@rules_python//python:defs.bzl", "py_binary")

def run_tool(name, tool, data):
    # TODO: Fix the driver file discovery in order to allow symlinks.
    py_binary(
        name = name,
        main = "run_tool.py",
        srcs = ["run_tool.py"],
        args = ["$(location {})".format(tool)],
        data = [tool] + data,
    )
