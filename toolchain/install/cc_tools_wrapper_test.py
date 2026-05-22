"""Wrapper script to run cc_tools_test from toolchain/install.

This script imports `test_tools` from `bazel.cc_toolchains.cc_tools_test`
and executes it. This allows running the test in `toolchain/install` package
while keeping the main logic in `bazel/cc_toolchains`.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from bazel.cc_toolchains.cc_tools_test import test_tools

if __name__ == "__main__":
    test_tools()
