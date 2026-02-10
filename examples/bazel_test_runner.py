#!/usr/bin/env python3

"""Checks various LLVM tool symlinks behave as expected."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
import unittest
from bazel_tools.tools.python.runfiles import runfiles
from bazel_integration_test.py import test_base


class BazelExampleTest(test_base.TestBase):
    def setUp(self) -> None:
        test_base.TestBase.setUp(self)
        self.runfiles = runfiles.Create()
        self.install_module = self.runfiles.Rlocation(
            "carbon/toolchain/install/prefix/lib/carbon"
        )
        self.startup_flags = [
            "--ignore_all_rc_files",
            "--batch",
        ]
        self.flags = [
            f"--override_module=carbon_toolchain={self.install_module}"
        ]

    def log_lines(self, lines: list[str]) -> None:
        for line in lines:
            print(line, file=sys.stderr)

    def test_compile_lib(self) -> None:
        # TODO: Can remove this if we can make linking a binary sufficiently
        # efficient.
        exit_code, stdout, stderr = self.RunBazel(
            self.startup_flags + ["build"] + self.flags + ["//:example_lib"]
        )
        self.log_lines(stderr)
        self.AssertExitCode(exit_code, 0, stderr)

    @unittest.skipUnless(
        "CARBON_BAZEL_TEST_FULL" in os.environ,
        "Skipping expensive test step for minimal testing",
    )
    def test_run(self) -> None:
        exit_code, stdout, stderr = self.RunBazel(
            self.startup_flags + ["run"] + self.flags + ["//:example"]
        )
        self.log_lines(stderr)
        self.AssertExitCode(exit_code, 0, stderr)
        self.assertEqual(stdout, ["Hello World!"])


if __name__ == "__main__":
    unittest.main()
