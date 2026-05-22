#!/usr/bin/env python3

"""Checks various LLVM tool symlinks behave as expected."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
import time
import unittest
from bazel_tools.tools.python.runfiles import runfiles
from bazel_integration_test.py import test_base


class BazelExampleTest(test_base.TestBase):
    def setUp(self) -> None:
        test_base.TestBase.setUp(self)
        self.runfiles = runfiles.Create()
        self.install_module = self.runfiles.Rlocation(
            "carbon/toolchain/install"
        )
        self.startup_flags = [
            "--ignore_all_rc_files",
            "--batch",
        ]
        self.flags = [
            f"--override_module=carbon_toolchain={self.install_module}"
        ]

    def _run_bazel(self, command: list[str]) -> str:
        """Runs bazel with retry logic for transient errors."""
        for attempt in range(5):
            exit_code, stdout, stderr = self.RunBazel(
                self.startup_flags + command + self.flags
            )

            # Several error codes are reliably permanent, break immediately.
            # `1`  -- The build failed.
            # `2`  -- Command line or environment problem.
            # `3`  -- Tests failed or timed out, we don't retry at this layer
            #         on execution timeout.
            # `4`  -- Test command but no tests found.
            # `8`  -- Explicitly interrupted build.
            #
            # Note that `36` is documented as "likely permanent", but we retry
            # it as most of our transient failures actually produce that error
            # code.
            perm_error = (1, 2, 3, 4, 8)
            if exit_code in perm_error:
                break

            for line in stderr:
                print(line, file=sys.stderr)

            if exit_code == 0:
                return stdout

            # Retry transient errors with a brief delay.
            print(f"Attempt {attempt + 1} failed with exit code {exit_code}")
            time.sleep(attempt)

        self.AssertExitCode(exit_code, 0, stderr)
        return stdout

    def test_compile_lib(self) -> None:
        # TODO: Can remove this in favor of always running `test_run` if we can
        # make linking a binary sufficiently efficient.
        self._run_bazel(["build", "//:example_lib"])

    @unittest.skipUnless(
        "CARBON_BAZEL_TEST_FULL" in os.environ,
        "Skipping expensive test step for minimal testing",
    )
    def test_run(self) -> None:
        stdout = self._run_bazel(["run", "//:example"])
        self.assertEqual(stdout, ["Hello World!"])


if __name__ == "__main__":
    unittest.main()
