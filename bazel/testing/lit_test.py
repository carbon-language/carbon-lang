"""Runs `lit` for testing."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
import sys


def main():
    env = os.environ.copy()
    # Adjust the PATH so that tools like `not` and `FileCheck` are present.
    env["PATH"] = "%s:%s" % (
        os.path.join(os.getcwd(), "external/llvm-project/llvm"),
        env["PATH"],
    )
    p = subprocess.run(
        args=["external/llvm-project/llvm/lit"] + sys.argv[1:], env=env
    )
    if p.returncode != 0:
        exit("lit failed, exit code %d" % p.returncode)


if __name__ == "__main__":
    exit(main())
