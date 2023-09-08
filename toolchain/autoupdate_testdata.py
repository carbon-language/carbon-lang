#!/usr/bin/env python3

"""Autoupdates testdata in toolchain."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess


def main() -> None:
    subprocess.run(
        [
            "bazel",
            "run",
            "-c",
            "opt",
            "//toolchain/testing:file_test",
            "--",
            "--autoupdate",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
