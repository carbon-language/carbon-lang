"""Merges stdout and stderr into a single stream with labels."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys


def _print(output: str, label: str) -> None:
    if output:
        for line in output.splitlines():
            if line:
                print(f"{label}: {line}")
            else:
                print(f"{label}:")


def main() -> None:
    p = subprocess.run(
        sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    # The `lambda line` forces prefixes on empty lines.
    _print(p.stdout, "STDOUT")
    _print(p.stderr, "STDERR")
    exit(p.returncode)


if __name__ == "__main__":
    main()
