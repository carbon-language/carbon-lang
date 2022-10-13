"""Merges stdout and stderr into a single stream with labels."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys
import textwrap


def main() -> None:
    p = subprocess.run(
        sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if p.stdout:
        print(textwrap.indent(p.stdout, "STDOUT: "), end="")
    if p.stderr:
        print(textwrap.indent(p.stderr, "STDERR: "), end="")
    exit(p.returncode)


if __name__ == "__main__":
    main()
