"""Merges stdout and stderr into a single stream with labels."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys


def label_output(label: str, output: str) -> None:
    """Prints output with labels.

    This mirrors label_output in scripts/autoupdate_testdata_base.py and should
    be kept in sync. They're separate in order to avoid a subprocess or import
    complexity.
    """
    if output:
        for line in output.splitlines():
            print(" ".join(filter(None, (label, line))))


def main() -> None:
    p = subprocess.run(
        sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    label_output("STDOUT:", p.stdout)
    label_output("STDERR:", p.stderr)
    exit(p.returncode)


if __name__ == "__main__":
    main()
