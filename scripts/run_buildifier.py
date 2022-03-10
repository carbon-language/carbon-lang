#!/usr/bin/env python3

"""Runs buildifier on passed-in BUILD files, mainly for pre-commit."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys

import scripts_utils  # type: ignore


def main() -> None:
    files = sys.argv[1:]
    if not files:
        return
    buildifier = scripts_utils.get_release(scripts_utils.Release.BUILDIFIER)
    subprocess.check_call([buildifier] + files)


if __name__ == "__main__":
    main()
