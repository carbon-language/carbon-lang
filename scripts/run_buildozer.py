#!/usr/bin/env python3

"""Runs buildozer on arguments.

This is provided for other scripts to run buildozer without requiring it be
manually installed.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys

import scripts_utils


def main() -> None:
    buildozer = scripts_utils.get_release(scripts_utils.Release.BUILDOZER)
    os.execv(buildozer, [buildozer] + sys.argv[1:])


if __name__ == "__main__":
    main()
