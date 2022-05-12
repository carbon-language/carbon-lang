#!/usr/bin/env python3

"""Runs buildifier on passed-in BUILD files, mainly for pre-commit."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys

import scripts_utils


def main() -> None:
    buildifier = scripts_utils.get_release(scripts_utils.Release.BUILDIFIER)
    os.execv(buildifier, [buildifier] + sys.argv[1:])


if __name__ == "__main__":
    main()
