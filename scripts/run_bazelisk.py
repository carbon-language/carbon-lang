#!/usr/bin/env python3

"""Runs bazelisk with arbitrary arguments."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys

import scripts_utils


def main() -> None:
    bazelisk = scripts_utils.get_release(scripts_utils.Release.BAZELISK)
    os.execv(bazelisk, [bazelisk] + sys.argv[1:])


if __name__ == "__main__":
    main()
