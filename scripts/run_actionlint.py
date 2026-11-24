#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///


"""Runs actionlint on GitHub workflow files, mainly for prek."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys

import scripts_utils


def main() -> None:
    actionlint = scripts_utils.get_release(scripts_utils.Release.ACTIONLINT)
    os.execv(actionlint, [actionlint] + sys.argv[1:])


if __name__ == "__main__":
    main()
