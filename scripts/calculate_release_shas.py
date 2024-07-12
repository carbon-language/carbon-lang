#!/usr/bin/env python3

"""Prints sha information for tracked tool releases."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import scripts_utils


def main() -> None:
    scripts_utils.calculate_release_shas()


if __name__ == "__main__":
    main()
