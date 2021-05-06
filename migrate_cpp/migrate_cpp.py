"""Migrates C++ code to Carbon."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse


def _parse_args(args=None):
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(args=args)


def _main():
    # parsed_args = _parse_args()
    print("Running cpp-tidy...")
    print("Done!")


if __name__ == "__main__":
    _main()
