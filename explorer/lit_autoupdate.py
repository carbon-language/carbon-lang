#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*")
    parser.add_argument(
        "--build_mode",
        metavar="MODE",
        default="opt",
        help="The build mode to use. Defaults to opt for faster execution.",
    )
    return parser.parse_args()


def main() -> None:
    # Calls the main script with explorer settings. This uses execv in order to
    # avoid Python import behaviors.
    parsed_args = parse_args()
    actual_py = Path(__file__).parent.parent.joinpath(
        "bazel", "testing", "lit_autoupdate_base.py"
    )
    args = [
        sys.argv[0],
        "--build_mode",
        parsed_args.build_mode,
        "--build_target",
        "//explorer",
        "--substitute",
        "%{explorer}",
        "./bazel-bin/explorer/explorer",
        "--testdata",
        "explorer/testdata",
        "--line_number_pattern",
        r"(?<=\.carbon:)(\d+)(?=(?:\D|$))",
    ] + parsed_args.tests
    os.execv(actual_py, args)


if __name__ == "__main__":
    main()
