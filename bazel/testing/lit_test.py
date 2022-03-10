"""Runs `lit` for testing."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
from pathlib import Path
import subprocess


def _parse_args():
    """Parses command line arguments, returning the result."""
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "--package_name", help="The directory containing tests to run."
    )
    arg_parser.add_argument(
        "lit_args", nargs="*", help="Arguments to pass through to lit."
    )
    return arg_parser.parse_args()


def main():
    parsed_args = _parse_args()

    args = [
        str(Path(os.environ["TEST_SRCDIR"]).joinpath("llvm-project/llvm/lit")),
        str(Path.cwd().joinpath(parsed_args.package_name)),
        "-v",
    ]

    # Force tests to be explicit about command paths.
    env = os.environ.copy()
    del env["PATH"]

    # Run lit.
    try:
        subprocess.check_call(args=args + parsed_args.lit_args, env=env)
    except subprocess.CalledProcessError as e:
        # Print without the stack trace.
        exit(e)


if __name__ == "__main__":
    exit(main())
