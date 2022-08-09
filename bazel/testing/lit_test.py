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
import tarfile
import tempfile


def _parse_args():
    """Parses command line arguments, returning the result."""
    arg_parser = argparse.ArgumentParser(description=__doc__)
    args_input = arg_parser.add_mutually_exclusive_group(required=True)
    args_input.add_argument(
        "--package_name", help="The directory containing tests to run."
    )
    args_input.add_argument(
        "--tarball_path", help="The tarball containing tests to run."
    )
    arg_parser.add_argument(
        "lit_args", nargs="*", help="Arguments to pass through to lit."
    )
    return arg_parser.parse_args()


def main():
    parsed_args = _parse_args()

    if parsed_args.package_name:
        args = [
            str(Path(os.environ["TEST_SRCDIR"]).joinpath("llvm-project/llvm/lit")),
            str(Path.cwd().joinpath(parsed_args.package_name)),
            "-v",
        ]
    else:
        tar = tarfile.open(parsed_args.tarball_path)
        for member in tar.getnames():
            if member.endswith(".carbon"): # TODO: do not hardcode extension here
                break
        else:
            exit(0) # there are no tests to run (which LIT would consider an error)
        tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd())
        tmppath = Path.cwd().joinpath(tmpdir.name)
        tar.extractall(tmppath)
        args = [
            str(Path(os.environ["TEST_SRCDIR"]).joinpath("llvm-project/llvm/lit")),
            str(tmppath),
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
