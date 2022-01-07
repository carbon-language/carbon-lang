"""Runs `lit` for testing."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import subprocess

_PASSTHROUGH_FLAGS = ["filter", "filter-out"]


def _parse_args():
    """Parses command line arguments, returning the result."""
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "test_dir", help="The directory containing tests to run."
    )
    arg_parser.add_argument(
        "lit_args", nargs="*", help="Arguments to pass through to lit."
    )
    return arg_parser.parse_args()


def _normalize(relative_base, target):
    """Given a target, normalizes it to a relative path."""
    assert target
    if target.startswith(":"):
        # Local target; :foo -> my/dir/foo
        return os.path.join(relative_base, target[1:])
    elif target[0].isalpha():
        # Local target; foo -> my/dir/foo
        return os.path.join(relative_base, target)

    if ":" in target:
        # Specified target; //foo:bar -> //foo/bar
        target = target.replace(":", "/")
    else:
        # Default target; //foo -> //foo/foo
        target = os.path.join(target, os.path.basename(target))

    if target.startswith("@"):
        return os.path.join("external/", target[1:])
    elif target.startswith("//"):
        return target[2:]
    else:
        raise ValueError("Unhandled target path: %s" % target)


def main():
    parsed_args = _parse_args()

    bin_dir = os.getcwd()

    # Figure out the actual path for the test_dir.
    relative_base = os.path.dirname(_normalize("", os.environ["TEST_TARGET"]))
    test_dir = os.path.join(
        bin_dir, _normalize(relative_base, parsed_args.test_dir)
    )

    args = [
        os.path.join(os.environ["TEST_SRCDIR"], "llvm-project/llvm/lit"),
        "--path=%s" % bin_dir,
        test_dir,
        "-sv",
    ]

    # Run lit.
    p = subprocess.run(args=args + parsed_args.lit_args)
    # Do this instead of check_call to hide stack traces.
    if p.returncode != 0:
        exit("lit failed, exit code %d" % p.returncode)


if __name__ == "__main__":
    exit(main())
