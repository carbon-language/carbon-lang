"""Migrates C++ code to Carbon."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import glob
import os
import subprocess
import sys

_CLANG_TIDY = "../external/bootstrap_clang_toolchain/bin/clang-tidy"
_CPP_EXTS = {".h", ".c", ".cc", ".cpp", ".cxx"}


def _data_file(relative_path):
    """Returns the path to a data file."""
    return os.path.join(os.path.dirname(sys.argv[0]), relative_path)


def _parse_args(args=None):
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dir",
        type=str,
        help="A directory containing C++ files to migrate to Carbon.",
    )
    parsed_args = parser.parse_args(args=args)
    return parsed_args


def _gather_files(parsed_args):
    """Returns the list of C++ files to convert."""
    all_files = glob.glob(
        os.path.join(parsed_args.dir, "**/*.*"), recursive=True
    )
    cpp_files = [f for f in all_files if os.path.splitext(f)[1] in _CPP_EXTS]
    if not cpp_files:
        sys.exit(
            "%r doesn't contain any C++ files to convert." % parsed_args.dir
        )
    return sorted(cpp_files)


def _clang_tidy(parsed_args, cpp_files):
    """Runs clang-tidy to fix C++ files in a directory."""
    print("Running clang-tidy...")
    clang_tidy = _data_file(_CLANG_TIDY)
    with open(_data_file("clang_tidy.yaml")) as f:
        config = f.read()
    subprocess.run([clang_tidy, "--fix", "--config", config] + cpp_files)


def _main():
    """Main program execution."""
    parsed_args = _parse_args()

    # Validate arguments.
    if not os.path.isdir(parsed_args.dir):
        sys.exit("%r must point to a directory." % parsed_args.dir)

    cpp_files = _gather_files(parsed_args)
    _clang_tidy(parsed_args, cpp_files)
    print("Done!")


if __name__ == "__main__":
    _main()
