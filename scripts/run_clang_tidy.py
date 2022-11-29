#!/usr/bin/env python3

"""Runs clang-tidy over all Carbon files."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import re
import subprocess

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    # Copied from run-clang-tidy.py for forwarding.
    parser.add_argument("-fix", action="store_true", help="Apply fix-its")
    # Local flags.
    parser.add_argument("files", nargs="*", help="Files to fix")
    parsed_args = parser.parse_args()

    # If files are passed in, resolve them; otherwise, add a path filter.
    if parsed_args.files:
        files = [str(Path(f).resolve()) for f in parsed_args.files]
    else:
        files = ["^(?!.*/(bazel-|third_party)).*$"]

    # Set the repo root as the working directory.
    os.chdir(Path(__file__).resolve().parent.parent)
    # Ensure create_compdb has been run.
    subprocess.check_call(["./scripts/create_compdb.py"])

    # Use the run-clang-tidy version that should be with the rest of the clang
    # toolchain. This exposes us to version skew with user-installed clang
    # versions, but avoids version skew between the script and clang-tidy
    # itself.
    with Path(
        "./bazel-execroot/external/bazel_cc_toolchain/"
        "clang_detected_variables.bzl"
    ).open() as f:
        clang_vars = f.read()
    clang_bindir_match = re.search(r"clang_bindir = \"(.*)\"", clang_vars)
    assert clang_bindir_match is not None, clang_vars

    args = [str(Path(clang_bindir_match[1]).joinpath("run-clang-tidy"))]

    # Forward flags.
    if parsed_args.fix:
        args.append("-fix")

    # Run clang-tidy from clang-tools-extra.
    exit(subprocess.call(args + files))


if __name__ == "__main__":
    main()
