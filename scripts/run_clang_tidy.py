#!/usr/bin/env python3

"""Runs clang-tidy over all Carbon files.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import subprocess
import sys

from pathlib import Path


def main() -> None:
    # Set the repo root as the working directory.
    os.chdir(Path(__file__).parent.parent)
    # Ensure create_compdb has been run.
    subprocess.check_call(["./scripts/create_compdb.py"])

    # Avoid adding a path filter if files are passed in.
    args = sys.argv[1:]
    if not args or "-fix" in args:
        args.append("^(?!.*/(bazel-|third_party)).*$")

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
    run_clang_tidy = str(Path(clang_bindir_match[1]).joinpath("run-clang-tidy"))

    # Run clang-tidy from clang-tools-extra.
    exit(subprocess.call([run_clang_tidy] + args))


if __name__ == "__main__":
    main()
