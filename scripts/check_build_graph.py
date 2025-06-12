#!/usr/bin/env python3

"""Verify that the bazel build graph is in a valid state, for pre-commit."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess

import scripts_utils


def main() -> None:
    scripts_utils.chdir_repo_root()
    bazel = scripts_utils.locate_bazel()
    subprocess.check_call([bazel, "build", "--nobuild", "//..."])


if __name__ == "__main__":
    main()
