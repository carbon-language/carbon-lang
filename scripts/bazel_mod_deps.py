#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///


"""Run `bazel mod deps` for pre-commit. Lets pre-commit handle modifications."""

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
    subprocess.run([bazel, "mod", "--curses=no", "deps"])


if __name__ == "__main__":
    main()
