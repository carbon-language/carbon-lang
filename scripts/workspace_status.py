#!/usr/bin/env python3

"""Bazel `--workspace_status_command` script.

This script is designed to be used in Bazel`s `--workspace_status_command` and
generate any desirable status artifacts.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess


def git_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()


def git_dirty_suffix() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], encoding="utf-8"
    ).strip()
    return ".dirty" if len(status) > 0 else ""


def main() -> None:
    print("STABLE_GIT_COMMIT_SHA " + git_commit_sha())
    print("STABLE_GIT_DIRTY_SUFFIX " + git_dirty_suffix())


if __name__ == "__main__":
    main()
