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

import os
import subprocess
import sys


def use_jj() -> bool:
    if os.path.isdir(".jj"):
        return True
    elif os.path.exists(".git"):
        return False
    print("Can't tell whether to use jj or git:", os.getcwd())
    sys.exit(1)


def git_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()


def git_dirty_suffix() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], encoding="utf-8"
    ).strip()
    return ".dirty" if len(status) > 0 else ""


def jj_commit_sha() -> str:
    # Get the first 9 characters of the commit id of the parent of the current
    # working copy.
    return subprocess.check_output(
        ["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id.shortest(9)"],
        encoding="utf-8",
    ).strip()


def jj_dirty_suffix() -> str:
    # This `jj log` template returns "true" if the current working copy is
    # empty, otherwise "false".
    status = subprocess.check_output(
        ["jj", "log", "-r", "@", "--no-graph", "-T", "empty"], encoding="utf-8"
    ).strip()
    return ".dirty" if status == "false" else ""


def main() -> None:
    if use_jj():
        print("STABLE_GIT_COMMIT_SHA " + jj_commit_sha())
        print("STABLE_GIT_DIRTY_SUFFIX " + jj_dirty_suffix())
    else:
        print("STABLE_GIT_COMMIT_SHA " + git_commit_sha())
        print("STABLE_GIT_DIRTY_SUFFIX " + git_dirty_suffix())


if __name__ == "__main__":
    main()
