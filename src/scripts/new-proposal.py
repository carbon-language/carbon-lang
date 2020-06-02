#!/usr/bin/env python3

"""Prepares a new proposal file and PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import shutil

if __name__ == "__main__":
  # Verify git and gh are available.
  git_bin = shutil.which('git')
  if not git_bin:
    sys.exit('Missing `git` CLI.')
  gh_bin = shutil.which('gh')
  if not gh_bin:
    sys.exit('Missing `gh` CLI. https://github.com/cli/cli#installation')

  # Create a proposal branch.
  # Copy template.md to new.md.
  # Create a PR with WIP+proposal labels.
  # Rename new.md to p####.md.
  # Update p####.md with PR information.
  # Push the PR update.
