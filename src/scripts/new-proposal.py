#!/usr/bin/env python3

"""Prepares a new proposal file and PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import shutil
import sys

_USAGE = """Usage:
  ./new-proposal.py <title>

Generates a branch and PR for a new proposal with the specified title.
"""

_PROMPT = """This will:
  - Create and switch to a new branch named '%s'.
  - Create a new proposal titled '%s'.
  - Create a PR for the proposal.

Continue? (Y/n) """

if __name__ == "__main__":
  # Require an argument.
  if len(sys.argv) != 2:
    sys.exit(_USAGE)
  title = sys.argv[1]
  # Only use the first 20 chars of the title for branch names.
  branch = 'proposal-%s' % (title.lower().replace(' ', '-')[0:20])

  # Verify git and gh are available.
  git_bin = shutil.which('git')
  if not git_bin:
    sys.exit('Missing `git` CLI.')
  gh_bin = shutil.which('gh')
  if not gh_bin:
    sys.exit('Missing `gh` CLI. https://github.com/cli/cli#installation')

  # Prompt before proceeding.
  response = "?"
  while response not in ("y", "n", ""):
    response = input(_PROMPT % (branch, title)).lower()
  if response == "n":
    sys.exit('Cancelled')

  # Create a proposal branch.
  # Copy template.md to new.md.
  # Create a PR with WIP+proposal labels.
  # Rename new.md to p####.md.
  # Update p####.md with PR information.
  # Push the PR update.
