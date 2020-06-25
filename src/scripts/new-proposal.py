#!/usr/bin/env python3

"""Prepares a new proposal file and PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os

if __name__ == "__main__":
  # Verify gh is installed.
  # Create a proposal branch.
  # Copy template.md to new.md.
  # Create a PR with WIP+proposal labels.
  # Rename new.md to p####.md.
  # Update p####.md with PR information.
  # Push the PR update.
