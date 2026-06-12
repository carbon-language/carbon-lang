#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs prek within a jj repository that is backed by a git repository, including
# the case where the repository is a non-colocated jj workspace.

set -eu

# Find the .git directory, and map `@` to a git commit.
export GIT_DIR="$(jj git root --ignore-working-copy)"
HEAD="$(jj show --no-patch --ignore-working-copy -r @ --template 'commit_id')"

# Create a git index file describing `@`.
export GIT_INDEX_FILE="$(mktemp)"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree "$HEAD"

# Run prek with the `.git` directory and index we built earlier.
prek run --from-ref trunk --to-ref "$HEAD"
