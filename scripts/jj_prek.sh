#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs prek within a jj repository that is backed by a git repository, including
# the case where the repository is a non-colocated jj workspace.

set -eu

# Map `@` to a git commit. This deliberately doesn't pass
# `--ignore-working-copy`, so that jj first snapshots the files on disk into `@`;
# otherwise the index built below can describe stale file contents, and hooks
# both check the wrong thing and fail to write back their fixes.
HEAD="$(jj show --no-patch -r @ --template 'commit_id')"

# Find the .git directory. The working copy was snapshotted above, so this
# doesn't need to do so again.
export GIT_DIR="$(jj git root --ignore-working-copy)"

# Create a git index file describing `@`.
export GIT_INDEX_FILE="$(mktemp)"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree "$HEAD"

# Run prek with the `.git` directory and index we built earlier.
prek run --from-ref trunk --to-ref "$HEAD"
