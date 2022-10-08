#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sync directories in the main Carbon repository into dedicated child
# repositories to better match repository-oriented installing and tooling.

set -eux

ORIGIN_DIR="$PWD"
COMMIT_SHA="$(git rev-parse --short $GITHUB_SHA)"
COMMIT_SUMMARY="Original $(git show -s --pretty=full "$COMMIT_SHA")

$(git diff --summary "${COMMIT_SHA}^!")
"

# Setup global git configuration.
GIT_USERNAME="CarbonInfraBot"
git config --global user.email "carbon-external-infra@google.com"
git config --global user.name "$GIT_USERNAME"

declare -A MIRRORS
MIRRORS["utils/vim"]="vim-carbon-lang"

for dir in "${!MIRRORS[@]}"; do
  SRC_DIR="$dir"
  DEST_REPO="${MIRRORS[$SRC_DIR]}"
  DEST_REPO_URL="https://$GIT_USERNAME:$API_TOKEN_GITHUB@github.com/carbon-language/$DEST_REPO.git"
  DEST_CLONE_DIR="$(mktemp -d)"

  git clone --single-branch "$DEST_REPO_URL" "$DEST_CLONE_DIR"
  cd "$DEST_CLONE_DIR"

  # Print out the destination repository to help with debugging failures in
  # GitHub's actions.
  ls -al

  # Remove all the existing files to rebuild it from scratch. We ignore when
  # this matches no files to handle freshly created repositories. Also print the
  # status afterward for debugging.
  git rm --ignore-unmatch -r .
  git status

  # Copy the basic framework from the origin repository.
  cp "$ORIGIN_DIR/.gitignore" \
    "$ORIGIN_DIR/CODE_OF_CONDUCT.md" \
    "$ORIGIN_DIR/LICENSE" \
    .

  # Copy the mirrored directory. We use `rsync` to get a more reliable way of
  # handling the mirroring of the contents of a directory. We also make this
  # verbose to help with debugging action failures on GitHub.
  rsync -av "$ORIGIN_DIR/$SRC_DIR/" .

  # Add back all the files now, and print the status for debugging.
  git add -A
  git status

  # See if there is anything to commit and push. This works the same way as
  # diff(1) and so exits zero when there are no changes.
  if ! git diff --cached --quiet; then
    # Commit the new state.
    git commit -F- <<EOF
Sync $DEST_REPO to carbon-language/carbon-lang@$COMMIT_SHA

$COMMIT_SUMMARY
EOF
    git log

    # Push the new commit.
    git push
  fi

  # Cleanup.
  cd "$ORIGIN_DIR"
  rm -rf "$DEST_CLONE_DIR"
done
