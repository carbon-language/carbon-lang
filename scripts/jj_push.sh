#!/usr/bin/env bash
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs prek over the commits `jj git push` would send, and pushes only if they
# pass. Takes the same arguments as `jj git push`.

set -eu

JJ_PREK="$(dirname "${BASH_SOURCE[0]}")/jj_prek.sh"

# Succeeds if the working-copy commit is in the given revset. This snapshots the
# working copy, so it sees anything the hooks rewrote.
working_copy_is() {
  [[ -n "$(jj log --no-graph -r "@ & ($1)" --template 'commit_id')" ]]
}

# Succeeds if the working copy is an empty, undescribed child of the target,
# which hooks can run in and write their fixes into.
working_copy_sits_on() {
  working_copy_is "empty() & description(exact:\"\") & children($1)"
}

# Returns the working copy to where it started. jj discards an empty,
# undescribed commit when the working copy moves off it, so the original may be
# gone. Build a new one on the same parents in that case.
restore_working_copy() {
  if [[ -n "$(jj log --no-graph --ignore-working-copy \
    -r "present($ORIG_CHANGE)" --template 'commit_id')" ]]; then
    jj edit --quiet "$ORIG_CHANGE"
  else
    jj new --quiet $ORIG_PARENTS
  fi
}

# `--help` describes `jj git push`, and has nothing to check.
for arg in "$@"; do
  case "$arg" in
    -h | --help)
      exec jj git push "$@"
      ;;
  esac
done

# Ask jj what the push would do. This checks the arguments and lists the commits
# being sent, without contacting the remote. A `--dry-run` already in `$@` is
# harmless here, and still suppresses the push at the end.
if ! PLAN="$(jj git push --dry-run "$@" 2>&1)"; then
  echo "$PLAN" >&2
  exit 1
fi

REMOTE="$(sed -n 's/^Changes to push to \(.*\):$/\1/p' <<<"$PLAN")"

# Each updated bookmark or tag reports the commit it moves to. Deletions have no
# such commit, and send nothing to check.
TARGETS="$(sed -n 's/^  \(bookmark\|tag\): .* to \([0-9a-f]\{8,\}\)\]$/\2/p' <<<"$PLAN" |
  paste -sd '|')"

# Nothing to check, so just push.
if [[ -z "$REMOTE" || -z "$TARGETS" ]]; then
  exec jj git push "$@"
fi

# Only the heads need checking; a head's range covers everything below it.
HEADS="$(jj log --no-graph --ignore-working-copy -r "heads($TARGETS)" \
  --template 'commit_id ++ "\n"')"

ORIG_CHANGE="$(jj log --no-graph -r @ --template 'change_id')"
ORIG_PARENTS="$(jj log --no-graph -r 'parents(@)' --template 'commit_id ++ " "')"

for target in $HEADS; do
  # Check with the working copy on top of the target.
  made_scratch=0
  if ! working_copy_sits_on "$target"; then
    jj new --quiet "$target"
    made_scratch=1
  fi

  # Check from the newest ancestor already on the remote. When there is none,
  # there is no range to diff, so check every file.
  base="$(jj log --no-graph --ignore-working-copy \
    -r "heads(::$target & ::remote_bookmarks(remote=exact:$REMOTE))" \
    --template 'commit_id ++ "\n"' | head -n 1)"
  if [[ -n "$base" ]]; then
    check=(--from-ref "$base" --to-ref "$target")
  else
    check=(--all-files)
  fi

  result=0
  "$JJ_PREK" "${check[@]}" || result=$?

  # Discard the scratch commit unless the hooks wrote something into it.
  if ((made_scratch)) && working_copy_is 'empty()'; then
    restore_working_copy
  fi

  if ((result)); then
    echo >&2
    if ! working_copy_is 'empty()'; then
      change="$(jj log --no-graph -r @ --template 'change_id.shortest()')"
      echo "Hooks changed files. They are in $change." >&2
    fi
    echo "Error: checks failed, nothing pushed." >&2
    exit 1
  fi
done

exec jj git push "$@"
