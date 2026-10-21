# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Bash completions for a `jj push` alias that runs `scripts/jj_push.sh`. See
# `scripts/completions/README.md` for how to install this.
#
# `jj` only knows the name of an alias, not what it expands to, so it completes
# file names after `jj push`. Rewriting `push` to `git push` in the command line
# before passing it to `jj` gets the completions of `jj git push`.

source <(COMPLETE=bash jj)

_carbon_jj_complete() {
  # Shadow the two variables `jj`'s completion function reads. Bash scopes them
  # dynamically, so it sees the rewrite below.
  local -a COMP_WORDS=("${COMP_WORDS[@]}")
  local COMP_CWORD=$COMP_CWORD
  local i

  # Only look before the cursor. A `push` at the cursor is still being typed.
  for ((i = 1; i < COMP_CWORD; i++)); do
    case "${COMP_WORDS[i]}" in
      -*) ;;
      push)
        COMP_WORDS=("${COMP_WORDS[@]:0:i}" git push "${COMP_WORDS[@]:i+1}")
        ((COMP_CWORD++))
        break
        ;;
      *) break ;;
    esac
  done

  _clap_complete_jj "$@"
}

if [[ "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -ge 4 || "${BASH_VERSINFO[0]}" -gt 4 ]]; then
  complete -o nospace -o bashdefault -o nosort -F _carbon_jj_complete jj
else
  complete -o nospace -o bashdefault -F _carbon_jj_complete jj
fi
