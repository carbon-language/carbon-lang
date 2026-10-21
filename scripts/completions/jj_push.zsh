# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Zsh completions for a `jj push` alias that runs `scripts/jj_push.sh`. See
# `scripts/completions/README.md` for how to install this.
#
# `jj` only knows the name of an alias, not what it expands to, so it completes
# file names after `jj push`. Rewriting `push` to `git push` in the command line
# before passing it to `jj` gets the completions of `jj git push`.

source <(COMPLETE=zsh jj)

_carbon_jj_complete() {
  local i

  # Only look before the cursor. A `push` at the cursor is still being typed.
  for ((i = 2; i < CURRENT; i++)); do
    case ${words[i]} in
      -*) ;;
      push)
        words=(${words[1, i - 1]} git push ${words[i + 1, -1]})
        ((CURRENT++))
        break
        ;;
      *) break ;;
    esac
  done

  _clap_dynamic_completer_jj "$@"
}

compdef _carbon_jj_complete jj
