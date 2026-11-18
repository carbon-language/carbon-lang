# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Fish completions for a `jj push` alias that runs `scripts/jj_push.sh`. See
# `scripts/completions/README.md` for how to install this.
#
# `jj` only knows the name of an alias, not what it expands to, so it completes
# file names after `jj push`. Rewriting `push` to `git push` in the command line
# before passing it to `jj` gets the completions of `jj git push`.
#
# This replaces the completions fish loads for `jj`, so it has to be installed
# with that file's name. Loading both leaves `jj`'s file-name completion
# registered.

function __jj_completion_tokens --description 'Command line tokens, with the `push` alias expanded'
    # `--cut-at-cursor` drops the token being completed, so any `push` here is
    # a complete word.
    set -l tokens (commandline --current-process --tokenize --cut-at-cursor)
    for i in (seq 2 (count $tokens))
        switch $tokens[$i]
            case '-*'
                continue
            case push
                printf '%s\n' $tokens[1..(math $i - 1)] git push \
                    $tokens[(math $i + 1)..-1]
                return
            case '*'
                break
        end
    end
    printf '%s\n' $tokens
end

complete -e -c jj
complete --keep-order --exclusive --command jj \
    --arguments "(COMPLETE=fish jj -- (__jj_completion_tokens) (commandline --current-token))"
