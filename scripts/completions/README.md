<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Shell completions for `jj push`

Completions for a `jj push` alias that runs
[`scripts/jj_push.sh`](/scripts/jj_push.sh). See
[the Jujutsu section of the contribution tools doc](/docs/project/contribution_tools.md#jujutsu-jj)
to set up the alias.

`jj` only knows the name of an alias, not what it expands to, so it completes
file names after `jj push`. Each file here rewrites `push` to `git push` in the
command line before passing it to `jj`, giving the alias the flags, bookmarks,
revsets, and remotes of `jj git push`.

Each file loads `jj`'s own completions itself, and needs `jj` on `PATH` when it
runs. Remove any other setup that loads `jj`'s completions.

Run the commands below from your Carbon checkout, so that
`jj workspace root` fills in its path.

## Bash

```sh
echo "source $(jj workspace root)/scripts/completions/jj_push.bash" >>~/.bashrc
```

If your distribution ships a `jj` file in
`/usr/share/bash-completion/completions`, Bash loads it on demand and it
overrides this. Symlink this file to
`~/.local/share/bash-completion/completions/jj` instead of sourcing it.

## Zsh

```sh
echo "source $(jj workspace root)/scripts/completions/jj_push.zsh" >>~/.zshrc
```

This has to come after `compinit` in `.zshrc`, so move the line if `compinit`
runs later in the file.

## Fish

Fish loads completions on demand, after running `config.fish`, so sourcing this
at startup doesn't work: what fish loads later is added on top. Install it as
the file fish loads for `jj`:

```sh
ln -s "$(jj workspace root)/scripts/completions/jj_push.fish" \
  ~/.config/fish/completions/jj.fish
```

`~/.config/fish/completions` is first in `$fish_complete_path`, so this
overrides any `jj.fish` from your distribution.
