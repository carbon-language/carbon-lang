---
name: Jujutsu (jj) usage
description:
    Instructions for using Jujutsu (jj) for version control in the Carbon
    repository.
---

# Jujutsu (jj) usage

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Jujutsu](https://github.com/jj-vcs/jj) is a Git-compatible version control
system that may be used in Carbon checkouts.

> [!IMPORTANT] You can detect if Jujutsu is in use by checking for a `.jj`
> directory in the repository root. If present, you **must** use `jj` and **must
> not** use `git`. If absent, you **must not** use `jj`.

## General usage

Always use the `--no-pager` flag when invoking `jj` to prevent the command from
blocking or waiting for terminal paging.

## Common commands

### Syncing with remote

-   **Fetch from remote**: `jj --no-pager git fetch`
-   **Create a new change on top of trunk**: `jj --no-pager new trunk`
-   **Show repository status**: `jj --no-pager status`
-   **Show commit history**: `jj --no-pager log`

### Managing changes

-   **View diff of current changes**: `jj --no-pager diff`
-   **Commit changes**: `jj --no-pager commit`
    -   _Note_: Prefer using `jj commit` over the combination of `jj describe`
        and `jj new`.
-   **Abandon/discard current changes**: `jj --no-pager abandon`
-   **Rebase current change onto trunk**: `jj --no-pager rebase -o trunk`
