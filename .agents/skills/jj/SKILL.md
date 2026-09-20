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

### Working with a stack of changes

A change is often built as a stack of commits sent up as a single pull request.
The stack is not necessarily based on `trunk`; it may be based on another change
that is itself still in flight.

> [!WARNING] **Never rewrite the history of a change that has been submitted as
> a pull request.** Reviewers track a PR by its commits, and squashing,
> reordering, or abandoning them discards review that is already in progress.
> This cannot be undone from their side.
>
> Before rewriting history in any other case, propose the exact command and wait
> for confirmation. This applies to `squash`, `rebase`, `abandon`, and
> `describe` on an existing change.

#### Finding the base of the stack

Bookmarks delimit the stack. List the bookmarks that are ancestors of the
working copy, nearest first:

```bash
jj --no-pager log -r '::@ & bookmarks()'
```

Reading the result takes care, because two situations produce similar output:

-   **Editing an existing change.** The nearest bookmark names the change being
    worked on, and the bookmark below it is the base.
-   **Starting a new change.** The commits above the nearest bookmark have no
    bookmark of their own yet, so the nearest bookmark is itself the base.

`trunk` is only ever a base. Finding `trunk` nearest means new work is being
built on top of it, never that `trunk` itself is being worked on.

The graph does not distinguish the two cases: an unbookmarked or empty commit
above a bookmark may be the next commit of that change or the start of a new
one. Ask which it is when it is not clear, and ask before choosing where a fix
should land rather than after. Guessing wrong means squashing into a change that
may already be under review.

Once the base is known, use it to scope commands to the current stack:

```bash
jj --no-pager log -r '<base-bookmark>..@'
```

#### Managing the stack

-   **Fold a fix into an earlier change**:
    `jj --no-pager squash --into <change-id> [path]`. Follow-up fixes and
    formatter reflows belong in the change that introduced the code, not in a
    trailing "fixes" commit, unless that change has already been submitted.
    Naming a path squashes only that part of the working copy, leaving unrelated
    work in place.
-   **Descriptions**: only one change in the stack needs a long description, the
    one used as the pull request description. Every other change gets a short
    one-line summary. Do not repeat the long text across the stack.
