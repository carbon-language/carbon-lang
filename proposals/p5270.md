# Move explorer out of toolchain git repository

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5270)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Use a branch to refer to the explorer codebase](#use-a-branch-to-refer-to-the-explorer-codebase)
    -   [Do nothing](#do-nothing)

<!-- tocstop -->

## Abstract

The explorer codebase is no longer actively developed or maintained, but retains
value as a place to look for implementations of various parts of the design.
Having both the explorer and toolchain in the working git repository causes
confusion for developers and users, and increases our incremental maintenance
burden. We will move explorer out of the working git repository so that it
remains accessible but is clearly deliniated from the toolchain and is more
clearly frozen or archived.

## Problem

Users have expressed confusion when they come across examples or tests for the
explorer that involve `Main()`, whereas the design and implementation of the
toolchain have moved on to `Run()` as the entry point.

Developers have attempted to update the explorer codebase to fix style guide
issues[^1], as tools like grep or clang-tidy do not differentiate our active
toolchain codebase from the frozen explorer one.

As we update Bazel, Clang, Clang-tidy, etc, we also have to update the explorer
codebase to keep it building cleanly.

As we modify the `//testing` harness, we have to accommodate and work around the
explorer tests[^2][^3][^4].

[^1]: https://github.com/carbon-language/carbon-lang/pull/5224

[^2]: https://github.com/carbon-language/carbon-lang/pull/4979

[^3]: https://github.com/carbon-language/carbon-lang/pull/5025

[^4]: https://github.com/carbon-language/carbon-lang/pull/5036

## Background

-   Previous proposal:
    [#3532: Focus implementation effort on the toolchain](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p3532.md)
-   Kick-off of this discussion
    [#5224](https://github.com/carbon-language/carbon-lang/pull/5224#pullrequestreview-2730512195)
-   Discord discussion:
    [2025-04-02 #explorer](https://discord.com/channels/655572317891461132/763516049710120960/1356979197070803095)

## Proposal

1. Add a tag `explorer-archived` in the main `carbon-lang` git repository.
2. Create a new `explorer` repository under the `carbon-language` organization
   that only contains the `//explorer` and `//installers` directories and their
   dependencies at head.
3. Locally ensure the explorer tests build and pass under the `explorer`
   repository.
4. Add a `README.md` to the `explorer` repository that explains explorer is
   archived and not under active development.
5. Stop building, or remove the "Explorer (trunk)" compiler option from
   [carbon.compiler-explorer.com](https://carbon.compiler-explorer.com).
6. Delete `//explorer` and `//installers` in the main `carbon-lang` repository.
7. Archive the `explorer` repository in GitHub, making it read-only.

Note that fuzzer test cases from the explorer are already relocated under
`//toolchain/*/fuzzer_corpus/`.

## Rationale

The primary purpose of the explorer codebase at this time is as a demonstration
of past work implementing the carbon language design. This can help to inform
the implementation of the toolchain as it catches up in various areas, as long
as the design has not deviated from the explorer implementation.

Searching in, and providing links to the explorer codebase comes up in design
discussions occasionally [^5][^6][^7][^8][^9][^10], and we should maintain the
ability of toolchain developers to look through the explorer easily. The primary
place they do so is on GitHub, where the code can be linked to. GitHub search
only works
[on the main branch](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax#using-qualifiers)
of a repository, so the `trunk` branch for the `carbon-lang` repository. To
maintain searchability, the explorer codebase must either remain on `trunk` in
`carbon-lang` or in a sibling repository. GitHub search does continue to work in
archived repositories.

Proposal
[p3532](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p3532.md)
directed to keep the explorer codebase active in the main repository, with its
tests building and running:

> We should keep the Explorer's code in place, building, and passing its basic
> regression tests because the built artifacts of the Explorer remain really
> valuable given its coverage of our design's feature sets.

However the [problems](#problem) discussed above result from this situation. We
can gain the benefit of access to the codebase while reducing its impact on
developers and users by moving it into a separate git repository.

[^5]:
    https://discord.com/channels/655572317891461132/998959756045713438/1225116234199203860

[^6]:
    https://discord.com/channels/655572317891461132/998959756045713438/1237143981150830673

[^7]:
    https://discord.com/channels/655572317891461132/709488742942900284/1250577021474443376

[^8]:
    https://discord.com/channels/655572317891461132/748959784815951963/1255669935439482993

[^9]:
    https://discord.com/channels/655572317891461132/655578254970716160/1302033729761443963

[^10]:
    https://discord.com/channels/655572317891461132/941071822756143115/1349523309682753606

## Alternatives considered

### Use a branch to refer to the explorer codebase

We could create a branch in the `carbon-lang` repository that contains explorer,
but delete it from `trunk`. This would also allow the code to be found and
linked to, however GitHub search does not support searching in a branch. The
main purpose of the explorer code while archived is for reading the
implementation, and search is an important part of that.

### Do nothing

In the future, we may want to restart development of the explorer codebase. At
that time we would benefit from keeping the codebase building as we upgrade
Bazel, Clang, Clang-tidy, etc. However we don't currently have the resources to
build two implementations of the language, and there's no plan in our roadmaps
to restart explorer development. So the cost of letting the explorer codebase
become stale is outweighed by the costs of keeping it active.
