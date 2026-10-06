# Disallow impl in match_first twice

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7493)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)

<!-- tocstop -->

## Abstract

Disallow declaring an `impl` in the same `match_first` block more than once.

## Problem

In
[proposal 5337](/proposals/p005337-interface-extension-and-final-impl-update.md)
we have the rule "An impl may appear in at most one match_first block." This
allows an `impl` to appear in one `match_first` block more than once, which
creates ambiguity about its ordering.

```carbon
match_first {
impl forall [T: X] T as Z {}
impl forall [T: Y] T as Z {}
impl forall [T: X] T as Z {}
}
```

The rules say that the first impl in the block should be selected, so given a
type that implements `X & Y`, the first impl (with `T: X`) should be chosen. But
there is also a `T: Y` impl that wins against the last `T: X` impl. While the
order of is clear, this leads to more confusing code than is necessary. There is
no need to write an impl twice in a `match_first` block.

## Background

-   [Proposal 5337](/proposals/p005337-interface-extension-and-final-impl-update.md)
    introduced rules for `match_first` blocks.

## Proposal

Modify the rule "An impl may appear in at most one match_first block" to instead
say "An impl may appear in at most one match_first block, and only once within
it".

## Details

This will disagnose the case where an impl is written multiple times in a
`match_first` block, but the later ones would have been ignored. This allows the
user to know about code they've written that isn't doing anything.

## Rationale

This advances the
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal, by avoiding the case where the user can write code that is never used. It
may help them catch mistakes such as copy/paste errors.
