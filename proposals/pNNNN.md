# Progressive disclosure principle

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [TODO: Initial proposal setup](#todo-initial-proposal-setup)
-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Rationale](#rationale)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./new_proposal.py "TITLE"` to do new proposal setup.

1. Copy this template to `new.md`, and create a commit.
2. Create a GitHub pull request, to get a pull request number.
    - Add the `proposal draft` label to the pull request.
3. Rename `new.md` to `/proposals/p####.md`, where `####` should be the pull
   request number.
4. Update the title of the proposal (the `TODO` on line 1).
5. Update the link to the pull request (the `####` on line 11).
6. Delete this section.

TODOs indicate where content should be updated for a proposal. See
[Carbon Governance and Evolution](/docs/project/evolution.md) for more details.

## Abstract

This proposal codifies our preference for designs that support "progressive
disclosure", meaning that programmers can ignore a given language concept (or
even be unaware of it) until it is directly relevant to the task they're doing.

## Problem

Carbon's design choices have been motivated in part by a desire for "progressive
disclosure", but that approach has not been codified as a design principle.

## Proposal

See
(`docs/project/principles/progressive_disclosure.md`)[/docs/project/principles/progressive_disclosure.md],
which is introduced by this proposal.

## Rationale

In order to have a vibrant
[community and culture](/docs/project/goals.md#community-and-culture), Carbon
needs to be effectively teachable and learnable. Progressive disclosure supports
that goal by minimizing the amount that must be learned at any one time, and by
enabling concepts to be introduced when they are most salient to the programmer.

Relatedly, although
["lies-to-children"](https://en.wikipedia.org/wiki/Lie-to-children) can be an
appropriate teaching tool, they also risk undermining the health of the
community by creating distance between those who know the full truth and those
who have only been taught the "lie". This principle helps minimize that risk,
through the expectation that progressive disclosure not invalidate the
programmer's prior understanding of the language.

Finally, this principle helps make Carbon code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
by minimizing the amount of Carbon knowledge that a programmer needs in order to
read, understand, or write a given piece of code.
