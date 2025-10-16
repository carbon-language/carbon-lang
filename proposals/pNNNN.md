# Disambiguate "value binding"

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
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./new_proposal.py "TITLE"` to do new proposal setup.

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

This proposal removes the definition of the term "value binding" as a primitive
category conversion from reference to value, replacing it with the term "value
borrowing". The other meaning of "value binding", a binding declared by a value
binding pattern, is unchanged.

## Problem

The design docs currently define "value binding" in two conflicting ways: it can
mean the binding declared by a value binding pattern, or it can mean a primitive
category conversion from reference to value. The two can usually be
disambiguated based on context, but it's not always straightforward, and the
double meaning complicates naming within the toolchain implementation.

## Background

TODO: Is there any background that readers should consider to fully understand
this problem and your approach to solving it?

## Proposal

This proposal removes the definition of the term "value binding" as a primitive
category conversion from reference to value, replacing it with the term "value
borrowing". This terminology is taken (borrowed?) from Rust: the semantics of
value borrowing in Carbon are very similar to the semantics of immutable
borrowing in Rust.

## Details

See the changes elsewhere in the proposal PR.

## Rationale

Using unambiguous terminology advances our
[community and culture](/docs/project/goals.md#community-and-culture) goals, by
facilitating clear communication.

## Alternatives considered

We could instead rename the other meaning of "value binding", but that would be
considerably more difficult because that meaning appears to be more common, and
because it's part of a cluster of other heavily-used terms, such as "reference
binding" and "binding pattern", which we would need to rename for consistency.
