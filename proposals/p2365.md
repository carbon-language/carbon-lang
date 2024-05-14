# Remove artificial version ceiling on C++ interop.

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2365)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Do nothing](#do-nothing)
    -   [Enumerate specific versions and feature sets](#enumerate-specific-versions-and-feature-sets)

<!-- tocstop -->

## Abstract

Remove a confusing mention of a specific version of C++ (C++17) from the
interoperability goals.

Expand the content of the goals to make it clear that we have a moving and
ongoing target of C++ as it continues to evolve. Also emphasize that we will
prioritize among the different features during Carbon's development based on how
they impact the overall project.

However, this intends to preserve the fact that there may exist long-tail or
corner-case features in C++ that never end up with high quality or exhaustive
support in our interop story simply because their impact on Carbon users is
sufficiently small that it doesn't justify the cost.

## Problem

The text referring to C++17 was bound to become out-of-date as time marched
onward. As a result, it was increasingly confusing and anchored us in the past.

It was also easily misunderstood in several cases as foreclosing _any_ effort to
interoperate with future versions or features of C++. While at the time written,
this may not have been a priority, it seems to increasingly be a source of
confusion or friction for users without benefit.

## Proposal

Remove the specific version and replace it with an attempt to explain the
long-term goal and how we will prioritize within that long-term goal.
Eventually, we should be aiming to support all major features and versions of
C++ that are used in the industry. However, we should be pragmatic in our
prioritization by focusing on those features and versions with the maximum
impact on the project.

## Details

See the updated text in the
[interop goals](/docs/design/interoperability/philosophy_and_goals.md#never-require-bridge-code).

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   This change should address confusion across the community, especially as
        that community has grown.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Our goal is overall interoperability with C++ and the new wording should
        better capture that.

## Alternatives considered

### Do nothing

We could not make this change, and live with some amount of confusion. However,
that seems to be enough confusion to be a distraction and so it seems worth
updating our documentation.

### Enumerate specific versions and feature sets

We could try to enumerate specific versions and feature sets but it seems likely
for this list to change frequently and be a new source of confusion or
maintenance burden.

Given the current stage of the project, it seems preferable to instead describe
the high level goal and how we expect to prioritize features. The actual work to
design interop can in turn present the specific features covered.
