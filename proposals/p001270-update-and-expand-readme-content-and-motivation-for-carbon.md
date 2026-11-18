# Update and expand `README` content and motivation for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1270)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Avoid discussion of motivations](#avoid-discussion-of-motivations)
    -   [Avoid summarizing key language design areas](#avoid-summarizing-key-language-design-areas)
    -   [Avoid discussing the difficulty of directly and incrementally improving C++](#avoid-discussing-the-difficulty-of-directly-and-incrementally-improving-c)

<!-- tocstop -->

## Problem

Feedback from folks outside of the immediate team working on Carbon surfaced
both some problems with the exact phrasing of our main README content, but more
importantly some major _gaps_ in our overall documentation. Specifically, we
failed to really explain our motivations for building Carbon and why this
approach might make sense.

Given the significance of the new content and the importance of these specific
topics, this level of change seems important to go through the proposal process.

## Background

We've been trying to polish and improve the positioning and explanation of
Carbon to help understand whether it makes sense to shift the project towards
being a _public_ experiment instead of private one.

## Proposal

This PR includes a significant update to the [`README`](/README.md) content, as
well as adding [a new document](/docs/project/difficulties_improving_cpp.md) to
explain the difficulties with incrementally improving C++.

It also tweaks the wording of our goals to try to further reduce confusion.

## Details

See the pull request for the detailed change.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   We should document clearly our motivations to ensure that aspect of the
        project remains transparent and clear.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Understanding the _motivations_ of the Carbon project will be important
        for future language evolution efforts.

## Alternatives considered

### Avoid discussion of motivations

We could instead choose to avoid discussion of the project's motivations. This
has largely been the status-quo prior to this change.

Advantages:

-   Less text.
-   Fewer opportunities for a misunderstanding to develop.

Disadvantages:

-   Fails to be transparent. We _do_ have motivations, and we can't
    realistically pretend otherwise.
-   Because we all _do_ have motivations, failing to write them down will
    largely result in an inconsistent and lower-quality presentation of them in
    informal discussions and forums.

### Avoid summarizing key language design areas

This proposal suggests some brief summaries around both
[generics](/README.md#generics) and [memory safety](/README.md#memory-safety).
We could instead skip these or only have a brief mention of these.

Advantages:

-   Less text.
-   May be inaccurate and will run the risk of drifting out of date.
    -   This was a larger concern previously when for example generics was
        undergoing more active development.

Disadvantages:

-   Fails to give people an easily consumed entry into some of the really
    exciting aspects of the language design.
-   Memory safety at least will likely be an immediate question for readers
    where we can front-load a well considered answer.

### Avoid discussing the difficulty of directly and incrementally improving C++

Previously we didn't go into details about the difficulties with incrementally
improving C++ itself that are an essential component of the motivation for
Carbon. We could stay with that approach.

Advantages:

-   Less text.
-   A very contentious subject that will have many divergent, well-reasoned, and
    strongly held positions.
-   Prior attempts to articulate this have ended up being easily misunderstood
    or implying significantly more than was intended in a way that actually
    reduced alignment between different readers rather than building alignment
    and shared understanding.

Disadvantages:

-   Despite the _difficulty_ of articulating this, it remains _important_. We
    shouldn't avoid doing the work here merely because it is difficult.
-   Omitting the discussion of these difficulties runs a risk of seeming
    disingenuous -- the premise of the Carbon project makes it clear that there
    is a significant motivation here.
-   Overcoming the difficulty of articulating these difficulties well and in an
    understandable form will significantly strengthen the Carbon project's
    overall motivation and how it can engage with the broader industry.
