# Roadmap for 2022

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1025)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Retrospective on 2021](#retrospective-on-2021)
    -   [Broaden core team representation so no organization is >50%](#broaden-core-team-representation-so-no-organization-is-50)
    -   [Example ports of C++ libraries to Carbon (100% of woff2, 99% of RE2)](#example-ports-of-c-libraries-to-carbon-100-of-woff2-99-of-re2)
    -   [Demo implementation of core features with working examples](#demo-implementation-of-core-features-with-working-examples)
    -   [Executable semantic specification for core features with test cases](#executable-semantic-specification-for-core-features-with-test-cases)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

It's (past) time to update our roadmap for 2022, following Carbon's annual
[roadmap process](/docs/project/roadmap_process.md).

## Background

Carbon has an annual roadmap to align and focus the work of the teams and
community. For 2021, our main objective was to speed up the development of the
Carbon project while it remains a private experiment, by:

-   Increasing the investment by existing individuals and organizations.
-   Increasing the breadth of different individuals and organizations investing
    in Carbon.

## Proposal

We have two primary goals for 2022:

-   Shift the experiment to being public.
-   Reach the point where the core language design is substantially complete.

See the
[updated roadmap](https://github.com/carbon-language/carbon-lang/pull/1025/files)
for more details.

## Retrospective on 2021

As we plan for 2022, we should look at how well we did at achieving our
[objectives for 2021 and their key results](https://github.com/carbon-language/carbon-lang/blob/9523ac97bf5c3b7e52fa14299c1391c62dd907f5/docs/project/roadmap.md).

### Broaden core team representation so no organization is >50%

> Our goal is that no single organization makes up >50% of the core team to
> ensure that we are including as broad and representative a set of perspectives
> in the evolution of Carbon as possible.

In 2021, we dissolved the core team, and introduced a set of three leads. Two of
those leads represent the same organization. The leads now have more
organizational diversity than the core team did at the start of 2021, so this is
a partial success. (And, vacuously, no organization makes up any part of the
core team any more!)

In the wider Carbon community, most active participants and nearly all proposals
are still from a single organization, but we are seeing significant and
increasing contribution outside that organization.

The spirit of this goal has been retained for 2022, but it has been reformulated
to better fit our current organization and governance model, and somewhat
reduced in scope. Instead of looking for <50% of the Carbon leadership from any
one organization, we're now looking for <50% of active participants from any one
organization. In future years, we hope to also reach the point where <50% of the
Carbon leads are from any one organization, but that doesn't seem like a
realistic goal for 2022.

### Example ports of C++ libraries to Carbon (100% of woff2, 99% of RE2)

We did not make much progress on this goal in 2021, in part because we made less
progress on the Carbon language design in 2021 than anticipated. It remains
important that we do this work, for the same reasons as in 2021: it both
measures our progress solidifying Carbon's design and demonstrating the value
proposition of Carbon.

For 2022, as we move towards going public and completing the language design,
our expectations for these ports become higher, and we now aim to not only
provide the ported example code but also to have a sufficiently complete
executable semantics implementation that parts of it can be demonstrated to work
correctly.

### Demo implementation of core features with working examples

> A core set of Carbon features should be implemented sufficiently to build
> working examples of those features and run them successfully.

The toolchain supports parsing for many basic features, such as functions,
variables, operators, and so on, but no type-checking or code generation.

> Basic benchmarking of the different phases of compilation (lexing, parsing,
> etc).

We have some benchmarks accompanying the toolchain, but the coverage here is
incomplete.

This takes us some of the way to our intended outcome, but there's a lot more to
do.

### Executable semantic specification for core features with test cases

> This should include both a human readable rendering of the formal semantics as
> well as an execution environment to run test cases through those semantics.
> [...]

In the 2021 roadmap, we prioritized completing the demo toolchain implementation
over work on executable semantics. We did not end up following that ethos
throughout 2021, and the outcome is that most of the implementation work was
performed in executable semantics rather than in the toolchain. It would not be
unfair to say that executable semantics has ended up as a better model of a demo
Carbon implementation than the toolchain.

Nonetheless, we made great progress here. Executable semantics supports a broad
subset of the currently approved Carbon feature set, and some things beyond that
feature set. What is less clear is whether our experiment of having a clear and
precise formal specification expressed as an implementation that favors clarity
of exposition over all else is successful.

## Rationale based on Carbon's goals

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   A roadmap is an important tool for setting community expectations and
        direction.
    -   Going public is fundamental to the open community we aim to have.
    -   Broadening participation is an essential factor in building a diverse
        and welcoming community.

## Alternatives considered
