# Defining the 0.1 language

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2759)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Narrowing the proposed milestone definitions to just 0.1 initially.](#narrowing-the-proposed-milestone-definitions-to-just-01-initially)
    -   [Make a more incremental, less ambitious initial milestone.](#make-a-more-incremental-less-ambitious-initial-milestone)
    -   [Skip the 0.1 milestone and aim for feature completeness.](#skip-the-01-milestone-and-aim-for-feature-completeness)

<!-- tocstop -->

## Abstract

Provides a clear definition of our goals for the 0.1 Carbon language, and a
concrete feature-set that is expected to satisfy these goals.

The high level goal proposed for the 0.1 language is to reach an evaluation-MVP
(Minimal Viable Product): it should be sufficiently complete to evaluate its
suitability specifically with respect to fitness as a C++ successor language.

The features proposed for 0.1 language in turn focus on C++ interoperability and
a minimal subset of foundational aspects of the language.

Beyond the language itself, the other project features and milestones proposed
focus on enabling evaluation of the language design and interoperating with C++
in practice.

## Problem

The initial set of concrete goals for the Carbon project have focused on
establishing the project itself (governance, evolution, community,
infrastructure, etc.) and exploratory work on both the language design and
implementation.

The project is now reasonably healthy and executing effectively in the open, and
we've made enough exploratory progress on both design and implementation that we
need to pick a much more specific target to prioritize further work effectively.

The obvious next goal, as outlined in our 2023 roadmap, is to prepare Carbon for
detailed and in-depth evaluation by potential users to see if it could
potentially meet their needs for a C++ successor language. While some aspects of
our potential value proposition can be considered abstractly, many of the hard
questions around "will it work with my code?" and "how well will it work?" need
direct evaluation to gain confidence.

We need to prioritize a specific set of features and milestones that are
sufficient for this evaluation. Once this 0.1 language is "done" we can start
advocating for in-depth evaluation.

## Background

-   [Carbon's 2023 roadmap](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/roadmap.md)

## Proposal

We suggest a set of initial milestones for the Carbon project with version
numbers to make them easy to talk about. For each milestone, we try to outline
what the theme is, provide some concrete goals that should be addressed, and a
list of concrete features we expect to be complete.

While the proposal only suggests three milestones initially, it isn't precluding
adding additional milestones when and where they make sense as the project
progresses.

This proposal is suggesting a concrete and near-complete set of features for the
first milestone. While this list may be updated based on new information, we are
reasonably confident in its completeness. However, subsequent milestones are
intentionally left incomplete and open-ended. They will be filled in as they get
closer. The intent in having them here is to allow us to explicitly defer things
to later milestones without leaving them out entirely in cases where there is
specific interest in knowing when a major feature is expected.

**Summary of proposed milestones:**

-   "0.1" -- the MVP (Minimum Viable Product) language for the _evaluation_ of
    Carbon by C++ community and developers.
-   "0.2" -- feature complete for initial users to finish their evaluation and
    the project to conclude its experiment
-   "1.0" -- the result of a successful experiment and production-ready

One important detail is that a core goal of Carbon is to be a language prepared
for on-going evolution. This means that even if Carbon gets to 1.0, that isn't
expected to be a _final_ version in any sense, nor is any version expected to be
final. Carbon should continue to evolve and develop going forward, and a
comprehensive plan for that is
[part of our later milestones](/docs/project/milestones.md#features-explicitly-deferred-beyond-02).

## Details

This proposal adds a [milestones](/docs/project/milestones.md) document to the
Carbon project with the full details of the initial milestones.

## Rationale

This proposal is advancing Carbon's goal for the community and culture, and
on-going language evolution:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   We need to engage the Carbon community and the broader C++ community in
        evaluating the Carbon experiment, and this proposal identifies concrete
        steps we expect to take in order to ensure we can do this.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Clearly define milestones help set direction and make language evolution
        effective. These milestones will provide that directional clarity for
        Carbon and markers where we can measure how effectively evolution is
        proceeding.

## Alternatives considered

### Narrowing the proposed milestone definitions to just 0.1 initially.

While this would narrow the scope of the proposal and remove some of the more
vague aspects, it would make it difficult for readers to understand when
something is missing entirely versus when it is merely deferred for a later
milestone.

We also expect that explicitly deferring things in this way will make it easier
to focus our energy and efforts on the next milestone by avoiding distractions
of features that _might_ be interesting absent that deferral.

### Make a more incremental, less ambitious initial milestone.

The initial milestone currently proposed is relatively ambitious, and much
larger than most programming language MVPs. Carbon could have a much less
aggressive initial milestone and hit it both sooner and with higher confidence.

Unfortunately, a more incremental milestone doesn't seem to add much value due
to the inherent goal of evaluating Carbon _in the context of C++ as it exists_.
That context forces us to have a much larger initial feature set in order to not
end up in somewhat of an "apples versus oranges" comparison where the feature
sets are so different as to thwart any attempt at in-depth comparison and
evaluation.

### Skip the 0.1 milestone and aim for feature completeness.

We could have fewer milestones overall and aim for the more ambitious goal. We
know that 0.1 will be insufficient to finish most real evaluations of the Carbon
experiment, and we might waste time putting it out and then iterating towards
0.2.

However, in general software and project planning is not hindered by having more
incremental milestones. If anything, having more incremental milestones is often
useful for staying focused and making rapid progress. Moreover, we expect to be
able to parallelize a reasonably large amount of time during the initial
evaluation with completing 0.2. By putting 0.1 out first we hope to
significantly reduce the latency on getting initial feedback from users on
whether the Carbon experiment is making sense for their use.
