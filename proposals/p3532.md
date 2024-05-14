# Focus implementation effort on the toolchain

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3532)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Alternatives considered](#alternatives-considered)
    -   [Do nothing](#do-nothing)
    -   [Front-load porting the functionality and output](#front-load-porting-the-functionality-and-output)

<!-- tocstop -->

## Abstract

Proposal to focus implementation effort for the next 1-2 years on the Carbon
toolchain instead of the explorer. This will impact the explorer in a few ways:

-   We will keep the explorer's code in place, building, and passing its basic
    tests. It can remain a good baseline for exploring Carbon's language
    features.
-   We won't prioritize expanding the explorer's coverage of Carbon features or
    other improvements -- it is good enough for what we need until the toolchain
    catches up.
-   We will stop actively fuzzing and expanding test coverage for the explorer.
-   Eventually, when we want to resume work on the explorer, we'll evaluate the
    best platform to build on -- the current explorer codebase or on top of the
    toolchain's semantic IR.

## Problem

Carbon is still a relatively small project, and we need to avoid spreading
ourselves out too much over too many efforts. Currently, we're maintaining two
implementation codebases -- the Carbon Explorer and the Carbon Toolchain. While
historically these have served importantly different needs of the project, the
current state has changed. We're now moving slower as a consequence of spreading
out our energy across both.

## Background

Originally, the Carbon Explorer served two major purposes:

1. A high-level or "abstract machine" executable semantic model for the design
   of the language.

2. A rapid prototyping platform with a generated parser and maximally simple &
   traditional internal architecture (ASTs, etc.).

The first purpose and use case remains extremely important and something that we
should support. However, long-term it may make more sense to build on the same
core internal representation as the toolchain in order to avoid duplicated
effort and maximize its utility. We'll understand the tradeoffs there better
once the toolchain is similarly feature complete.

The second use case is no longer critical. At the time, the toolchain was
nascent and using a highly experimental architecture with many unknowns. It was
hard to be confident it would work at all, much less add a feature to it. We
also had a large number of design features for Carbon that needed to have some
form of implementation experience in order to validate the designs themselves.

The current state of the project is very different. The driving purpose of
prototyping many of the core language features, from inheritance, to generics,
to expression category, has been achieved. While there are large designs that
are not yet in the explorer, we have proven out the most critical components and
gotten critical implementation feedback.

The toolchain is also rapidly maturing and its core architecture is holding up
well. While implementing a new feature on top of the toolchain's architecture is
significantly more expensive than on the explorer's architecture, it isn't
starting from nothing and the most uncertain aspects of the core architecture
and design have become concrete. It is also increasingly necessary for us to get
features implemented here so that we can evaluate them in a realistic
compilation context and integrate them with C++ interop which we only anticipate
to be able to build in the toolchain architecture.

The toolchain has also developed a Semantics IR that is an especially
interesting potential platform for building support for the executable semantics
use case. When the toolchain reaches feature coverage and maturity, it seems
important to carefully consider whether that's the best approach to take, and
what the tradeoffs are there.

## Proposal

We should focus all of our efforts for the next 1-2 years on the toolchain, in
order to make as much progress as possible. This in turn is likely to give us
the best evaluation of the Carbon experiment in the shortest time frame.

We should keep the Explorer's code in place, building, and passing its basic
regression tests because the built artifacts of the Explorer remain really
valuable given its coverage of our design's feature sets. But we shouldn't take
on new work, or drive significant fuzzing, refactoring, or code cleanups in the
explorer.

We should also explicitly consider re-building the core functionality of the
Explorer's abstract machine semantic execution on top of the toolchain's
Semantics IR model. If this is the right approach given all the tradeoffs, it
should still leverage the existing work done and lessons learned on the Explorer
for how to model these semantics and how to render them in an understandable way
for users. This is another reason why we should keep all of the code in place,
building and passing its tests, until we complete this. While the code itself is
unlikely to just directly port across in this way, having the artifacts ensures
we'll have a good point of comparison and reference for the ideas and designs.
We also may find that it isn't the right direction, and instead resume work on
top of the explorer's model directly.

## Details

The short term practical details of this proposal are essentially:

-   Update all of our documentation to direct contributors to the toolchain
    rather than the Explorer.
-   Document that the explorer's code is largely an archive and we're not
    planning to do significant bug fixes, much less feature development in it.
-   Document that we plan to eventually rebuild the explorer's core
    functionality on top of the toolchain's semantic model, once that model is
    sufficiently feature complete. And that we will _not_ remove the explorer's
    code until that replacement is ready.
-   Close the various bugs open for working on the explorer with links to this
    proposal & documentation updates.
-   Close open PRs and new PRs for the explorer linking to this proposal for
    details. Note that closed PRs aren't deleted and so we can revisit them if
    our priorities change and they end up remaining relevant.
-   Update the issue form to make it clear that we're not planning to do more
    work on the explorer codebase as-is.

Long term, we need to plan to figure out how we want to pursue the core
functionality of the explorer, whether in its current implementation or on top
of the toolchain's Semantic IR model. Even if there isn't much code that can be
directly reused, we should still heavily learn from and incorporate the relevant
ideas of the design of the explorer's abstract machine model, output rendering,
and test suite.

## Alternatives considered

### Do nothing

We could simply leave things in the status-quo. However, this has the serious
downside that the contributors to Carbon are sufficiently stretched currently
that they are foregoing work on the explorer already. In some ways, at the
current size of the project, it isn't clear that we can sustain the status quo
in practice even if we don't update our documentation.

### Front-load porting the functionality and output

We could immediately try to port the functionality and output so that the
explorer tooling and workflow is immediately available on top of the toolchain's
semantic representation.

Unfortunately there is a significant amount of implementation work left for the
toolchain to catch up in terms of feature completeness with the explorer. We
need to complete the functionality before we can really pursue this path. And
this proposal is trying to maximize how much energy the project can devote to
that.

Beyond this fundamental limit, the current project roadmap and priorities focus
on getting C++ interop working with the toolchain above expanding functionality
of the explorer. As a consequence, it seems better to focus on that priority in
the toolchain and to revisit rebuilding the explorer on top of the toolchain's
semantic model later, at least after the main interop goals are achieved. At
that point, it is also likely that most if not all of the feature gaps will have
been closed.

The proposed approach to essentially archiving the explorer code seems like the
best way to align the effort and energy of the project contributors with these
priorities.
