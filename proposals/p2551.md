# Roadmap for 2023 and retrospective for 2022

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2551)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Proposal](#proposal)
-   [Retrospective on 2022](#retrospective-on-2022)
    -   [Broaden participation so no organization >50%](#broaden-participation-so-no-organization-50)
    -   [Example ports of C++ libraries to Carbon (100% of woff2, 99% of RE2)](#example-ports-of-c-libraries-to-carbon-100-of-woff2-99-of-re2)
    -   [Carbon explorer implementation of core features with test cases](#carbon-explorer-implementation-of-core-features-with-test-cases)
    -   [Demo implementation of core features with working examples](#demo-implementation-of-core-features-with-working-examples)

<!-- tocstop -->

## Abstract

We propose a roadmap for 2023 focus on:

-   Progressing towards a concrete goal of an MVP / 0.1 language.
-   Engaging more broadly and deeply with the C++ community.

We also reflect on our overly ambitious
[roadmap for 2022](https://github.com/carbon-language/carbon-lang/blob/3d90a85f2439bb74b71a553c5017012369ec0f63/docs/project/roadmap.md)
and how the year went.

## Proposal

Our primary goals for 2023 are:

-   Define a concrete set of milestones for our Minimum Viable Product or MVP.
    -   Because Carbon is an experiment, our MVP is focused on the _evaluation_
        of the Carbon language, not any other usage.
    -   We consider this MVP-for-evaluation our 0.1 language.
-   Complete all of the 0.1 design and feature milestones.
-   Complete as many of the 0.1 implementation milestones as we can.
    -   Realistically, we don't expect to finish all of them in 2023.
-   Begin actively engaging and sharing Carbon's design and ideas with the C++
    community.

See our [updated roadmap](/docs/project/roadmap.md) for more details and how we
expect to measure our success.

## Retrospective on 2022

Our
[roadmap for 2022](https://github.com/carbon-language/carbon-lang/blob/3d90a85f2439bb74b71a553c5017012369ec0f63/docs/project/roadmap.md)
was in retrospect wildly optimistic. We're sorry about that, and are going to
work to set more realistic goals and milestones going forward. That said, we
still achieved a tremendous amount and it's useful to look in some detail at how
everything went.

We had two primary goals for 2022 and somewhat split the difference between
them:

-   Make the Carbon experiment public: **100%** as we
    [announced](https://youtu.be/omrY53kbVoA) Carbon publicly in July at
    [CppNorth](https://cppnorth.ca/)!
-   Complete the language design: lots of progress, but nowhere near the finish
    line here. This ended up also being poorly defined in some cases, which
    we're going to try to address going forward.

We can also measure the key results we had in mind with more precision.

### Broaden participation so no organization >50%

> Our goal is that no single organization makes up >50% of participation in the
> Carbon project, to ensure that we are including as broad and representative a
> set of perspectives in the evolution of Carbon as possible. As a proxy for the
> amount of participation, we will count the number of active participants from
> each organization in 2022, with the aim that each organization is represented
> by less than 50% of all active participants.

The simplest participation measures are from commits to the repository over 2022
which shows 92 contributors over the year, and definitely fewer than 46 of those
from a single organization. But it's hard to consider a single typo fix in July
as being an _active_ participant. Some other measures:

-   During the last quarter of 2022 we had 14 authors contributing to patches,
    and likely the largest single organization was only 5 of them.
-   We had 171 issue authors this year, and 43 filing more than one issue. Well
    below 50% from any single organization.
-   We had 258 issue commenters, and 64 making over 4 comments over the course
    of the year, both _far_ below 50% from a single organization.
-   Looking at the weekly meetings in August, September, and October, we had
    just under 50% of the meetings with less than 50% of attendees from a single
    organization.

Largely, we feel we hit this goal solidly. However, we still see specific areas
where we need to broaden participation, for example:

-   Language proposal authors (as opposed to other kinds of PRs and commits).
-   Design discussion and meeting participation.

Some of this will likely need Carbon to make substantial progress beyond
experimentation in order to have more organizations devote the significant
resources that can be necessary for more in-depth participation.

### Example ports of C++ libraries to Carbon (100% of woff2, 99% of RE2)

We ended up de-prioritizing this entirely so we could focus on the design and
moving the project public.

### Carbon explorer implementation of core features with test cases

Like the top-level goals for 2022, this specific result was much too ambitious.
However, we have made tremendous progress on the design and the Carbon Explorer.
For example, we have a
[detailed mapping](https://github.com/carbon-language/carbon-lang/wiki/Are-we-explorer-yet%3F)
of the implementation status of the designed features, with approximately 50% of
these implemented.

The Carbon Explorer is also now integrated into the amazing
[compiler explorer](https://carbon.godbolt.org/)! This largely addresses the
core of the "repl" and experimentation access goals, and our focus has otherwise
been on completing the implementation.

### Demo implementation of core features with working examples

We ended up prioritizing the Carbon Explorer over the toolchain in 2022 in order
to have an easier path to a minimal demo implementation.

However, we did begin fleshing out more of the toolchain implementation that
will eventually be used to compile Carbon into working binaries, and it has
started to provide the first pieces of semantic analysis along with a much
improved parser for Carbon.
