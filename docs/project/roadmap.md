# Roadmap

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Objective for 2023: get ready to evaluate the Carbon Language.](#objective-for-2023-get-ready-to-evaluate-the-carbon-language)
-   [Key results in 2023](#key-results-in-2023)
    -   [A concrete definition of our Minimum Viable Product for evaluation, the 0.1 language](#a-concrete-definition-of-our-minimum-viable-product-for-evaluation-the-01-language)
    -   [Complete design coverage of the 0.1 language's necessary features](#complete-design-coverage-of-the-01-languages-necessary-features)
    -   [Complete 0.1 language implementation coverage in the Carbon Explorer](#complete-01-language-implementation-coverage-in-the-carbon-explorer)
    -   [A toolchain that can build a minimal mixed C++ and Carbon program](#a-toolchain-that-can-build-a-minimal-mixed-c-and-carbon-program)
    -   [Give talks at 2-3 conferences covering 3-4 different Carbon topics](#give-talks-at-2-3-conferences-covering-3-4-different-carbon-topics)
-   [Beyond 2023](#beyond-2023)
    -   [Potential 2024 goals: ship a working 0.1 language for evaluation](#potential-2024-goals-ship-a-working-01-language-for-evaluation)
    -   [Potential 2025-2026 goals: finish 0.2 language, stop experimenting](#potential-2025-2026-goals-finish-02-language-stop-experimenting)
    -   [Potential goals _beyond_ 2026: ship 1.0 language & organization](#potential-goals-beyond-2026-ship-10-language--organization)

<!-- tocstop -->

## Objective for 2023: get ready to evaluate the Carbon Language.

Our focus throughout 2023 will be to get the Carbon Language and project ready
for serious evaluation of the experiment. There are two aspects to this
evaluation:

1. The language and tools need to be complete enough to evaluate.
2. The users and communities we are targeting need both context and awareness of
   the technical ideas and design principles on which Carbon is built.

## Key results in 2023

### A concrete definition of our Minimum Viable Product for evaluation, the 0.1 language

While we have talked about our 0.1 language, or our Minimum Viable Product (MVP)
for evaluation purposes, we need to pin down exactly what this includes. We need
concrete milestones that need to be reached for us to have confidence in
potential users and communities being able to evaluate Carbon as a successor to
C++.

We expect this to include a reasonably precise set of requirements across:

-   Necessary language features
-   Nice-to-have features
-   Features can we omit and reasonably expect to not obstruct credible
    evaluation
-   Implementation coverage of those features in the Carbon Explorer to validate
    the design
-   Implementation coverage of those features in the toolchain
-   Quality of implementation in the toolchain, both in general and for specific
    features
-   Necessary documentation, strategy, or other supporting material

We expect this to also include at least enough of Carbon's features and C++
interop to support [our basic example](/docs/images/snippets.md#mixed).

Note that we don't expect to finish implementing the 0.1 design in 2023. Our
goal is to make sufficient progress that we can complete the implementation in
2024, but there are still many things that can go wrong and cause significant
delays.

### Complete design coverage of the 0.1 language's necessary features

This year we plan to finish the design of the necessary feature-set that we
define above for the 0.1 language.

### Complete 0.1 language implementation coverage in the Carbon Explorer

We expect to complete the level of Carbon Explorer validation of the design
needed for 0.1 in 2023 so that we have high confidence in the design's cohesion
and behavior.

### A toolchain that can build a minimal mixed C++ and Carbon program

Our end goal is to compile a minimal but non-trivial example of bi-directionally
mixing C++ and Carbon code such as our main
[example](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/images/snippets.md#mixed)
and run it successfully. However, completing everything involved in this example
isn't expected to be realistic by the end of the year. We expect to work towards
this example and in rough priority order across the following interop features
and all the Carbon features they depend on:

-   Calling C++ functions from Carbon.
-   Importing concrete C++ types as Carbon types.
-   Using Carbon generics with a C++ type in Carbon.
-   (stretch) Calling Carbon functions from C++.
-   (stretch) Importing concrete Carbon types into C++.

### Give talks at 2-3 conferences covering 3-4 different Carbon topics

We want to engage both more broadly and in more depth with the C++ community as
the Carbon 0.1 language becomes increasingly concrete. This should help set the
stage for evaluations of Carbon when 0.1 is finished and available. To broaden
our engagement, we want to give talks at 2-3 conferences spanning 2-3 geographic
regions.

We also want these talks to provide the C++ community a deeper understanding of
the Carbon language and project, spanning 3-4 different topics. For example, we
might share the details of the language design, governance, and implementation.

## Beyond 2023

Longer term goals are hard to pin down and always subject to change, but we want
to give an idea of what kinds of things are expected at a high level further out
in order to illustrate how the goals and priorities we have in 2023 feed into
subsequent years.

### Potential 2024 goals: ship a working 0.1 language for evaluation

As we adjust our schedule and roadmap to reflect the realistic rate of progress,
the _earliest_ it seems feasible to have everything we need to evaluate the 0.1
language is 2024. However, this is just a lower bound. We may discover as we
progress things that further push out the schedule here. That is the nature of
an experimental project like Carbon.

We expect that once we reach this milestone the community will be able to start
realistically evaluating Carbon as a C++ successor language. Of course, this
evaluation will take some time.

### Potential 2025-2026 goals: finish 0.2 language, stop experimenting

Once Carbon is moving quickly and getting public feedback, we should be able to
conclude the experiment. We should know if this is the right direction for
moving C++ forward for a large enough portion of the industry and community, and
whether the value proposition of this direction outweighs the cost.

However, there will still be a lot of work left to make Carbon into a production
quality language, even if the experiment concludes successfully.

Some concrete goals that might show up in this time frame:

-   Self-hosting toolchain, including sufficient Carbon standard library
    support.
-   Expand design of standard library to include, at least directionally,
    critical and complex areas. For example: concurrency/parallelism and
    networking/IO.
-   Migration tooling sufficient to use with real-world libraries and systems.
    This might be used to help with self-hosting Carbon, as well as by initial
    early adopters evaluating Carbon.
-   Create a foundation or similar organization to manage the Carbon project,
    separate from any corporate entities that fund work on Carbon.

### Potential goals _beyond_ 2026: ship 1.0 language & organization

A major milestone will be the first version of a production language. We also
plan to finish transferring all governance of Carbon to an independent open
source organization at that point. However, we won't know what a more realistic
or clear schedule for these milestones will be until we get closer.

Goals in this time frame will expand to encompass the broader ecosystem of the
language:

-   End-to-end developer tooling and experience.
-   Teaching and training material.
-   Package management.
-   Etc.
