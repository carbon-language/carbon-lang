# Roadmap

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Objectives for 2025: demo of C++ interop and design of memory safety](#objectives-for-2025-demo-of-c-interop-and-design-of-memory-safety)
-   [Key results in 2025](#key-results-in-2025)
    -   [Access most non-template C++ APIs in Carbon](#access-most-non-template-c-apis-in-carbon)
    -   [Access non-generic Carbon APIs in C++](#access-non-generic-carbon-apis-in-c)
    -   [Detailed safety strategy update, including expected tradeoffs and prioritization](#detailed-safety-strategy-update-including-expected-tradeoffs-and-prioritization)
    -   [Design for compile-time temporal and mutation memory safety](#design-for-compile-time-temporal-and-mutation-memory-safety)
    -   [Give talks at 2-3 conferences about Carbon topics, expanding our audience](#give-talks-at-2-3-conferences-about-carbon-topics-expanding-our-audience)
-   [Beyond 2025](#beyond-2025)
    -   [Potential 2026 goals: ship a working 0.1 language for evaluation](#potential-2026-goals-ship-a-working-01-language-for-evaluation)
    -   [Potential 2027-2028 goals: finish 0.2 language, stop experimenting](#potential-2027-2028-goals-finish-02-language-stop-experimenting)
    -   [Potential goals _beyond_ 2028: ship 1.0 language & organization](#potential-goals-beyond-2028-ship-10-language--organization)

<!-- tocstop -->

## Objectives for 2025: demo of C++ interop and design of memory safety

We have two areas of focus for 2025:

1. Get a major chunk of our C++ interop working to the point where we can
   demonstrate it in realistic scenarios.
2. Build a concrete and specific design for memory safety in Carbon.

We will scope the first one to non-template C++ APIs, and prioritize accessing
C++ APIs from Carbon. This still will require major progress on the
implementation of all the relevant Carbon features, and even design in some
cases.

The second is focused on moving from a vague direction of "we will have a memory
safe dialect of Carbon that is a reasonable default", to a specific and concrete
design. We want to be able to illustrate exactly what it will look like to
migrate existing unsafe C++ to Carbon (possibly at large scale), and then begin
incrementally adopting and integrating memory safety into that otherwise unsafe
Carbon codebase.

Achieving these should dramatically reduce the risk around Carbon, especially in
environments where memory safety is increasingly a necessary part of any future
software development plans. They will also move the project much closer to our
0.1 milestone.

## Key results in 2025

### Access most non-template C++ APIs in Carbon

Beyond excluding templates, this excludes coroutines, and any aspects that
require accessing Carbon types in C++ such as templates with Carbon types as
template arguments.

This result includes both the implementation in the toolchain and the underlying
design underpinning this implementation. It also includes implementation and
design work on necessary Carbon language features that underpin the interop
provided.

### Access non-generic Carbon APIs in C++

This excludes generics to make the scope more tractable, but this remains a bit
of a stretch goal for 2025, and how much progress we make will depend on how
many unexpected difficulties we encounter getting the other direction to work,
and any other delays.

### Detailed safety strategy update, including expected tradeoffs and prioritization

We haven't been focused on the safe side of Carbon for several years and will
need to refresh our safety strategy to reflect the current plan, as well as
expanding and making it more detailed to support building our initial memory
safety design.

### Design for compile-time temporal and mutation memory safety

We expect our memory safety story for temporal memory safety to at the highest
level follow the direction of Rust, using the type system to ensure compile-time
guarantees of safety without the runtime overhead of garbage collection or
reference counting. We want our design here to cover both temporal and mutation
safety. While the exact level of safety and the tradeoffs we're willing to
accept will be part of updating our safety strategy, at a fundamental level we
need to fully address the security requirements on memory safety, much like
other modern languages including Swift, Kotlin, Go, or Rust. A significantly
lower security bar won't be acceptable for the expected users of safe Carbon.

### Give talks at 2-3 conferences about Carbon topics, expanding our audience

Beyond continuing to share details about Carbon with the open source and C++
communities, we also want to expand our audience reach in 2025. We want to give
talks at a conference in the Asia/Pacific region, and at a conference in the
broader open source world beyond LLVM and C++ specific conferences.

## Beyond 2025

Longer term goals are hard to pin down and always subject to change, but we want
to give an idea of what kinds of things are expected at a high level further out
in order to illustrate how the goals and priorities we have in 2025 feed into
subsequent years.

### Potential 2026 goals: ship a working [0.1 language] for evaluation

[0.1 language]:
    /docs/project/milestones.md#milestone-01-a-minimum-viable-product-mvp-for-evaluation

Because we are adding a design for memory safety to our 0.1 milestone, we are
also expecting to push it out by at least a year. Shipping 0.1 in 2026 will be a
very ambitious goal and may not be possible, but the end of 2026 is now the
_soonest_ that 0.1 could realistically be ready to ship.

We expect that once we reach this milestone the community will be able to start
realistically evaluating Carbon as a C++ successor language. Of course, this
evaluation will take some time.

### Potential 2027-2028 goals: finish [0.2 language], stop experimenting

[0.2 language]:
    /docs/project/milestones.md#milestone-02-feature-complete-product-for-evaluation

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

### Potential goals _beyond_ 2028: ship [1.0 language] & organization

[1.0 language]:
    /docs/project/milestones.md#milestone-10-no-longer-an-experiment-usable-in-production

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
