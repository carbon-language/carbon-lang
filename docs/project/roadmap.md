# Roadmap

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Objective for 2024: a working toolchain that supports C++ interop](#objective-for-2024-a-working-toolchain-that-supports-c-interop)
-   [Key results in 2024](#key-results-in-2024)
    -   [Carbon's toolchain implements enough of the language to build realistic code](#carbons-toolchain-implements-enough-of-the-language-to-build-realistic-code)
    -   [Carbon's toolchain can build C++ code](#carbons-toolchain-can-build-c-code)
    -   [Carbon's toolchain works with existing, simple C++ build systems](#carbons-toolchain-works-with-existing-simple-c-build-systems)
    -   [Carbon has a design and toolchain implementation of basic C++ interop](#carbon-has-a-design-and-toolchain-implementation-of-basic-c-interop)
    -   [Give talks at 2-3 conferences covering 3-4 different Carbon topics](#give-talks-at-2-3-conferences-covering-3-4-different-carbon-topics)
    -   [Start building our initial tutorial and introductory material](#start-building-our-initial-tutorial-and-introductory-material)
-   [Beyond 2024](#beyond-2024)
    -   [Potential 2025 goals: ship a working 0.1 language for evaluation](#potential-2025-goals-ship-a-working-01-language-for-evaluation)
    -   [Potential 2026-2027 goals: finish 0.2 language, stop experimenting](#potential-2026-2027-goals-finish-02-language-stop-experimenting)
    -   [Potential goals _beyond_ 2027: ship 1.0 language & organization](#potential-goals-beyond-2027-ship-10-language--organization)

<!-- tocstop -->

## Objective for 2024: a working toolchain that supports C++ interop

Our focus for 2024 will be to get the Carbon toolchain working, including C++
interop. We see three key criteria:

-   Building realistic Carbon code for interesting interop with C++.
-   Building realistic C++ code for interesting interop with Carbon.
-   The interop itself to allow a single program mixing the two languages.

This will both allow folks to explore Carbon using a more traditional and
realistic compiler model, and allow that exploratory Carbon to lean on C++ for
libraries and other functionality that doesn't exist in Carbon yet. It will also
demonstrate how the interop will work in practice.

This objective and focus are oriented around the toolchain and implementation of
Carbon. We still expect some work on language design, but for its priority to be
driven largely as a function of being in the critical path of some aspect of our
implementation work.

## Key results in 2024

### Carbon's toolchain implements enough of the language to build realistic code

This goal is not necessarily about complete support for the entire language
design, but rather enough of it to support building the realistic and
interesting Carbon code that interoperates with C++.

Some example language features that we think are key to success here but far
from an exhaustive list:

-   Imports and a working [prelude] (the earliest stages of a standard library)
-   Operator overloading and dispatch for expressions
-   Generic types and functions
-   Templates (likely only partial support and focused on interop use cases)

[prelude]: /docs/design#name-lookup-for-common-types

### Carbon's toolchain can build C++ code

We need the toolchain to be able to build C++ code as if it were Clang in order
to build the C++ code that Carbon is interoperating with. This isn't about
building anything new or novel, but about packaging and exposing Clang for this
purpose.

### Carbon's toolchain works with existing, simple C++ build systems

We should be able to drop Carbon's toolchain into at least simple `Makefile` or
CMake build systems as a replacement for the C++ toolchain and provide a Carbon
toolchain. This doesn't include supporting everything or even moderately complex
builds; only the simplest of builds using these build systems need to work at
first.

### Carbon has a design and toolchain implementation of basic C++ interop

Our end goal is to compile a minimal but non-trivial example of bi-directionally
mixing C++ and Carbon code such as our main example and run it successfully.
However, completing everything involved in this example isn't expected to be
realistic by the end of the year. We expect to work towards this example and in
rough priority order across the following interop features and all the Carbon
features they depend on:

-   Calling C++ functions from Carbon.
-   Importing concrete C++ types as Carbon types.
-   (stretch) Using Carbon generics with a C++ type in Carbon.
-   (stretch) Calling Carbon functions from C++.
-   (stretch) Importing concrete Carbon types into C++.

### Give talks at 2-3 conferences covering 3-4 different Carbon topics

We want to continue to engage with the external C++ community as the Carbon
toolchain becomes a more real and complete toolchain. We specifically want to
share when interop becomes something people can experiment with and explore.

### Start building our initial tutorial and introductory material

Because of the nature of Carbon's experiment, the tutorial and introductory
material won't be focused on typical teaching of the language to general
developers. Instead, it will be focused on enabling C++ developers to start
evaluating specific aspects of Carbon for interoperating with existing C++
codebases.

We only expect to _start_ building this material in 2024. We want to learn what
any critical gaps are for folks to start evaluating C++ interop and how best to
close them going into 2025.

## Beyond 2024

Longer term goals are hard to pin down and always subject to change, but we want
to give an idea of what kinds of things are expected at a high level further out
in order to illustrate how the goals and priorities we have in 2024 feed into
subsequent years.

### Potential 2025 goals: ship a working [0.1 language] for evaluation

[0.1 language]:
    /docs/project/milestones.md#milestone-01-a-minimum-viable-product-mvp-for-evaluation

As we adjust our schedule and roadmap to reflect the realistic rate of progress,
the _earliest_ it seems feasible to have everything we need to evaluate the 0.1
language is 2025. We're starting to be optimistic in 2024 that we'll be able to
hit this in 2025, but ultimately this remains a lower bound. We may discover as
we progress things that further push out the schedule here. That is the nature
of an experimental project like Carbon.

We expect that once we reach this milestone the community will be able to start
realistically evaluating Carbon as a C++ successor language. Of course, this
evaluation will take some time.

### Potential 2026-2027 goals: finish [0.2 language], stop experimenting

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

### Potential goals _beyond_ 2027: ship [1.0 language] & organization

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
