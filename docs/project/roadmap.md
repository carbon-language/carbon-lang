# Roadmap

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Objective for 2022: make Carbon public, finish 0.1 language](#objective-for-2022-make-carbon-public-finish-01-language)
    -   [Completing the language design](#completing-the-language-design)
    -   [Going public](#going-public)
-   [Key results in 2022](#key-results-in-2022)
    -   [Broaden participation so no organization is >50%](#broaden-participation-so-no-organization-is-50)
    -   [Example ports of C++ libraries to Carbon (100% of woff2, 99% of RE2)](#example-ports-of-c-libraries-to-carbon-100-of-woff2-99-of-re2)
        -   [Language design covers the syntax and semantics of the example port code.](#language-design-covers-the-syntax-and-semantics-of-the-example-port-code)
    -   [Demo implementation of core features with working examples](#demo-implementation-of-core-features-with-working-examples)
    -   [Carbon explorer implementation of core features with test cases](#carbon-explorer-implementation-of-core-features-with-test-cases)
-   [Beyond 2022](#beyond-2022)
    -   [Potential 2023 goals: finish 0.2 language, stop experimenting](#potential-2023-goals-finish-02-language-stop-experimenting)
    -   [Potential 2024-2025 goals: _ship_ 1.0 language & organization](#potential-2024-2025-goals-ship-10-language--organization)

<!-- tocstop -->

## Objective for 2022: make Carbon public, finish 0.1 language

We have two primary goals for 2022:

-   Shift the experiment to being public.
-   Reach the point where the core language design is substantially complete.

### Completing the language design

By the end of 2022, the core Carbon language design should be substantially
complete, including designs for expressions and statements, classes, generics
and templates, core built-in types and interfaces such as integers and pointers,
and interoperability with C++. The design choices made to reach this point are
expected to be experimental, and many of them may need revisiting before we
reach 1.0, but the broad shape of the language should be clear at this point,
and it should be possible to write non-trivial Carbon programs.

An initial rough framework for the core standard library functionality should be
provided, as necessary to support the core language components. A largely
complete implementation of the core language design should be available in
Carbon explorer. The toolchain should be able to parse the core language design,
with some support for name lookup and type-checking.

We should have begun writing non-trivial portions of the standard library, such
as common higher-level data structures and algorithms.

### Going public

At some point in 2022 we should shift the experiment to be public. This will
allow us to significantly expand both those directly involved and contributing
to Carbon but also those able to evaluate and give us feedback.

We don't expect Carbon to shift away from an experiment until after it becomes
public and after we have been able to collect and incorporate a reasonable
amount of feedback from the broader industry and community. This feedback will
be central in determining whether Carbon should continue past the experimental
stage.

## Key results in 2022

There are several milestones that we believe are on the critical path to
successfully achieving our main goal for the year, and point to concrete areas
of focus for the project.

### Broaden participation so no organization is >50%

Our goal is that no single organization makes up >50% of participation in the
Carbon project, to ensure that we are including as broad and representative a
set of perspectives in the evolution of Carbon as possible.

As a proxy for the amount of participation, we will count the number of active
participants from each organization in 2022, with the aim that each organization
is represented by less than 50% of all active participants.

There are many ways in which someone could be an active participant, and when
the leads come to reflect on this at the end of the year, we expect this to be a
judgment call. We will consider at least the following when measuring our
success on this objective:

-   Pull requests authored and reviewed, including proposals, code changes, and
    documentation changes.
-   Contribution to discussions, including Discord, teleconferences, and GitHub
    issues.

### Example ports of C++ libraries to Carbon (100% of [woff2](https://github.com/google/woff2), 99% of [RE2](https://github.com/google/re2))

The first part of this result is that all of the woff2 library is ported to
Carbon in a way that exports the same C++ API. There should be no gaps in this
port given that woff2 has a very simple C++ API and uses few C++ language
features.

RE2 is a larger library using significantly more language features. For that
part of the result, fewer than 1% of its C++ lines of code should be missing a
semantically meaningful port into Carbon code.

An important nuance of this goal is that it doesn't include building a complete
Carbon standard library beyond the most basic necessary types. The intent is to
exercise and show the interoperability layers of Carbon by re-using the C++
standard library in many cases and exporting a compatible C++ API to both woff2
and RE2's current API.

While this key result isn't directly tied to the main objective, we believe it
represents a critical milestone for being able to achieve this objective. It
both measures our progress solidifying Carbon's design and demonstrating the
value proposition of Carbon.

Note that both woff2 and RE2 libraries are chosen somewhat arbitrarily and could
easily be replaced with a different, more effective libraries to achieve the
fundamental result of demonstrating a compelling body of cohesive design and the
overarching value proposition.

#### Language design covers the syntax and semantics of the example port code.

We should have a clear understanding of the syntax and semantics used by these
example ports. We should be able to demonstrate that self-contained portions of
the ported code work correctly using Carbon explorer.

### Demo implementation of core features with working examples

A core set of Carbon features should be implemented sufficiently to build
working examples of those features and run them successfully. These features
could include:

-   User-defined types, functions, namespaces, packages, and importing.
-   Basic generic functions and types using interfaces.
-   Initial/simple implementation of safety checking including at least bounds
    checking, simple lifetime checking, and simple initialization checking.
-   Sum types sufficient for optional-types to model nullable pointers.
-   Pattern matching sufficient for basic function overloading on types and
    arity, as well as unwrapping of optional types for guard statements.

Stretch goals if we can hit the above:

-   Instantiating a basic C++ template through interop layer for use within
    Carbon.

The demo implementation should also provide demos outside of specific language
features including:

-   Basic benchmarking of the different phases of compilation (lexing, parsing,
    etc).
-   A basic REPL command line.

Stretch goals if we can hit the above:

-   Automatic code formatter on top of the implementation infrastructure.
-   A [compiler explorer](https://compiler-explorer.com/) fork with REPL
    integrated.

Benchmarking at this stage isn't expected to include extensive optimization.
Instead, it should focus on letting us track large/high-level impact on
different phases as they are developed or features are added. They may also help
illustrate initial high-level performance characteristics of the implementation,
but the long term focus should be on end-to-end user metrics.

Automatic code formatting could be achieved many ways, but it seems useful to
ensure the language and implementation both support use cases like formatting.

### Carbon explorer implementation of core features with test cases

This should include both a human readable rendering of the formal semantics as
well as an execution environment to run test cases through those semantics. The
implementation should cover enough of the core language that example code, such
as the above ports of woff2 and RE2 and the Carbon standard library, can be
verified with Carbon explorer.

## Beyond 2022

Longer term goals are hard to pin down and always subject to change, but we want
to give an idea of what kinds of things are expected at a high level further out
in order to illustrate how the goals and priorities we have in 2022 feed into
subsequent years.

### Potential 2023 goals: finish 0.2 language, stop experimenting

Once Carbon is moving quickly and getting public feedback, we should be able to
conclude the experiment. We should know if this is the right direction for
moving C++ forward for a large enough portion of the industry and community, and
whether the value proposition of this direction outweighs the cost.

However, there will still be a _lot_ of work to make Carbon into a production
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

### Potential 2024-2025 goals: _ship_ 1.0 language & organization

A major milestone will be the first version of a production language. We should
also have finished transferring all governance of Carbon to an independent open
source organization at that point. However, we won't know what a more realistic
or clear schedule for these milestones will be until we get closer.

Another important aspect of our goals in this time frame is expanding them to
encompass the broader ecosystem of the language:

-   End-to-end developer tooling and experience.
-   Teaching and training material.
-   Package management.
-   Etc.
