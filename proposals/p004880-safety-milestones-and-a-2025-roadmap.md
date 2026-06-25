# Safety milestones and a 2025 roadmap

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4880)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Proposal](#proposal)
-   [Retrospective on 2024](#retrospective-on-2024)

<!-- tocstop -->

## Abstract

We propose updating our milestones to accelerate design and implementation of
memory safety in Carbon, and a roadmap for 2025 reflecting this change. We also
provide a retrospective for 2024's progress.

## Proposal

Recently, we have seen several new contributors join the Carbon project and a
corresponding up-tick in activity across the project. We are all really excited
both by the new folks participating and seeing things move faster.

However, we have also had some good conversations with potential users and other
interested parties about the project and our roadmap. Everyone has been very
happy to see the rapid progress on a realistic toolchain, but we have also
gotten even stronger feedback than previously around memory safety, and we
propose updating our plans to better address that feedback.

For the past two years we have been working on the C++-interop focused aspect of
Carbon, and deferring the work to build out a strong memory safety story. While
there was strong interest in memory safety, and for many an essential
requirement long term, it seemed reasonable to tackle first interop, and then
look at safety. This was heavily informed both by having a very small set of
contributors and a desire to ship an 0.1 milestone with C++ interop as quickly
as possible.

Now we both have a larger set of contributors, and even stronger and more
specific feedback asking for a concrete and detailed design for memory safety in
Carbon sooner rather than later. We propose as a consequence to add a concrete
_design_ for memory safe Carbon to our 0.1 milestone, and begin working on this
in parallel to the toolchain in 2025.

Our projected timeline for reaching 0.1 was historically based on growing the
set of contributors while keeping all of them focused on the smaller prior 0.1
milestone. Adding a memory safety design to our target for 0.1 will make it
impossible to ship in 2025 even with the larger contributor base. We are
shifting our target from 2025 for 0.1 to, at the soonest, the end of 2026; and
as always with these projections, they should be understood as a lower bound.

With these updated milestones, we propose the following goals for 2025:

-   Complete design and implementation for remaining Carbon features needed for
    non-template C++ interop
-   Complete most of the interop layer for non-template Carbon ↔ C++,
    prioritizing calling/using C++ APIs from Carbon, and then exposing Carbon
    APIs back to C++.
-   Full support for compiling C++ code with the Carbon toolchain using Clang
-   Update our safety strategy, and establish a detailed strategy for memory
    safety
-   Design for compile-time & type-system based temporal and mutation safety

Both the C++ interop and the memory safety designs are expected to have
dependencies that we will need to prioritize in 2025. The C++ interop will
depend on finishing the implementation of many parts of the Carbon language in
our toolchain, and the safety designs will build on top of the existing Carbon
type-system design, including refinements to it from our implementation
experience.

As with previous years, we will also continually share our progress and what we
learn with the broader open source and C++ communities.

## Retrospective on 2024

Our [roadmap for 2024] was an ambitious pivot to focus heavily on implementing
the Carbon language in a realistic compiler and toolchain. At a high level, this
pivot was _very_ successful. Carbon's compiler has made massive strides over the
year, and all of the active contributors to Carbon have thoroughly ramped up on
our implementation and are making meaningful contributions to it.

[roadmap for 2024]:
    https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md

Going into 2024, Carbon's compiler didn't have any support for generics,
importing, expression categories, debug info, mangling, a prelude, or so many
other things now in flight. Constant evaluation didn't know about aggregates,
and classes couldn't be initialized at all. We had no examples, much less
working ones. By the end of 2024, with a surprisingly minimal number of hacks,
it was possible to solve much of Advent of Code! The progress here has been
phenomenal, and we made awesome strides towards our goal of a working toolchain.

Looking in more detail at our key results for 2024:

-   [Carbon's toolchain implements enough of the language to build realistic code](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#carbons-toolchain-implements-enough-of-the-language-to-build-realistic-code)
    -   A smashing success, overall.
    -   We have imports, a working prelude, generic types and functions.
    -   We have many of the building blocks of dispatching through interfaces
        for operator overloading, but there are still a few gaps left. Despite
        the gaps, we're actually using our generic building blocks effectively,
        including with a working `Core.Int` generic integer type that backs
        `i32` instead of a hard coded type!
    -   We didn't get to the more stretch parts of this goal like templates
        though.
-   [Carbon's toolchain can build C++ code](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#carbons-toolchain-can-build-c-code)
    -   We have Clang integrated into the toolchain!
    -   It works reasonably well for C++ code without system `#include`s.
    -   But we still need to get some critical headers and other data that Clang
        depends on to get this fully working with system `#include`s.
-   [Carbon's toolchain works with existing, simple C++ build systems](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#carbons-toolchain-works-with-existing-simple-c-build-systems)
    -   Until we have all the system `#include` support work done, we can't drop
        our toolchain into a C++ build system. =/ This one ended up largely not
        landing this year.
    -   We did end up integrating Carbon's toolchain into very simple Bazel
        build rules that we use to continuously build and test a collection of
        example Carbon code. This made sure the compilation model does work in a
        realistic toolchain situation for Carbon code.
-   [Carbon has a design and toolchain implementation of basic C++ interop](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#carbon-has-a-design-and-toolchain-implementation-of-basic-c-interop)
    -   This was too ambitious of a goal, but we actually got very close. The
        first importing of C++ headers (but without doing anything) has already
        landed, so we weren't too far away from the most basic parts of this.
-   [Give talks at 2-3 conferences covering 3-4 different Carbon topics](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#give-talks-at-2-3-conferences-covering-3-4-different-carbon-topics)
    -   We had a really strong year delivering talks at both C++ and LLVM
        conferences: 5 talks on different topics plus a panel session.
-   [Start building our initial tutorial and introductory material](https://github.com/carbon-language/carbon-lang/blob/10189bbb78db7b143a6d9d62797fc9698363fe4d/docs/project/roadmap.md#start-building-our-initial-tutorial-and-introductory-material)
    -   We got an important start here with newsletters every other month,
        packaged pre-releases, and even some "first look" courses introducing
        people to Carbon.
    -   However, tutorial material still seems somewhat far away.

Overall, we achieved the majority of what we set out to for 2024. While our
roadmaps are always a bit ambitious to push us, this year didn't seem excessive
and is likely to reflect roughly how ambitious we want to be in our roadmaps.
