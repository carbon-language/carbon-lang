# Carbon language

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The Carbon Language project is an **_experiment_** to explore a possible,
distant future for the C++ programming language. It is designed around a
specific set of goals, priorities, and use cases:

1.  Performance-critical software
2.  Both software and language evolution
3.  Code that is easy to read, understand, and write
4.  Practical safety guarantees and testing mechanisms
5.  Fast and scalable development
6.  Current hardware architectures, OS platforms, and environments as they
    evolve
7.  Interoperability with and migration from existing C++ code

The first six of these represent a set of priorities for C++ shared by a
significant subset of the C++ community, industry, and ecosystem. However, C++
is increasingly constrained by a diverse set of concerns and priorities
(including some that are irrelevant to or in opposition to these goals, such as
ABI stability), and carries a significant historical legacy that makes it
challenging to evolve effectively. The result is that these users struggle to
meet our goals using C++ today, and that is unlikely to change in the near
future. Carbon is an attempt to explore what it would look like to rapidly and
systematically re-engineer C++ into a near optimal future state along the top
six priorities, which nonetheless is still reachable through interoperability,
tooling, automation, and incremental large-scale migration efforts.

For more information, see [our goals document](docs/project/goals.md).

## What about other languages?

Other programming languages don't _currently_ address these needs effectively.
They present interoperability, migration, and performance challenges that make
it expensive and potentially impossible to migrate a large C++ code base. An
approach which requires rewriting an entire binary at once would be infeasible.
A large-scale migration must be incremental, meaning that interoperability and
tool-assisted code rewrites are critical.

There are projects for several languages to reduce obstacles affecting migration
from C++. Some contributors to Carbon are also contributing to those efforts in
parallel in order to understand all of the options in this space. TODO: write up
a detailed analysis of these languages specifically through the lens of the
above goals.

One especially interesting aspect not addressed by the active and widely used
languages that might serve this purpose is that they have not been designed
specifically to enable migration from and interoperability with _today's_ C++.
They don't build on top of C++'s _existing_ ecosystem. There are only a few
significant examples of programming languages that center around incremental
migration of large existing codebases. They are specifically designed to not
require complete rewrites, new programming models, or building an entire new
stack/ecosystem. However, there is no comparable option for C++ today:

- JavaScript → TypeScript
- Java → Kotlin
- C++ → **???**

Carbon explores what it would look like to fill this gap and align it with the
above priorities.

## Project status

**The project is just getting started.** Everything is at a very early stage. If
you are hoping to see lots of concrete ideas and plans, you'll probably want to
check back in 6 months to a year. At this stage, we're just beginning to lay the
foundations.

It is important to understand that **this is a science experiment**, not a
production effort. There are several initial questions that we want to explore
and answer with this experiment:

- Can we deliver a design and implementation that is familiar and compelling to
  C++ programmers and supports our goals?
- How seamless and effective can we make interoperability?
- How easy and scalable can we make migration?
- Will a significant segment of the ecosystem and industry adopt Carbon given
  these tradeoffs?

We are committed to learning the answers to these questions, but that may well
not result in a production language. There is a very real chance that this
project will never leave the experimental phase. Anyone considering contributing
or using Carbon should be extremely mindful of that fact: **core contributors
may abandon the experiment**.

While we may sometimes refer to Carbon as a language, it is crucial to
understand that the goals of this science experiment are not about new
languages, but about how to move today's C++ forward effectively. For example, a
near optimal outcome would be to convince the C++ community to adopt this as its
official path forward.

## What will make Carbon a compelling future path for C++?

We hope that eventually Carbon will provide **significant advantages compared to
today's C++**. Areas where we think we can most dramatically improve C++ for
both software systems and developers are:

- A cohesive and principled language design, even when supporting advanced
  features.
- Making common coding patterns safe by default whenever practical, with
  affordable security mitigations available for any unsafety.
  - We will provide static checks for as many safety issues as we can by
    default.
  - We will provide a spectrum of build modes with different trade-offs between
    dynamic safety and performance. For example:
    - The default build mode will include as many dynamic safety checks as we
      can while keeping the software's performance reasonable for normal
      development, testing, and debugging.
    - Release builds will favor performance, with opt-in dynamic safety checks
      and security mitigations for applications with higher security
      requirements.
  - Over time, we also expect to both track and drive research into increasing
    the degree of safety available without compromising our other goals.
- Keeping our core language implementation simple, fast, and easily extended in
  ways that will make all of our language tools better.
- Providing an effective, open, and inclusive language evolution process aligned
  with our goals and priorities.

Carbon will also aim to allow a single layer of a legacy C++ library stack to be
migrated to Carbon, without migrating the code above or below. This will make it
easier for developers to start using Carbon. Key features underpin Carbon's
compatibility and interoperability with C++:

- The memory, execution, and threading model will be compatible with C++.
- Access to existing C++ types, interfaces, and even templates will be provided
  as part of the core language.
- Carbon will be able to export types, interfaces, and templates for consumption
  by C++.

**However, Carbon's approach still requires a nearly complete re-engineering of
the language as well as large-scale migration for users.** This is extremely
expensive, and so the bar for Carbon to be a compelling direction for C++ is
very high.

## Repository structure overview

Carbon's main repositories are:

- **carbon-lang** - Carbon language specification and documentation.
- **carbon-toolchain** - Carbon language toolchain and reference implementation.
- **carbon-proposals** - An archive of reviewed Carbon language proposals.
