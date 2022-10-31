# Difficulties improving C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

C++ is the dominant programming language for the performance critical software
our goals prioritize. The most direct way to deliver a modern and excellent
developer experience for those use cases and developers would be to improve C++.

Improving C++ to deliver the kind of experience developers expect from a
programming language today is difficult in part because **C++ has decades of
technical debt** accumulated in the design of the language. It inherited the
legacy of C, including
[textual preprocessing and inclusion](https://clang.llvm.org/docs/Modules.html#problems-with-the-current-model).
At the time, this was essential to C++'s success by giving it instant and high
quality access to a large C ecosystem. However, over time this has resulted in
significant technical debt ranging from
[integer promotion rules](https://shafik.github.io/c++/2021/12/30/usual_arithmetic_confusions.html)
to complex syntax with
"[the most vexing parse](https://en.wikipedia.org/wiki/Most_vexing_parse)".

**C++ has also prioritized backwards compatibility** including both syntax and
[ABI](https://en.wikipedia.org/wiki/Application_binary_interface). This is
heavily motivated by preserving its access to existing C and C++ ecosystems, and
forms one of the foundations of common Linux package management approaches. A
consequence is that rather than changing or replacing language designs to
simplify and improve the language, features have overwhelmingly been added over
time. This both creates technical debt due to complicated feature interaction,
and fails to benefit from on cleanup opportunities in the form of replacing or
removing legacy features.

Carbon is exploring significant backwards incompatible changes. It doesn't
inherit the legacy of C or C++ directly, and instead is starting with solid
foundations, like a modern generics system, modular code organization, and
consistent, simple syntax. Then, it builds a simplified and improved language
around those foundational components that remains both interoperable with and
migratable from C++, while giving up transparent backwards compatibility. This
is fundamentally **a successor language approach**, rather than an attempt to
incrementally evolve C++ to achieve these improvements.

Another challenge to improving C++ in these ways is the current evolution
process and direction. A key example of this is the committee's struggle to
converge on a clear set of high-level and long-term goals and priorities aligned
with [ours](https://wg21.link/p2137). When [pushed](https://wg21.link/p1863) to
address
[the technical debt caused by not breaking the ABI](https://wg21.link/p2028),
**C++'s process
[did not reach any definitive conclusion](https://cor3ntin.github.io/posts/abi/#abi-discussions-in-prague)**.
This both failed to meaningfully change C++'s direction and priorities towards
improvements rather than backwards compatibility, and demonstrates how the
process can fail to make directional decisions.

Beyond C++'s evolution direction, the mechanics of the process also make
improving C++ difficult. **C++'s process is oriented around standardization
rather than design**: it uses a multiyear waterfall committee process. Access to
the committee and standard is restricted and expensive, attendance is necessary
to have a voice, and decisions are made by live votes of those present. The
committee structure is designed to ensure representation of nations and
companies, rather than building an inclusive and welcoming team and community of
experts and people actively contributing to the language.

Carbon has a more accessible and efficient [evolution process](evolution.md)
built on open-source principles, processes, and tools. Throughout the project,
we explicitly and clearly lay out our [goals and priorities](goals.md) and how
those directly shape our decisions. We also have a clear
[governance structure](evolution.md#governance-structure) that can make
decisions rapidly when needed. The open-source model enables the Carbon project
to expand its scope beyond just the language. We will build a holistic
collection of tools that provide a rich developer experience, ranging from the
compiler and standard library to IDE tools and more. **We will even try to close
a huge gap in the C++ ecosystem with a built-in package manager.**

Carbon is particularly focused on a specific set of [goals](goals.md). These
will not align with every user of C++, but have significant interest across a
wide range of users that are capable and motivated to evolve and modernize their
codebase. Given the difficulties posed by C++'s technical debt, sustained
priority of backwards compatibility, and evolution process, we wanted to explore
an alternative approach to achieve these goals -- through a
backwards-incompatible successor language, designed with robust support for
interoperability with and migration from C++. We hope other efforts to
incrementally improve C++ continue, and would love to share ideas where we can.
