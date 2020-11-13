# Language-level safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [High-level overview of safety principles](#high-level-overview-of-safety-principles)
-   [Detailed discussion of safety](#detailed-discussion-of-safety)
    -   [What do we mean by "safety"?](#what-do-we-mean-by-safety)
    -   [Guaranteed safety != hardened or mitigated](#guaranteed-safety--hardened-or-mitigated)
    -   [Ideal model for handling safety violations](#ideal-model-for-handling-safety-violations)
    -   [Practical reality for unsafe operations](#practical-reality-for-unsafe-operations)
    -   [Managing bugs without language-guaranteed safety](#managing-bugs-without-language-guaranteed-safety)

<!-- tocstop -->

## High-level overview of safety principles

-   Where there is a choice between safe and unsafe, incentivize the safe option
    by making it equally or more easy to use. If there is a default, it should
    be the safe option. Idiomatic and unremarkable code should be safe. Unsafe
    code should be identifiable.
-   The rules for determining whether code will pass compile time safety
    checking should be articulable, documented, and possible to understand by
    local reasoning.
-   The default development build will diagnose the most common safety
    violations either at compile time or at runtime with high probability, and
    additional modes will cover any other safety violations in the same way.
-   The default optimized build will mitigate any safety violations (that are
    not rejected statically) whenever the performance is below the noise of hot
    path application code.
-   There will be a build option for the optimized build to select a point in
    the performance (speed, binary size, memory size) vs hardening/mitigation
    space for the remaining safety violations.
-   Developers need a strong testing methodology to engineer correct software.
    We will encourage this and then leverage it with the checking build modes to
    find and fix bugs and vulnerabilities.
-   The language makes design choices that allow more efficient implementations
    of the hardening/testing build modes and better automation of
    testing/fuzzing.

Taken together, these principles imply that the language must use both compile
time and runtime checks for safety. It should use simple rules and programmer
annotations to prove safety at compile time when possible (e.g. a subset of what
can be proved safe using Rust's lifetime-parameters and borrowing rules). And it
should provide build modes which check the remaining safety, with high
probability, at runtime. Over time, with more experience, the set of things
checked at compile time can incrementally increase and, if desired, arrive at
similar levels of static safety as a language like Rust.

## Detailed discussion of safety

### What do we mean by "safety"?

We define the term safety in the context of programming languages as protection
from software bugs, whether required by the language or merely an implementation
option. Specifically, certain operations that are known components of security
vulnerabilities are entirely precluded, even in the face of an incorrect program
that violates language rules. These operations are called safety violations.
Based on the nature of the prevented safety violation, safety is often
decomposed into the categories of memory, type, and data race safety.

-   Memory safety protects against invalid memory accesses, both spatially when
    accessing out of bounds and temporally where the access occurs outside the
    lifetime of the intended object.
-   Type safety protects against accessing objects with an incorrect type. These
    issues are sometimes called "type confusion".
-   Data race safety protects against racing (concurrent and unsynchronized)
    memory accesses (reads and writes).

### Guaranteed safety != hardened or mitigated

We need to distinguish between giving a safety guarantee from hardening against
or mitigating a vulnerability. A safety guarantee is an especially strong
requirement: we must enforce those properties in a clearly defined way even for
programs that contain bugs due to violating the language rules. As a
consequence, the behavior intended to catch bugs remains observable by
intentional code.

Hardening and mitigations are importantly different from safety guarantees. They
address the feasibility of exploiting a vulnerability, but may not provide the
strict guarantee in order to provide better performance in some dimension or
better scaling. Let's consider an example:
[memory tagging](https://llvm.org/devmtg/2018-10/slides/Serebryany-Stepanov-Tsyrklevich-Memory-Tagging-Slides-LLVM-2018.pdf)
for details). It provides a very strong mitigation of memory unsafety by making
each attempt at an invalid read or write have a high probability of trapping but
not be guaranteed to trap in every case. Because realistic attacks require many
such operations, this mitigation can make the attack itself infeasible.
Alternatively, the trap might be asynchronous, leaving only a tiny window of
time prior to the attack being detected and program terminated. Both of these
mitigation strategies are sufficiently strong (the difficulty of the attack
becomes dramatically higher and potentially infeasible) but neither provides any
clear or strong semantic guarantee that would lead us to claim it provides
memory safety.

Regardless of the semantic guarantee, in a development build we will produce
precise and easy to understand diagnostics of any safety violations, enabling
programmers to effectively understand and fix the bugs in their code leading to
that violation.

### Ideal model for handling safety violations

Our goal for every safety violation is:

-   In the development build mode, it is checked and diagnosed with a detailed
    report of the error, statically or dynamically with high probability per
    single occurrence of the bug.
-   In the optimized release build mode, it is hardened or mitigated from any
    exploit by an attacker.

While it is convenient to use the same strategy in both build modes, this isn't
even the ideal we strive toward because many diagnostic techniques may be
ineffective as hardening techniques.

### Practical reality for unsafe operations

The ideal model should be both our goal and our starting point, but we expect to
make at least two primary concessions for practical reasons: the development
build may need to be augmented by specialized checking build modes to cover all
safety violations, and the optimized build may not be able to mitigate all
safety violations due to performance overhead. Unfortunately, the current
best-in-class techniques for preventing data races at compile time cannot handle
any "unsafe" code (including existing C and C++ code) and still have unanswered
questions around ease of adoption and compilation scalability. As a consequence,
some safety violations must be detected at runtime.

Currently, detecting data races at runtime is also too expensive to do in the
development build mode. As a consequence, we expect responsibility for
exhaustive checking for safety violations to be split among a small number of
additional build modes that diagnose different kinds of safety violations at
runtime. These should be kept to the smallest number that permits all such modes
to have acceptable overhead.

The optimized build mode should harden against any safety violations by default,
but current hardening techniques are still too expensive to enable in every case
due to memory or performance overhead. Our criteria for the overhead being
acceptable is that it falls below the noise on the hot path of application code.
This is a very high bar, but we think it is critical to hold that high bar so
that we can migrate existing C++ systems that currently expect that degree of
performance.

Beyond these two concessions, we think it is important to allow users of the
language to adjust the balance between hardening and the various axes of
performance, at a reasonably coarse level. Different users and different
applications will have different tolerances for such performance overhead, and
we should provide as much flexibility here as we reasonably can. Specific
examples here include
[Control Flow Integrity](https://en.wikipedia.org/wiki/Control-flow_integrity)
as [implemented in Clang](http://clang.llvm.org/docs/ControlFlowIntegrity.html),
trapping overflow, and automatic bounds checking, probabilistic use-after-free
and race detection.

### Managing bugs without language-guaranteed safety

We can offer users of a language security mitigations to manage their risk, but
we still need a way for developers to reliably find and fix the bugs that will
inevitably be written. We believe that the cornerstone of this is the use of
strong testing methodologies. This doesn't just mean good test coverage. It
means the combination of:

-   Ensuring unsafe or risky operations and interfaces can easily be recognized
    by humans.
-   Static analysis tooling to detect common bugs integrated into the build
    and/or code review developer workflow. Think of these as static testing of
    code.
-   Good test coverage, including unit, integration, and system tests.
-   Continuous integration, including automatic and continuous running of these
    tests, in the development, checked build mode as well as any additional
    build modes necessary to cover different forms of behavior checking.
-   Coverage-directed fuzz testing in combination with checking build modes to
    discover bugs outside of human-authored test coverage, especially for
    interfaces handling untrusted data.
-   Language features that make automated testing and fuzzing easier. For
    example, if the language encourages value types and pure functions of some
    sort, they can be automatically fuzzed.

These practices are necessary for reliable, large-scale software engineering.
Maintaining correctness of business logic over time requires continuous and
thorough testing. Without it, such software systems cannot be changed and
evolved over time reliably. We can then re-use these practices in conjunction
with checking build modes to mitigate the absence of language-guaranteed safety
without imposing overhead on production systems.

When the practical realities outlined previously preclude language-guaranteed
safety, we believe adhering to this kind of testing methodology is essential. As
a consequence, the language ecosystem (from the language itself to the libraries
and tooling around it) need to directly work to remove barriers and encourage
the development of these methodologies.

On the other hand, if the practical realities of a domain preclude this degree
of testing rigor, we suggest that it becomes imperative to accept the overhead
of a language that gives true safety guarantees.
