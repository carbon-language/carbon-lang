# Safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Details](#details)
    -   [What does "safety" mean in Carbon?](#what-does-safety-mean-in-carbon)
    -   [Guaranteed safety is different from hardening and mitigation](#guaranteed-safety-is-different-from-hardening-and-mitigation)
    -   [Ideal model for handling safety violations](#ideal-model-for-handling-safety-violations)
    -   [Practical reality for unsafe operations](#practical-reality-for-unsafe-operations)
    -   [Managing bugs without guaranteed safety](#managing-bugs-without-guaranteed-safety)
-   [Alternatives considered](#alternatives-considered)
    -   [Full compile-time enforced safety by default (Rust's model)](#full-compile-time-enforced-safety-by-default-rusts-model)
    -   [Rust's model but without preventing data races](#rusts-model-but-without-preventing-data-races)
    -   [Dynamic lifetime safety and compile-time enforced safety otherwise (Swift's model)](#dynamic-lifetime-safety-and-compile-time-enforced-safety-otherwise-swifts-model)
    -   [Dynamic lifetime safety and defined behavior (Java's model)](#dynamic-lifetime-safety-and-defined-behavior-javas-model)

<!-- tocstop -->

## Overview

Carbon's safety strategy is based on the goal to provide
[practical safety guarantees and testing mechanisms](../goals.md#practical-safety-guarantees-and-testing-mechanisms),
and its priority relative to other goals, particularly performance and
usability. Carbon's safety strategy will be built on several key features:

-   Where there is a choice between safe and unsafe, the safe option should be
    incentivized by making it equally or more easy to use. If there is a
    default, it should be the safe option. Idiomatic and unremarkable code
    should be safe. Unsafe code should be identifiable.

-   The rules for determining whether code will pass compile time safety
    checking should be articulable, documented, and possible to understand by
    local reasoning.

-   The default development build will diagnose the most common safety
    violations either at compile time or at runtime with high probability, and
    additional modes will cover any other safety violations in the same way.

-   The default optimized build will provide runtime mitigations for safety
    violations whenever the performance is below the noise of hot path
    application code.

-   There will be a build option for the optimized build to select a trade-off
    between performance (including speed, binary size, and memory size) and
    hardening or mitigation for the remaining safety violations.

-   Developers need a strong testing methodology to engineer correct software.
    Carbon will encourage testing and then leverage it with the checking build
    modes to find and fix bugs and vulnerabilities.

-   Language design choices should allow more efficient implementations of the
    hardening and testing build modes. They should also allow better automation
    of testing and fuzzing.

Taken together, these principles imply that the language must use both compile
time and runtime checks for safety. It should use simple rules and code
annotations to prove safety at compile time when possible; for example, a subset
of what can be proved safe using Rust's lifetime-parameters and borrowing rules.
It should also provide build modes which check the remaining safety at runtime
with high probability. Over time and with more experience, the set of things
checked at compile time can incrementally increase and, if desired, arrive at
similar levels of static safety as a language like Rust.

## Details

### What does "safety" mean in Carbon?

In Carbon, safety is protection from software bugs, whether the protection is
required by the language or merely an implementation option. Safety can be
decomposed into categories of memory, type, and data race safety, based on the
related security vulnerability:

-   **Memory safety** protects against invalid memory accesses.

    -   _Spatial_ memory safety protects against accessing out of bounds, such
        as for an array.

    -   _Temporal_ memory safety protects against access to memory outside the
        lifetime of the intended object.

-   **Type safety** protects against accessing objects with an incorrect type,
    also known as "type confusion".

-   **Data race safety** protects against racing memory accesses, whether
    concurrent or unsynchronized.

### Guaranteed safety is different from hardening and mitigation

A safety guarantee must be distinguished from hardening against or mitigating a
vulnerability. A safety guarantee is an especially strong requirement: the
properties must be enforced in a clearly defined way even for programs that
contain bugs due to violating the language rules. As a consequence, the behavior
intended to catch bugs remains observable by intentional code.

Hardening and mitigations are importantly different from safety guarantees. They
address the feasibility of exploiting a vulnerability, but may not provide the
strict guarantee in order to provide better performance in some dimension or
better scaling.

For example,
[memory tagging](https://llvm.org/devmtg/2018-10/slides/Serebryany-Stepanov-Tsyrklevich-Memory-Tagging-Slides-LLVM-2018.pdf)
provides a very strong mitigation for memory safety violations by making each
attempt at an invalid read or write have a high probability of trapping, while
still not guaranteeing to trap in every case. Realistic attacks require many
such operations, so memory tagging will probably stop attacks. Alternatively,
the trap might be asynchronous, leaving only a tiny window of time prior to the
attack being detected and program terminated. Both of these mitigation
strategies reduce the feasibility attacks, but neither provides any clear or
strong semantic memory safety guarantee.

Regardless of whether a safety violation is found through guaranteed safety,
hardening, or mitigation, development builds will provide precise and easy to
understand diagnostics of discovered safety violations, enabling programmers to
effectively understand and fix the bugs in their code leading to that violation.

### Ideal model for handling safety violations

As many safety violations as possible should be statically checked. However,
developers should expect that at least some safety violations will only be
detected dynamically, where behavior relies on separate build modes:

-   In the _development_ build mode, dynamic checks detect safety violations
    with a high probability per single occurrence of the bug. Detected
    violations are accompanied by a detailed diagnostic report to ease developer
    bug-finding.

-   In the _optimized release_ build mode, safety violations are hardened or
    mitigated from any exploit by an attacker.

While it might appear convenient to use the same strategy in both build modes,
divergence may occur where particular diagnostic techniques are ineffective as
hardening techniques or have substantial performance overhead.

### Practical reality for unsafe operations

Carbon will likely make two concessions to the ideal model for practical
reasons:

-   The development build may need to be augmented by additional, specialized
    checking build modes to cover all safety violations. Currently, detecting
    data races at runtime is too expensive to do in a single, comprehensively
    checked development build mode. Each specialized build development build
    mode will be less expensive by splitting different dynamic safety violation
    checks to different build modes. These should be kept to the smallest number
    that permits all such modes to have acceptable overhead.

-   The optimized build may not be able to harden or mitigate all safety
    violations. Although it should harden against any safety violations by
    default, current hardening techniques are still too expensive to enable in
    every case due to memory or performance overhead. Carbon's criteria for
    acceptable overhead is that it falls below the noise on the hot path of
    application code. This is a very high bar, but will likely be critical for
    allowing migration of existing C++ systems which require high performance.

Beyond these two concessions, it's important to allow users of the language to
adjust the balance between hardening and the various axes of performance, at
least at a coarse level. Different users and different applications will have
different tolerances for such performance overhead, and Carbon should provide
flexibility where feasible. Specific examples here include
[Control Flow Integrity](https://en.wikipedia.org/wiki/Control-flow_integrity)
as [implemented in Clang](http://clang.llvm.org/docs/ControlFlowIntegrity.html),
trapping overflow, bounds checking, and probabilistic use-after-free and race
detection.

### Managing bugs without guaranteed safety

Carbon can offer safety mitigations to manage their security risk, but it still
needs to provide developers a way to reliably find and fix the inevitable bugs.
The cornerstone of this is the application of strong testing methodologies.
Strong testing is more than good test coverage; it means the combination of:

-   Ensuring unsafe or risky operations and interfaces can easily be recognized
    by humans.

-   Static analysis tooling to detect common bugs integrated into the build and
    code review developer workflows. These could be viewed as static testing of
    code.

-   Good test coverage, including unit, integration, and system tests.

-   Continuous integration, including automatic and continuous running of these
    tests. The checked development build mode should be validated, as well as
    any additional build modes necessary to cover different forms of behavior
    checking.

-   Coverage-directed fuzz testing in combination with checking build modes to
    discover bugs outside of human-authored test coverage, especially for
    interfaces handling untrusted data.

-   Language features that make automated testing and fuzzing easier. For
    example, if the language encourages value types and pure functions of some
    sort, they can be automatically fuzzed.

These practices are necessary for reliable, large-scale software engineering.
Maintaining correctness of business logic over time requires continuous and
thorough testing. Without it, such software systems cannot be changed and
evolved over time reliably. Carbon will re-use these practices in conjunction
with checking build modes to mitigate the limitations of Carbon's safety
guarantees without imposing overhead on production systems.

Where the practical realities outlined previously preclude guaranteed safety,
adhering to this kind of testing methodology is essential. As a consequence,
Carbon's ecosystem, including the language, tools, and libraries, will need to
directly work to remove barriers and encourage the development of these
methodologies.

On the other hand, if the practical realities of a domain preclude this degree
of testing rigor, developers should evaluate whether a language with more
comprehensive safety guarantees would be better suited.

## Alternatives considered

### Full compile-time enforced safety by default (Rust's model)

We could commit to the same level of compile-time enforced safety as Rust does.
This still allows for `unsafe` blocks as in Rust, and library types that provide
_dynamic_ safety enforcement but are implemented with some `unsafe` internals.
Note that our intent is to leave the door _open_ to this model despite not
pursuing it initially.

Pros:

-   Full safety (even against data races) provided at compile time.
-   Early evidence shows _significant_ impact in reducing bugs generally.
-   Unsafe operations can effectively be treated as bugs rather than becoming
    "valid", but surprising and often unintended, behaviors.
-   Can likely leverage the huge work of the Rust community to understand what
    is necessary.
-   Careful use of narrow `unsafe` escape hatches can be effectively
    encapsulated behind otherwise safe APIs.

Cons:

-   Effectively requires widespread different design patterns and idioms from
    C++.
    -   Designing data structures to avoid sharing mutable state.
    -   Fully modeling lifetime and exclusivity in the type system.
    -   Increased complexity of node/pointer based data structures like linked
        lists.
-   Many techniques currently used by Rust have unsolved compilation scaling
    challenges due to interprocedural inference and increases in the set of
    types with generic code.
-   Complexity of type system proofs of safety may incentivize unnecessary
    dynamic checking of safety properties. For example, an unnecessary `RefCell`
    or `Rc`.
-   Some of the most essential dynamic safety tools that ease the ergonomic
    burden of the Rust-style lifetime model (`Rc`) introduce _semantic_
    differences that cannot then be eliminated in a context where performance is
    the dominant priority.
-   Remains an open (if fairly difficult) evolution path for the language even
    if we don't eagerly pursue.

### Rust's model but without preventing data races

We could replicate the Rust model as above but additionally not trying to
statically preclude data races.

Pros:

-   Outside of data races provides same benefits as above.

Cons:

-   Unclear that removing the data race prevention aspect (law of exclusivity)
    is sufficient to meaningfully address the cons above. Most of the lifetime
    complexity remains unchanged.
    -   Would no longer need `RefCell`, but would still need `Rc` and `ARc`.

### Dynamic lifetime safety and compile-time enforced safety otherwise (Swift's model)

Swift defaults to dynamically enforced lifetime safety and no data race
prevention, and only requires compile time enforcement of the remaining spatial
safety properties. This _does_ remove the majority of the type system complexity
needed to support the safety in Rust's model.

Pros:

-   Significantly simpler model than Rust's.
-   Safe for all of the most common and important classes of memory safety bugs.

Cons:

-   Results in significant design differences as the distinction between value
    types and "class types" (that are held by a reference counted pointer and
    thus lifetime safe) becomes extremely important.
-   Reference counting based safety introduces significant performance costs
    with very difficult tools for controlling them. This puts the option in
    tension with the performance goal of Carbon.
-   Garbage collection based safety has less direct performance overhead but has
    a greater unpredictability of performance.

### Dynamic lifetime safety and defined behavior (Java's model)

Another approach to safety is to largely provide defined and predictable
behavior for all potential safety violations, which is what Java does (at the
highest level). This, combined with some form of dynamic lifetime management,
otherwise known as garbage collection, forms the basis of Java's safety.

Pros:

-   Among the most robust and well studied models with decades of practical
    usage and analysis for security properties.
-   Extremely suitable for efficient implementation on top of a virtual machine
    like the JVM.

Cons:

-   Extremely high complexity to fully understand the implications of complex
    cases like data races.
-   Tends to require _significant_ performance overhead without the aid of a
    very powerful VM-based execution environment with highly dynamic
    optimizations.
    -   The complexity of the implementation in turn creates difficult to
        _predict_ performance.
