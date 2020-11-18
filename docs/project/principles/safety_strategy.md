# Safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [What does "safety" mean in Carbon?](#what-does-safety-mean-in-carbon)
    -   [Guaranteed safety is different from hardening and mitigation](#guaranteed-safety-is-different-from-hardening-and-mitigation)
    -   [Model for handling safety violations](#model-for-handling-safety-violations)
    -   [Managing bugs without guaranteed safety](#managing-bugs-without-guaranteed-safety)
-   [Alternatives considered](#alternatives-considered)
    -   [Guaranteed safety by default (Rust's model)](#guaranteed-safety-by-default-rusts-model)
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

### Model for handling safety violations

Developers should expect that safety violations will largely be detected
dynamically, where behavior relies on separate build modes:

-   In the _development_ build mode, dynamic checks detect safety violations
    with a high probability per single occurrence of the bug. Detected
    violations are accompanied by a detailed diagnostic report to ease developer
    bug-finding.

    -   The main development build mode is expected to be augmented by multiple,
        specialized build modes. Dynamic detection of safety violations remains
        expensive, and splitting up checks into different build modes should
        make it more manageable to test code for more types of safety
        violations. The goal will to keep this number as small as possible while
        maintaining acceptable overhead.

-   In the _optimized release_ build mode, safety violations are hardened or
    mitigated from any exploit by an attacker.

    -   Only hardening techniques which don't measurably impact application hot
        path performance will be enabled by default. This is a very high bar,
        but will likely be critical for allowing migration of existing C++
        systems.

    -   Developers need to be able to choose where they want the balance to be
        between hardening and the various axes of performance. Different
        applications will have different tolerances for performance overhead.
        Specific examples here include
        [Control Flow Integrity](https://en.wikipedia.org/wiki/Control-flow_integrity)
        as
        [implemented in Clang](http://clang.llvm.org/docs/ControlFlowIntegrity.html),
        trapping overflow, bounds checking, and probabilistic use-after-free and
        race detection.

While it might appear convenient to use the same strategy in both build modes,
divergence may occur where particular diagnostic techniques are ineffective as
hardening techniques or have substantial performance overhead.

Carbon's design will still endeavor to provide static detection of safety
violations. However, dynamic detection will remain a priority to ease the
transition of C++ developers and codebases to Carbon. It's important that Carbon
provides improved safety for migrated codebases without creating substantial
costs for migration.

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

### Guaranteed safety by default (Rust's model)

Carbon could provide guaranteed safety by default, with compile-time validation
equivalent to Rust. This approach still allows for
[`unsafe` blocks](https://doc.rust-lang.org/rust-by-example/unsafe.html) and
library types that provide _dynamic_ safety enforcement while using some
`unsafe` internals.

Advantages:

-   Guaranteed safety, including against data races, is provided at compile
    time.
-   With Rust, there is early evidence that there's a significant impact in
    reducing bugs generally.
-   Could build upon the huge work of the Rust community, reducing the risks of
    implementing similar in Carbon.
-   Careful use of narrow `unsafe` escape hatches can be effectively
    encapsulated behind otherwise safe APIs.

Disadvantages:

-   Creates widespread divergence from the design patterns and idioms of C++.
    -   Data structures must be redesigned to avoid sharing mutable state.
    -   Requires fully modeling lifetime and exclusivity in the type system.
    -   Increases complexity of node and pointer based data structures, such as
        linked lists.
-   Many techniques currently used by Rust have unsolved compilation scaling
    challenges due to interprocedural inference and increases in the set of
    types with generic code.
-   Complexity of type system proofs of safety may incentivize unnecessary
    dynamic checking of safety properties. For example, an unnecessary
    [`RefCell`](https://doc.rust-lang.org/std/cell/struct.RefCell.html) or
    [`Rc`](https://doc.rust-lang.org/std/rc/struct.Rc.html).
-   Some of the most essential dynamic safety tools that ease the ergonomic
    burden of the Rust-style lifetime model (`Rc`) introduce _semantic_
    differences that cannot then be eliminated in a context where performance is
    the dominant priority.

While guaranteed safety is desirable, and Carbon should support it as much as
possible, Carbon's priorities include migration of large C++ codebases. These
codebases and developers are crucial to Carbon's success, and as such is
considered infeasible to require in Carbon at this time. While a migration tool
could in theory mark all migrated code as `unsafe`, that is considered to
undermine the spirit of guaranteed safety by default.

That does not mean Carbon will never adopt guaranteed safety by default, only
that it will not be an initial goal of Carbon. It should still be possible to
adopt later, although at higher cost than if Carbon start with it.

### Rust's model but without preventing data races

Carbon could replicate most of the Rust model as previously described, only
excluding safety guarantees for data races.

Advantages:

-   Provides the same benefits as the Rust model, excluding safety guarantees
    for data races.

Disadvantages:

-   It's unclear whether removing the data race prevention feature is sufficient
    to meaningfully address the disadvantages above. Most of the lifetime
    complexity remains unchanged.
    -   Although this would avoid the need for `RefCell`, it's likely that `Rc`
        and [`Arc`](https://doc.rust-lang.org/std/sync/struct.Arc.html) would
        still be needed.

### Dynamic lifetime safety and compile-time enforced safety otherwise (Swift's model)

Swift defaults to dynamically enforced lifetime safety and no data race
prevention, and only requires compile time enforcement of the remaining spatial
safety properties. This _does_ remove the majority of the type system complexity
needed to support the safety in Rust's model.

Advantages:

-   Significantly simpler model than Rust's.
-   Safe for all of the most common and important classes of memory safety bugs.

Disadvantages:

-   Safety based on reference counting introduces significant performance costs,
    and tools for controlling these costs are difficult. This puts the option in
    tension with the performance goal of Carbon.
    -   Safety based on garbage collection has less direct performance overhead,
        but has a greater unpredictability of performance.
-   Significant design differences versus C++ still result, as the distinction
    between value types and "class types" becomes extremely important.
    -   Class types are held by a reference counted pointer and are thus
        lifetime safe.

### Dynamic lifetime safety and defined behavior (Java's model)

Another approach to safety is to largely provide defined and predictable
behavior for all potential safety violations, which is what Java does (at the
highest level). This forms the basis of Java's safety, combined with dynamic
lifetime management in the form of garbage collection.

Advantages:

-   This approach is among the most robust and well studied models, with decades
    of practical usage and analysis for security properties.
-   Extremely suitable for efficient implementation on top of a virtual machine,
    such as the JVM.

Disadvantages:

-   Extremely high complexity to fully understand the implications of complex
    cases like data races.
-   Tends to require _significant_ performance overhead without the aid of a
    very powerful VM-based execution environment with highly dynamic
    optimizations.
    -   The complexity of the implementation makes it difficult to _predict_
        performance; for example, Java applications experience latency spikes
        when garbage collection runs. Predictable performance is a key Carbon
        goal.
