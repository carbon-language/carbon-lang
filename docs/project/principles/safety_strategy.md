# Safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO (rewriting)](#todo-rewriting)
-   [Background](#background)
    -   [What are we talking about when we discuss safety?](#what-are-we-talking-about-when-we-discuss-safety)
    -   [Guaranteed safety versus safety hardening](#guaranteed-safety-versus-safety-hardening)
-   [Philosophy](#philosophy)
-   [Principles](#principles)
-   [Details](#details)
    -   [Model for handling safety violations](#model-for-handling-safety-violations)
    -   [Managing bugs without guaranteed safety](#managing-bugs-without-guaranteed-safety)
-   [Alternatives considered](#alternatives-considered)
    -   [Guaranteed safety by default (Rust's model)](#guaranteed-safety-by-default-rusts-model)
    -   [Rust's model but without preventing data races](#rusts-model-but-without-preventing-data-races)
    -   [Dynamic lifetime safety and compile-time enforced safety otherwise (Swift's model)](#dynamic-lifetime-safety-and-compile-time-enforced-safety-otherwise-swifts-model)
    -   [Dynamic lifetime safety and defined behavior (Java's model)](#dynamic-lifetime-safety-and-defined-behavior-javas-model)

<!-- tocstop -->

## TODO (rewriting)

-   Integer overflow
    -   C++ defines wrapping
    -   Development mode defaults to trapping
    -   Performance mode may give undefined behavior
-   Bounds checking dynamic checking
    -   Likely a similar approach
-   Borrow checking static checks
    -   Redesign
-   Runtime improvements that are easier to harden

-   Having to redesign C++ code to meet Rust provability
-   C++ developers may not be happy adopting Rust proof-based mechanisms, either
    giving up performance or familiar patterns
-   Rust code requires invariants that C++ code may not respect, causing
    problems with interop for example, multiple pointers to the same constant
    address Sanitizer approaches degrade better around interop

## Background

Carbon's goal is to provide
[practical safety guarantees and testing mechanisms](../goals.md#practical-safety-guarantees-and-testing-mechanisms).

### What are we talking about when we discuss safety?

Safety is protection from software bugs, whether the protection is required by
the language or merely an implementation option. Safety can be decomposed into
categories of memory, type, and data race safety, based on the related security
vulnerability:

-   **Memory safety** protects against invalid memory accesses.

    -   _Spatial_ memory safety protects against accessing out of bounds, such
        as for an array.

    -   _Temporal_ memory safety protects against access to memory outside the
        lifetime of the intended object.

-   **Type safety** protects against accessing objects with an incorrect type,
    also known as "type confusion".

-   **Data race safety** protects against racing memory accesses, whether
    concurrent or unsynchronized.

### Guaranteed safety versus safety hardening

A safety guarantee is an especially strong requirement: the properties must be
enforced in a clearly defined way even for programs that contain bugs due to
violating the language rules. As a consequence, the behavior intended to catch
bugs remains observable by intentional code.

Safety hardening, or mitigations, address the feasibility of exploiting a
vulnerability, but may not provide the strict guarantee in order to provide
better performance in some dimension or better scaling.

Compile-time or static checking is, by definition, a safety guarantee. Runtime
or dynamic checking may be either; it must always run and check every potential
safety violation, even in an optimized production build mode, to be a safety
guarantee. If it is probabilistic or does not run in optimized production build
modes, it is hardening.

For example,
[memory tagging](https://llvm.org/devmtg/2018-10/slides/Serebryany-Stepanov-Tsyrklevich-Memory-Tagging-Slides-LLVM-2018.pdf)
provides a very strong mitigation for memory safety violations by making each
attempt at an invalid read or write have a high probability of trapping, while
still not guaranteeing to trap in every case. Realistic attacks require many
such operations, so memory tagging will probably stop attacks. Alternatively,
the trap might be asynchronous, leaving only a tiny window of time prior to the
attack being detected and program terminated. Both of these are hardening
because they reduce the feasibility of attacks, rather than guaranteeing attacks
are impossible.

## Philosophy

In order to provide
[practical safety guarantees and testing mechanisms](../goals.md#practical-safety-guarantees-and-testing-mechanisms),
Carbon will use a hybrid approach combining guaranteed safety and hardening. The
language's design should incentivize safe programming, although it will not be
required.

When writing code, Carbon developers should expect to receive safety without
explicit safety annotations. However, safety annotations are expected to support
opting in to additional safety checks, tuning safety behaviors, and adjusting
edge-case performance characteristics.

Carbon will favor compile-time safety checks are especially favorable because
they will make applications more reliable. Runtime checks will be enabled where
safety cannot be proven at compile-time. Performance concerns make it likely
that runtime checks will be probabilistic and thus hardening rather than
guaranteed safety.

Performance (including speed, binary size, and memory size) concerns will lead
to multiple build modes:

-   A development build mode with safety enabled to provide a balance with
    performance for fast development.
    -   Additional development testing build modes that can be used with tests
        and fuzzing to run more performance-intensive safety checks.
-   An optimized production build mode with some safety by default, but can also
    be tuned for application-specific hardening and performance trade-offs.

Carbon will _encourage_ developers to enable more safety checks in their code,
and be ready to support them if they do, but will also improve safety for code
for developers who cannot make the same investment.

## Principles

-   Safety must not become a barrier to the
    [usability](../goals.md#code-that-is-easy-to-read-understand-and-write) of
    Carbon. Carbon must remain easy to ramp-up with, even if it means new
    developers may only get some extra safety. Extra work to adopt opt-in safety
    eatures is acceptable, as is needing to do extra work for edge-case
    performance-tuning.

    -   Where there is a choice between safe and unsafe, the safe option should
        be incentivized by making it equally or more easy to use. If there is a
        default, it should be the safe option. It should be identifiable when
        the unsafe option is used.

    -   Language design choices should allow more efficient implementations of
        safety checks. They should also allow better automation of testing and
        fuzzing.

-   It's critical that Carbon provide better safety for C++ developers. In doing
    so, Carbon must avoid creating barriers for
    [interoperable or migrated C++ code](../goals.md#interoperability-with-and-migration-from-existing-c-code),
    so that C++ developers can take advantage of Carbon's improvements.

    -   Safety mechanisms will ideally be designed to apply to automatically
        migrated C++ code. Providing immediate safety improvements to Carbon
        adopters will help motivate adoption.

    -   In the other direction, safety mechansims must not force manual
        rewriting of C++ code in order to migrate, either by creating design
        incompatibilities or performance degradations. Automated migration of
        C++ code to Carbon must work for most developers, even if it means that
        Carbon's safety design takes a different approach.

    -   Carbon's safety should degrade gracefully when Carbon code calls C++
        code. Applications should be expected to use interoperability. Although
        some safety features will be Carbon-specific, safety should not stop at
        the language boundary.

-   Development and production (optimized) builds have different performance and
    safety needs. Some safety guarantees will be too expensive to run in a
    production system. As a result, developers should be given separate build
    modes wherein the development build mode prioritizes detection of safety
    issues over performance, whereas production builds put
    [performance first](../goals.md#performance-critical-software).

    -   Development builds should diagnose the most common safety violations
        either at compile time or runtime with high probability. Supplemental
        build modes may be offered to cover safety violations which are too
        expensive to detect by default, even for a development build.

    -   The default production build will provide runtime mitigations for safety
        violations whenever the performance is below the noise of hot path
        application code.

    -   There will be a build option for the production build to select a
        trade-off between performance (including speed, binary size, and memory
        size) and hardening or mitigation for the remaining safety violations.

-   The rules for determining whether code will pass compile-time safety
    checking should be articulable, documented, and possible to understand by
    local reasoning.

-   Developers need a strong testing methodology to engineer correct software.
    Carbon will encourage testing and then leverage it with the checking build
    modes to find and fix bugs and vulnerabilities.

Taken together, these principles imply that the language must use both compile
time and runtime checks for safety. It should use simple rules and code
annotations to prove safety at compile time when possible; for example, a subset
of what can be proved safe using Rust's lifetime-parameters and borrowing rules.
It should also provide build modes which check the remaining safety at runtime
with high probability. Over time and with more experience, the set of things
checked at compile time can incrementally increase and, if desired, arrive at
similar levels of static safety as a language like Rust.

## Details

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
