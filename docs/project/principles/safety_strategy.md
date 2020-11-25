# Safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
    -   [What are we talking about when we discuss safety?](#what-are-we-talking-about-when-we-discuss-safety)
    -   [Safety guarantees versus hardening](#safety-guarantees-versus-hardening)
-   [Philosophy](#philosophy)
-   [Principles](#principles)
-   [Details](#details)
    -   [Incremental adoption of safety](#incremental-adoption-of-safety)
    -   [Using build modes to manage safety checks](#using-build-modes-to-manage-safety-checks)
    -   [Managing bugs without compile-time safety](#managing-bugs-without-compile-time-safety)
-   [Alternatives considered](#alternatives-considered)
    -   [Guaranteed safety by default (Rust's model)](#guaranteed-safety-by-default-rusts-model)
    -   [Runtime lifetime safety and compile-time enforced safety otherwise (Swift's model)](#runtime-lifetime-safety-and-compile-time-enforced-safety-otherwise-swifts-model)
    -   [Runtime lifetime safety and defined behavior (Java's model)](#runtime-lifetime-safety-and-defined-behavior-javas-model)

<!-- tocstop -->

## Background

Carbon's goal is to provide
[practical safety guarantees and testing mechanisms](../goals.md#practical-safety-guarantees-and-testing-mechanisms).

### What are we talking about when we discuss safety?

Safety is protection from software bugs, whether the protection is required by
the language or merely an implementation option. Safety can be decomposed into
categories of memory, type, and data race safety, based on the related security
vulnerability:

-   **Memory safety** protects against invalid memory accesses.

    -   _Spatial_ memory safety protects against accessing an address that's out
        of bounds for the source. This includes out array boundaries as well as
        dereferencing invalid pointers such as uninitialized pointers, `NULL` in
        C++, or manufactured pointer addresses.

    -   _Temporal_ memory safety protects against use-after-free access. This
        typically involves dereferencing pointers to dynamically allocated
        objects, but can also include pointers that are given the address of
        stack objects.

-   **Type safety** protects against accessing objects with an incorrect type,
    also known as "type confusion".

-   **Data race safety** protects against racing memory access: when a thread
    accesses (read or write) a memory location concurrently with a different
    writing thread and without synchronizing.

### Safety guarantees versus hardening

The underlying goal of safety is to prevent attacks from turning a _logic error_
into a _security vulnerability_. The three ways of doing this can be thought of
how to prevent attacks:

-   **Safety guarantees** prevent bugs. They offer a strong requirement that a
    particular security vulnerability cannot exist. Compile-time safety checks
    are always a safety guarantee, but safety guarantees may also be done at
    runtime. For example:

    -   At compile-time, range-based for loops offer a spatial safety guarantee
        that out-of-bounds issues cannot exist.

    -   At runtime, garbage collected languages offer a temporal safety
        guarantee because objects cannot be freed while there's still a
        reference.

-   **Error detection** checks for common logic errors at runtime. For example:

    -   An array lookup function might offer spatial memory error detection by
        verifying that the passed index is in-bounds.

    -   A program can implement reference counting to detect a temporal memory
        error by ensuring that no references remain when memory is freed.

-   **Safety hardening** mitigates bugs, minimizing the feasibility of an
    attack. For example:

    -   [Memory tagging](https://llvm.org/devmtg/2018-10/slides/Serebryany-Stepanov-Tsyrklevich-Memory-Tagging-Slides-LLVM-2018.pdf)
        makes each attempt at an invalid read or write operation have a high
        probability of trapping, while still not detecting the underlying bug in
        every case. Realistic attacks require many such operations, so memory
        tagging will probably stop attacks. Alternatively, the trap might be
        asynchronous, leaving only a tiny window of time prior to the attack
        being detected and program terminated. These are probabilistic hardening
        and reduces the feasibility of both spatial and temporal memory attacks.

Under both error detection and safety hardening, even if a safety is protected,
the underlying bugs will still exist and will need to be fixed. For example,
program termination could be used for a denial-of-service attack.

## Philosophy

When providing
[practical safety and testing mechanisms](../goals.md#practical-safety-and-testing-mechanisms),
Carbon will put the most emphasis on error detection and safety hardening. Where
feasible, guaranteed safety will be offered in a way that removes the need for
other mitigations. The language's design should incentivize safe programming,
although it will not be required.

When writing code, Carbon developers should expect to receive some safety
without explicit safety annotations. However, safety annotations are expected to
support opting in to additional safety checks, tuning safety behaviors, and
adjusting edge-case performance characteristics.

Carbon will favor compile-time safety checks because catching issues early will
make applications more reliable. However, the comprehensiveness of compile-time
safety will be limited because Carbon won't require redesigning code to take
advantage of them. Runtime checks, either error detection or safety hardening,
will be enabled where safety cannot be proven at compile-time. Over time, this
hybrid approach to safety checks should evolve to provide a similar level of
safety to a statically checked language such as Rust.

Performance (including speed, binary size, and memory size) concerns will lead
to multiple build modes:

-   A main development build mode with safety enabled to provide a balance with
    performance for fast development.
    -   Additional development testing build modes that can be used with tests
        and fuzzing to run more performance-intensive safety checks.
-   An optimized release build mode with some safety by default, but can also be
    tuned for application-specific trade-offs between performance, error
    detection, and safety hardening.

Carbon will _encourage_ developers to enable more safety checks in their code,
and be ready to support them if they do, but will also improve safety for code
for developers who cannot make the same investment.

## Principles

-   Safety must be
    [easy to ramp-up with](../goals.md#code-that-is-easy-to-read-understand-and-write),
    even if it means new developers may only get some extra safety.

    -   The common case should be that developers don't need to rewrite their
        code to take advantage of Carbon's safety model. Some should be enabled
        by default, and some safety will require work to opt-in. Developers
        concerned with performance should only need to work to disable safety in
        rare edge-cases.

    -   Where there is a choice between safe and unsafe, the safe option should
        be incentivized by making it equally or more easy to use. If there is a
        default, it should be the safe option. It should be identifiable when
        the unsafe option is used.

    -   Language design choices should allow more efficient implementations of
        safety checks. They should also allow better automation of testing and
        fuzzing.

-   Safety in Carbon must work with
    [interoperable or migrated C++ code](../goals.md#interoperability-with-and-migration-from-existing-c-code),
    so that C++ developers can readily take advantage of Carbon's improvements.

    -   Safety mechanisms will ideally be designed to apply to automatically
        migrated C++ code. Providing immediate safety improvements to Carbon
        adopters will help motivate adoption.

    -   In the other direction, safety mechanisms must not force manual
        rewriting of C++ code in order to migrate, either by creating design
        incompatibilities or performance degradations. Automated migration of
        C++ code to Carbon must work for most developers, even if it means that
        Carbon's safety design takes a different approach.

    -   Carbon's safety should degrade gracefully when Carbon code calls C++
        code. Applications should be expected to use interoperability. Although
        some safety features will be Carbon-specific, safety should not stop at
        the language boundary.

-   The development build mode should prioritize detection of safety issues over
    performance, whereas optimized release builds put
    [performance first](../goals.md#performance-critical-software).

    -   Compile-time safety checks should occur regardless of build mode.

    -   Development builds should diagnose the most common safety violations
        either at runtime with high probability. Supplemental build modes may be
        offered to cover safety violations which are too expensive to detect by
        default, even for a development build.

    -   The default optimized release build will provide runtime mitigations for
        safety violations whenever the performance is below the noise of hot
        path application code.

    -   There will be a build option for the optimized release build to choose
        whether to make non-default choices about the trade-off between
        performance (including speed, binary size, and memory size) and
        measurably expensive error detection and safety hardening.

-   The rules for determining whether code will pass compile-time safety
    checking should be articulable, documented, and possible to understand by
    local reasoning.

-   Developers need a strong testing methodology to engineer correct software.
    Carbon will encourage testing and then leverage it with the checking build
    modes to find and fix bugs and vulnerabilities.

## Details

### Incremental adoption of safety

Carbon is prioritizing usability of the language, particularly minimizing
retraining of C++ developers and easing migration of C++ codebases, over the
kind of provable safety that some other languages pursue, particularly Rust.

A key motivation of Carbon is to move C++ developers to a better, safer
language. However, if Carbon requires manually rewriting or redesigning C++ code
in order to maintain performance, it creates additional pressure on C++
developers to learn and spend time on safety. Safety will often not be the top
priority for developers; as a result, Carbon must be thoughtful about how and
when it forces developers to think about safety.

Relying on multiple build modes to provide safety should fit into normal
development workflows. Carbon can also have features to enable additional
safety, so long as developers can start using Carbon in their applications
_without_ leaning new paradigms.

Carbon should enable developers to incrementally adopt of safety features, and
also work to incrementally improve safety without requiring code or design
alterations. This does not mean that _every_ feature needs to be

### Using build modes to manage safety checks

Carbon will likely start in a state where most safety checks are done at
runtime. However, runtime detection of safety violations remains expensive. In
order to make as many safety checks as possible available to developers, Carbon
will adopt a strategy based on multiple build modes that separate out expensive
runtime checks to where they're feasible to run from a performance perspective.

The main development build mode will emphasize debugability of both bugs and
safety issues. This means it needs to perform well enough to be run as part of
the normal developer workflow, but does not need to have the performance of an
optimized release build. This mode should run runtime checks for the most common
safety issues. Developers should do most of their testing in this build mode.

Some safety checks will still be considered too expensive to run in the main
development build mode, particularly in combination with each other. These
checks will be placed into supplemental development build modes that will
collectively provide more comprehensive safety checks for the codebase as a
whole.

All development build modes will place a premium on the debugability of safety
violations. Where safety checks rely on hardening instead of guaranteed safety,
violations should be detected with a high probability per single occurrence of
the bug. Detected bugs will be accompanied by a detailed diagnostic report to
ease developer bug-finding.

The optimized release build mode will emphasize performance; as a consequence,
fewer safety checks will be run, and those that are can be expected to provide
less diagnostic information. Only safety techniques which don't measurably
impact application hot path performance will be enabled by default. This is a
very high bar, but is crucial for meeting Carbon's performance goals, as well as
allowing migration of existing C++ systems which may not have been designed with
Carbon's safety semantics in mind.

Different applications will have different tolerances for performance overhead,
so build modes will also allow developers to choose the balance that meets their
application's hardening versus performance needs.

For example, a possible approach to integer overflow might:

-   Check and trap in the development build mode.
-   Be unchecked and wrap in the optimized release mode, for performance.
-   Have an option to enable trapping in the optimized release mode, for
    safety-critical applications.
-   Provide integer types that either always wrap, for libraries requiring that
    behavior, or always trap, for safety-critical libraries.

### Managing bugs without compile-time safety

Carbon's reliance on runtime checks will allow developers to manage their
security risk. Developers will still need to reliably find and fix the
inevitable bugs, including both safety violations and regular business logic
bugs. The cornerstone of managing bugs will be strong testing methodologies,
with built-in support from Carbon.

Strong testing is more than good test coverage. It means a combination of:

-   Ensuring unsafe or risky operations and interfaces can easily be recognized
    by developers.

-   Using static analysis tools to detect common bugs, and ensuring they're
    integrated into build and code review workflows. These could be viewed as
    static testing of code.

-   Writing good test coverage, including unit, integration, and system tests.

-   Generating coverage-directed fuzz testing to discover bugs outside of
    manually authored test coverage, especially for interfaces handling
    untrusted data. Fuzz testing is a robust way to catch bugs when APIs may be
    used in ways developers don't consider.

-   Running continuous integration, including automatic and continuous running
    of these tests. The checked development build mode should be validated, as
    well as any additional build modes necessary to cover different forms of
    behavior checking.

-   Easing automated testing and fuzzing through language features. For example,
    if the language encourages value types and pure functions of some sort, they
    can be automatically fuzzed.

These practices are necessary for reliable, large-scale software engineering.
Maintaining correctness of business logic over time requires continuous and
thorough testing. Without it, such software systems cannot be changed and
evolved over time reliably. Carbon will re-use these practices in conjunction
with checking build modes to mitigate the limitations of Carbon's safety
guarantees without imposing overhead on production systems.

When a developer chooses to use Carbon, adhering to this kind of testing
methodology is essential for maintaining safety. As a consequence, Carbon's
ecosystem, including the language, tools, and libraries, will need to directly
work to remove barriers and encourage the development of these methodologies.

The reliance on testing may make Carbon a poor choice in some environments; in
environments where such testing rigor is infeasible, a language with a greater
degree of static checking may be better suited.

## Alternatives considered

When considering alternatives, they are evaluated against what can secure the
optimized release build mode. Some techniques may have performance implications
which are too expensive to use in the optimized release build, while still
offering inspiring techniques for catching safety violations in development
build modes.

### Guaranteed safety by default (Rust's model)

Carbon could provide guaranteed safety by default. With Rust as an example, this
would require a combination of compile-time and runtime memory safety
techniques. This approach still allows for
[`unsafe` blocks](https://doc.rust-lang.org/rust-by-example/unsafe.html), as
well as types that offer runtime safety while wrapping `unsafe` interfaces.

Advantages:

-   Guaranteed safety, including against data races, is provided for the
    binaries.
    -   The emphasis on compile-time safety limits the scope of the runtime
        memory safety costs.
    -   With Rust, there is early evidence that there's a significant impact in
        reducing bugs generally.
-   Imitating Rust's techniques would allow building on the huge work of the
    Rust community, reducing the risks of implementing similar in Carbon.
-   Careful use of narrow `unsafe` escape hatches can be effectively
    encapsulated behind otherwise safe APIs.

Disadvantages:

-   Rust's approach to compile-time safety requires use of
    [design patterns and idioms](https://github.com/rust-unofficial/patterns)
    that are substantially different from C++.
    -   Conversion of C++ code to Rust results in either rewrites of code, or
        use of runtime safety checks that impair performance.
    -   Requires fully modeling lifetime and exclusivity in the type system.
    -   Data structures must be redesigned to avoid sharing mutable state.
    -   Increases complexity of node and pointer based data structures, such as
        linked lists.
-   Rust's compiler has been a long-standing target of performance improvements,
    but is still considered slow. A lot of this comes from how borrow checking
    is implemented, and the compile-time validation necessary for that. It's
    likely that approaches imitating Rust would have the same issues.
-   The complexity of using Rust's compile-time safety may incentivize
    unnecessary runtime checking of safety properties. For example, using
    [`RefCell`](https://doc.rust-lang.org/std/cell/struct.RefCell.html) or
    [`Rc`](https://doc.rust-lang.org/std/rc/struct.Rc.html) to avoid changing
    designs to fit compile-time safety models.
-   Some of the most essential safety tools that ease the ergonomic burden of
    the Rust-style lifetime model (`Rc`) introduce _semantic_ differences that
    cannot then be eliminated in a context where performance is the dominant
    priority.

It's possible to modify the Rust model several ways in order to reduce the
burden on C++ developers:

-   Don't offer safety guarantees for data races, eliminating `RefCell`.
    -   This would likely not avoid the need for `Rc` or `Arc`, and wouldn't
        substantially reduce the complexity.
-   Require manual destruction of `Rc`, allowing safety guarantees to be
    disabled in optimized release builds for better performance.
    -   This is closer to a decision to _not_ provide guaranteed safety, but
        would still require redesign of C++ code.

Overall, Carbon is making a compromise around safety in order to give a path for
C++ to evolve. C++ developers must be comfortable migrating their codebases, and
able to do so in a largely automated manner. In order to achieve automated
migration, Carbon cannot require fundamental redesigns of migrated C++ code.
While a migration tool could in theory mark all migrated code as `unsafe`,
Carbon should use a safety strategy that degrades gracefully and offers
improvements for C++ code, whether migrated or not.

That does not mean Carbon will never adopt guaranteed safety by default, only
that performance and migration of C++ code takes priority, and any design will
need to be considered in the context of other goals. It should still be possible
to adopt guaranteed safety later, although it will require identifying a
migration path.

### Runtime lifetime safety and compile-time enforced safety otherwise (Swift's model)

Carbon could provide runtime lifetime safety with no data race prevention,
mirroring Swift's model. This only requires compile-time enforcement of the
remaining spatial safety properties. This _does_ remove the majority of the type
system complexity needed to support the safety in Rust's model.

Advantages:

-   Significantly simpler model than Rust's.
-   Safe for all of the most common and important classes of memory safety bugs.

Disadvantages:

-   Safety based on reference counting introduces significant performance costs,
    and tools for controlling these costs are difficult.
    -   Safety based on garbage collection has less direct performance overhead,
        but has a greater unpredictability of performance.
-   Significant design differences versus C++ still result, as the distinction
    between value types and "class types" becomes extremely important.
    -   Class types are held by a reference counted pointer and are thus
        lifetime safe.

Swift was designated by Apple as the replacement for Objective-C. The safety
versus performance trade-offs that it makes fit Apple's priorities. Carbon's
performance goals should lead to different trade-off decisions with a higher
priority on peak performance.

### Runtime lifetime safety and defined behavior (Java's model)

Another approach to safety is to largely provide defined and predictable
behavior for all potential safety violations, which is what Java does (at the
highest level). This forms the basis of Java's safety, combined with dynamic
memory management in the form of garbage collection.

Advantages:

-   This approach is among the most robust and well studied models, with decades
    of practical usage and analysis for security properties.
-   Extremely suitable for efficient implementation on top of a virtual machine,
    such as the JVM.

Disadvantages:

-   Extremely high complexity to fully understand the implications of complex
    cases like data races.
-   Tends to require _significant_ performance overhead without the aid of a
    very powerful VM-based execution environment with extensive optimizations.
    -   The complexity of the implementation makes it difficult to _predict_
        performance; for example, Java applications experience latency spikes
        when garbage collection runs.

Carbon's performance goals prioritize reliability of performance. The
unpredictable performance impacts of a garbage collector make it a less
desirable choice.
