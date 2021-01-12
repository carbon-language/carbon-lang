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
    -   [Incremental work when safety requires work](#incremental-work-when-safety-requires-work)
    -   [Using build modes to manage safety checks](#using-build-modes-to-manage-safety-checks)
        -   [Debug](#debug)
        -   [Performance](#performance)
        -   [Hardened](#hardened)
    -   [Managing bugs without compile-time safety](#managing-bugs-without-compile-time-safety)
-   [Alternatives considered](#alternatives-considered)
    -   [Alternative models](#alternative-models)
        -   [Guaranteed safety by default (Rust's model)](#guaranteed-safety-by-default-rusts-model)
        -   [Runtime lifetime safety and compile-time enforced safety otherwise (Swift's model)](#runtime-lifetime-safety-and-compile-time-enforced-safety-otherwise-swifts-model)
        -   [Runtime lifetime safety and defined behavior (Java's model)](#runtime-lifetime-safety-and-defined-behavior-javas-model)
    -   [Build mode names](#build-mode-names)

<!-- tocstop -->

## Background

Carbon's goal is to provide
[practical safety and testing mechanisms](../goals.md#practical-safety-and-testing-mechanisms).

### What are we talking about when we discuss safety?

Safety is protection from software bugs, whether the protection is required by
the language or merely an implementation option. Application-specific logic
errors can be prevented by testing, but can lead to security vulnerabilities in
production; these vulnerabilities are something Carbon will protect against. The
protections are named based on the kind of security vulnerability they protect
against:

-   [**Memory safety**](https://en.wikipedia.org/wiki/Memory_safety) protects
    against invalid memory accesses. We use
    [two main subcategories](https://onlinelibrary.wiley.com/doi/full/10.1002/spe.2105)
    for memory safety:

    -   _Spatial_ memory safety protects against accessing an address that's out
        of bounds for the source. This includes array boundaries, as well as
        dereferencing invalid pointers such as uninitialized pointers, `NULL` in
        C++, or manufactured pointer addresses.

    -   _Temporal_ memory safety protects against use-after-free access. This
        typically involves dereferencing pointers to dynamically allocated
        objects, but can also include pointers that are given the address of
        stack objects.

-   [**Type safety**](https://en.wikipedia.org/wiki/Type_safety) protects
    against accessing valid memory with an incorrect type, also known as "type
    confusion".

-   [**Data race safety**](https://en.wikipedia.org/wiki/Race_condition#Data_race)
    protects against racing memory access: when a thread accesses (read or
    write) a memory location concurrently with a different writing thread and
    without synchronizing.

### Safety guarantees versus hardening

The underlying goal of safety is to prevent attacks from turning a _logic error_
into a _security vulnerability_. The three ways of doing this can be thought of
in terms of how they prevent attacks:

-   **Safety guarantees** prevent bugs. They offer a strong requirement that a
    particular security vulnerability cannot exist. Compile-time safety checks
    are always a safety guarantee, but safety guarantees may also be done at
    runtime. For example:

    -   At compile-time, range-based for loops offer a spatial safety guarantee
        that out-of-bounds issues cannot exist in the absence of concurrent
        modification of the sequence.

    -   At runtime, garbage collected languages offer a temporal safety
        guarantee because objects cannot be freed while there's still a
        reference.

-   **Error detection** checks for common logic errors at runtime. For example:

    -   An array lookup function might offer spatial memory error detection by
        verifying that the passed index is in-bounds.

    -   A program can implement reference counting to detect a temporal memory
        error by ensuring that no references remain when memory is freed.

-   **Safety hardening** mitigates bugs, typically by minimizing the feasibility
    of an attack. For example:

    -   [Control Flow Integrity (CFI)](https://en.wikipedia.org/wiki/Control-flow_integrity)
        monitors for behavior which can subvert the program's control flow. In
        [Clang](http://clang.llvm.org/docs/ControlFlowIntegrity.html), it is
        optimized for use in release builds. Typically CFI analysis will only
        detect a subset of attacks because it can't track each possible code
        path separately. It should still reduce the feasibility of both spatial
        memory, temporal memory, and type attacks.

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
make applications more reliable. Runtime checks, either error detection or
safety hardening, will be enabled where safety cannot be proven at compile-time.

There will be a split of build modes driven by runtime safety approaches,
wherein each has a specific focus and will be treated as its own ABI:

-   A **debug** build mode for routine development. This will balance fast
    development and testing with the need for reliable detection and easy
    debugging of safety issues.

-   A **performance** build mode for most application releases. This will focus
    on maintaining high performance. It will enable hardening where it does not
    measurably affect performance.

-   A **hardened** build mode for security-sensitive appliations. This will
    prioritize providing hardening for _all_ safety issues, which requires
    significant performance sacrifices.

The addition of further safety-oriented build modes may occur based on safety
versus performance trade-offs, but three is considered the likely long-term
stable outcome.

Over time, safety should [evolve](../goals.md#software-and-language-evolution)
using a hybrid compile-time and runtime safety approach to provide a similar
level of safety to a statically checked language, such as Rust. However, while
Carbon may _encourage_ developers to modify code in support of more efficient
safety checks, it will remain important to improve the safety of code for
developers who cannot invest into safety-specific code modifications.

## Principles

-   Safety must be
    [easy to ramp-up with](../goals.md#code-that-is-easy-to-read-understand-and-write),
    even if it means new developers may only get some extra safety.

    -   Developers should benefit from Carbon's safety without needing to learn
        and apply Carbon-specific design patterns. Some safety should be enabled
        by default, without safety-specific work, although some safety will
        require work to opt in. Developers concerned with performance should
        only need to work to disable safety in rare edge-cases.

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
        code, although this may require use of the Carbon toolchain to compile
        the C++ code. Applications should be expected to use interoperability.
        Although some safety features will be Carbon-specific, safety should not
        stop at the language boundary.

-   The rules for determining whether code will pass compile-time safety
    checking should be articulable, documented, and easy to understand.

    -   Compile-time safety checks should not change significantly across
        different build modes. The purpose of the build modes is to determine
        code generation.

-   Each build mode will treat safety differently based on its priority.

    -   The debug build mode will provide high-probability runtime diagnostics
        for the most common safety violations.

    -   The performance build mode will provide runtime mitigations for safety
        violations when they don't have measurable performance impact for hot
        path application code.

    -   The hardened build mode will mitigate safety issues consistently. It is
        acceptable for this to have techniques that dramatically reduce
        performance.

    -   Although tuning options may be supported, first-class support will be on
        the primary three build modes.

-   Any further build modes beyond the noted three will need to be carefully
    evaluated for merging into one of the the primary three.

    -   For example, Carbon may add build modes for particularly expensive
        sanitizers, to improve error detection in tests. However, making an
        additional sanitizer build mode useful would require developers test
        under both debug and sanitizer build modes; that is expected to be
        easily forgotten by developers, and less-effective sanitizers that can
        be integrated into the debug build mode are expected to catch issues
        more frequently, and will be preferred.

-   Each distinct safety-related build mode (debug, performance, and hardened)
    should be treated as its own ABI.

    -   Standard cross-ABI interfaces will exist in Carbon, and will need to be
        used by developers interested in combining libraries built under
        different build modes.

-   Runtime safety hardening and mitigations will typically terminate the
    program in response to safety issues, because they indicate a logic error
    and recovery would leave a program in an unpredictable state. Safety issues
    will still be bugs, even if they're prevented from becoming a security hole.

-   Developers need a strong testing methodology to engineer correct software.
    Carbon will encourage testing and then leverage it with the checking build
    modes to find and fix bugs and vulnerabilities.

## Details

### Incremental work when safety requires work

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
_without_ learning new paradigms.

Where possible, safety checks shouldn't require work on the part of Carbon
developers. A safety check that requires no code edits or can be handled by
automated migration may be opt-out, as there is negligible cost to developers.
One which requires local code changes should be opt-in because costs will scale
with codebase size. Safety check approaches which would require substantial
redesign by developers will be disfavored based on adoption cost, even if the
alternative is a less-comprehensive approach.

### Using build modes to manage safety checks

Carbon will likely start in a state where most safety checks are done at
runtime. However, runtime detection of safety violations remains expensive. In
order to make as many safety checks as possible available to developers, Carbon
will adopt a strategy based on multiple build modes that target key use-cases.

#### Debug

The debug build mode targets developers who are iterating on code and running
tests. It will emphasize debugability, especially for safety issues.

It needs to perform well enough to be run frequently by developers, but will
make performance sacrifices to catch more safety issues. This mode should have
runtime checks for the most common safety issues, but it can make trade-offs
that improve performance in exchange for less frequent, but still reliability,
detection. Developers should do most of their testing in this build mode.

The debug build mode will place a premium on the debugability of safety
violations. Where safety checks rely on hardening instead of guaranteed safety,
violations should be detected with a high probability per single occurrence of
the bug. Detected bugs will be accompanied by a detailed diagnostic report to
ease developer bug-finding.

#### Performance

The performance build mode targets the average developer who wants high
performance from Carbon code, where performance considers processing time,
memory, and disk space. Trade-offs will be made that maximize the performance.

Only safety techniques which don't measurably impact application hot path
performance will be enabled by default. This is a very high bar, but is crucial
for meeting Carbon's performance goals, as well as allowing migration of
existing C++ systems which may not have been designed with Carbon's safety
semantics in mind.

#### Hardened

The hardened build mode targets developers who are ready to sacrifice
performance in order to improve safety. It will run as many safety checks as it
can, even with significant performance overheads. It should be expected to
detect safety issues _consistently_, in ways that attackers cannot work around.

The hardened build mode will prefer non-probabilistic techniques because it's
assumed that attackers can manipulate probabilities. For example, a detection
technique may be able to detect safety issues with a 95% chance per memory
operation assuming random memory locations. However, an attacker may be able to
cause memory allocations that result in non-random memory locations, which may
be usable to reliably evade the probabilistic detection.

Consequently, probabilistic techniques will only be used where either it can be
proven conclusively that attackers cannot manipulate the probabilities (which is
typically infeasible to prove), or where performance overheads of
non-probabilistic detection are infeasible, even given the assumed lower
performance of the hardened build mode.

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

### Alternative models

When considering models, they must be adoptable without hindering performance
builds. Carbon will not create build-mode-specific programming models.

#### Guaranteed safety by default (Rust's model)

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
-   Imitating Rust's techniques may prove insufficient for achieving Carbon's
    [compiler performance goals](../goals.md#fast-and-scalable-development).
    Rust compilation performance suggests its borrow checking performance is
    slow, although it's difficult to determine how significant this is or
    whether it could be improved.
    -   The Rust compiler
        [is slow](https://pingcap.com/blog/rust-compilation-model-calamity),
        although
        [much has been done to improve it](https://blog.mozilla.org/nnethercote/2020/09/08/how-to-speed-up-the-rust-compiler-one-last-time/).
    -   Details of type checking, particularly requiring parsing of function
        bodies to type check signatures, as well as wide use of
        [monomorphization](https://doc.rust-lang.org/book/ch10-01-syntax.html)
        are likely significant contributors to Rust compilation performance.
    -   LLVM codegen is also a significant cost for Rust compilation
        performance.
    -   With
        [Fuchsia](https://fuchsia.dev/fuchsia-src/development/languages/rust) as
        an example, in December 2020, borrow checking and type checking combined
        account for around 10% of Rust compile CPU time, or 25% of end-to-end
        compile time. The current cost of borrow checking is obscured both
        because of the combination with type checking, and because Fuchsia
        disables some compiler parallelization due to build system
        incompatibility.
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
-   Require manual destruction of `Rc`, allowing safety checks to be disabled in
    the performance build mode to eliminate overhead.
    -   This still requires redesigning C++ code to take advantage of `Rc`.
    -   The possibility of incorrect manual destruction means that the safety
        issue is being turned into a bug, which means it is hardening and no
        longer a safety guarantee.
    -   Carbon can provide equivalent hardening through techniques such as
        [MarkUs](https://www.cl.cam.ac.uk/~tmj32/papers/docs/ainsworth20-sp.pdf),
        which does not require redesigning C++ code.

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

#### Runtime lifetime safety and compile-time enforced safety otherwise (Swift's model)

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
    -   Swift is planning to add an option for unique ownership, although the
        specifics are not designed yet. Unique ownership by itself does not
        address performance issues because it also needs unowned/unsafe access
        for "borrowing". Swift provides these unsafe features, but they are not
        idiomatic. Also, requiring unsafe access will constrain safety checks.
-   Significant design differences versus C++ still result, as the distinction
    between value types and "class types" becomes extremely important.
    -   Class types are held by a reference counted pointer and are thus
        lifetime safe.

Swift was designated by Apple as the replacement for Objective-C. The safety
versus performance trade-offs that it makes fit Apple's priorities. Carbon's
performance goals should lead to different trade-off decisions with a higher
priority on peak performance.

#### Runtime lifetime safety and defined behavior (Java's model)

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

### Build mode names

The build mode concepts are difficult to name. Other names that were evaluated,
and are ultimately similar, are:

-   "Debug" is a common term for the intended use of this build mode. Also,
    tooling including Visual Studio frequently uses the debug term for
    describing similar.

    -   "Development" was also considered, but could be less focused on the
        primary goal.

-   "Performance" aligns with the phrasing of the language performance goal.

    -   "Optimized" implies that other modes would not be fully optimized, but
        hardened should be optimized.

    -   "Fast" would be okay, but "performance" aligns better with the language
        goals.

-   "Hardened" is the choice for succinctly describing the additional safety
    measures that will be taken, and is a well-known term in the safety space.
    It could be incorrectly inferred that "performance" has no hardening, but
    the preference is to clearly indicate the priority of the "hardened" build
    mode.

    -   "Safe" implies something closer to guaranteed safety. However, safety
        bugs should be expected to result in program termination, which can
        still be used in other attacks, such as Denial-of-Service.

    -   "Mitigated" is an overloaded term, and it may not be succinctly clear
        that it's about security mitigations.

-   Some terms which were considered and don't fit well into the above groups
    are:

    -   "Release" is avoided because both "performance" and "hardened" could be
        considered to be "release" build modes.

The names "performance" and "hardened" may lead to misinterpretations, with some
developers who should use "hardened" using "performance" because they are
worried about giving up too much performance, and the other way around. The
terms try to balance the utility of well-known terminology with the succinctness
of a short phrase for build modes, and that limits the expressivity. Some
confusion is expected, and documentation as well as real-world experience (for
example, a developer who cares about latency benchmarking both builds) should be
expected to help mitigate mix-ups.
