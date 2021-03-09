# Safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
    -   [What "safety" means in Carbon](#what-safety-means-in-carbon)
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
-   [Caveats](#caveats)
    -   [Probabilistic techniques likely cannot stop attacks](#probabilistic-techniques-likely-cannot-stop-attacks)
-   [Alternatives considered](#alternatives-considered)
    -   [Guaranteed memory safety programming models](#guaranteed-memory-safety-programming-models)
        -   [Guaranteed compile-time memory safety using borrow checking](#guaranteed-compile-time-memory-safety-using-borrow-checking)
        -   [Guaranteed run-time memory safety using reference counting](#guaranteed-run-time-memory-safety-using-reference-counting)
        -   [Guaranteed run-time memory safety using garbage collection](#guaranteed-run-time-memory-safety-using-garbage-collection)
    -   [Build mode names](#build-mode-names)
    -   [Performance versus safety in the hardened build mode](#performance-versus-safety-in-the-hardened-build-mode)
    -   [Add more build modes](#add-more-build-modes)

<!-- tocstop -->

## Background

Carbon's goal is to provide
[practical safety and testing mechanisms](../goals.md#practical-safety-and-testing-mechanisms).

### What "safety" means in Carbon

Safety is protection from software bugs, whether the protection is required by
the language or merely an implementation option. Application-specific logic
errors can be prevented by testing, but can lead to security vulnerabilities in
production. Safety categories will be referred to using names based on the type
of
[security vulnerability](<https://en.wikipedia.org/wiki/Vulnerability_(computing)#Software_vulnerabilities>)
they protect against.

A key subset of safety categories Carbon should address are:

-   [**Memory safety**](https://en.wikipedia.org/wiki/Memory_safety) protects
    against invalid memory accesses. Carbon uses
    [two main subcategories](https://onlinelibrary.wiley.com/doi/full/10.1002/spe.2105)
    for memory safety:

    -   _Spatial_ memory safety protects against accessing an address that's out
        of bounds for the source. This includes array boundaries, as well as
        dereferencing invalid pointers such as uninitialized pointers, `NULL` in
        C++, or manufactured pointer addresses.

    -   _Temporal_ memory safety protects against accessing an address that has
        been deallocated. This includes use-after-free for heap and
        use-after-return for stack addresses.

-   [**Type safety**](https://en.wikipedia.org/wiki/Type_safety) protects
    against accessing valid memory with an incorrect type, also known as "type
    confusion".

-   [**Data race safety**](https://en.wikipedia.org/wiki/Race_condition#Data_race)
    protects against racing memory access: when a thread accesses (read or
    write) a memory location concurrently with a different writing thread and
    without synchronizing.

### Safety guarantees versus hardening

In providing safety, the underlying goal is to prevent attacks from turning a
_logic error_ into a _security vulnerability_. The three ways of doing this can
be thought of in terms of how they prevent attacks:

-   **Safety guarantees** prevent bugs. They offer a strong requirement that a
    particular security vulnerability cannot exist. Compile-time safety checks
    are always a safety guarantee, but safety guarantees may also be done at
    runtime. For example:

    -   At compile-time, range-based for loops offer a spatial safety guarantee
        that out-of-bounds issues cannot exist in the absence of concurrent
        modification of the sequence.

    -   At runtime, garbage collected languages offer a temporal safety
        guarantee because objects cannot be freed while they're still
        accessible.

-   **Error detection** checks for common logic errors at runtime. For example:

    -   An array lookup function might offer spatial memory error detection by
        verifying that the passed index is in-bounds.

    -   A program can implement reference counting to detect a temporal memory
        error by checking whether any references remain when memory is freed.

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
        tagging may stop attacks in some environments. Alternatively, the trap
        might be asynchronous, leaving only a tiny window of time prior to the
        attack being detected and program terminated. These are probabilistic
        hardening and reduces the feasibility of both spatial and temporal
        memory attacks.

Under both error detection and safety hardening, even if a safety is protected,
the underlying bugs will still exist and will need to be fixed. For example,
program termination could be used for a denial-of-service attack.

## Philosophy

Carbon's
[practical safety and testing mechanisms](../goals.md#practical-safety-and-testing-mechanisms)
will emphasize guaranteed safety where feasible without creating barriers to
Carbon's [other goals](../goals.md), particularly performance and
interoperability. This limits Carbon's options for guaranteed safety, and as a
result there will be more reliance upon error detection and safety hardening.
The language's design should incentivize safe programming, although it will not
be required.

When writing code, Carbon developers should expect to receive safety without
needing to add safety annotations. Carbon will have optional safety annotations
for purposes such as optimizing safety checks or providing information that
improves coverage of safety checks.

Carbon will favor compile-time safety checks because catching issues early will
make applications more reliable. Runtime checks, either error detection or
safety hardening, will be enabled where safety cannot be proven at compile-time.

There will be three high-level use cases or directions that Carbon addresses
through different build modes that prioritize safety checks differently:

-   A [debug](#debug) oriented build mode that prioritizes detecting bugs and
    reporting errors helpfully.
-   A [performance](#performance) oriented build mode that skips any dynamic
    safety checks to reduce overhead.
-   A [hardened](#hardened) oriented build mode that prioritizes ensuring
    sufficient safety to prevent security vulnerabilities, although it may not
    allow detecting all of the bugs.

These high level build modes may be tuned, either to select specific nuanced
approach for achieving the high level goal, or to configure orthogonal
constraints such as whether to prioritize binary size or execution speed.
However, there is a strong desire to avoid requiring more fundamental build
modes to achieve the necessary coverage of detecting bugs and shipping software.
These build modes are also not expected to be interchangeable or compatible with
each other within a single executable -- they must be a global selection.

Although expensive safety checks could be provided through additional build
modes, Carbon will favor safety checks that can be combined into these three
build modes rather than adding more.

Over time, safety should [evolve](../goals.md#software-and-language-evolution)
using a hybrid compile-time and runtime safety approach to eventually provide a
similar level of safety to a language that puts more emphasis on guaranteed
safety, such as [Rust](#guaranteed-safety-by-default-rusts-model). However,
while Carbon may _encourage_ developers to modify code in support of more
efficient safety checks, it will remain important to improve the safety of code
for developers who cannot invest into safety-specific code modifications.

## Principles

-   Safety must be
    [easy to ramp-up with](../goals.md#code-that-is-easy-to-read-understand-and-write),
    even if it means new developers don't receive the full safety that Carbon
    can offer.

    -   Developers should benefit from Carbon's safety without needing to learn
        and apply Carbon-specific design patterns. Some safety should be enabled
        by default, without safety-specific work, although some safety will
        require work to opt in. Developers concerned with performance should
        only need to disable safety in rare edge-cases.

    -   Where there is a choice between safety approaches, the safe option
        should be incentivized by making it equally easy or easier to use. If
        there is a default, it should be the safe option. It should be
        identifiable when the unsafe option is used. Incentives will prioritize,
        in order:

        1.  Guaranteed safety.
        2.  Error detection.
        3.  Safety hardening.
        4.  Unsafe and unmitigated code.

    -   Language design choices should favor more efficient implementations of
        safety checks. They should also allow favor automation of testing and
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
        C++ code to Carbon must work for most developers, even if it forces
        Carbon's safety design to take a different approach.

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

-   Each build mode will prioritize performance and safety differently:

    -   The [debug build mode](#debug) will produce development-focused binaries
        that prioritize fast iteration on code with safety checks that assist in
        identification and debugging of errors.

    -   The [performance build mode](#performance) will produce release-focused
        binaries that prioritize performance over safety.

    -   The [hardened build mode](#hardened) will produce release-focused
        binaries that prioritize safety that is resistant to attacks at the cost
        of performance.

-   Safety checks should try to be identical across build modes.

    -   There will be differences, typically due to performance overhead and
        detection rate trade-offs of safety check algorithms.

-   The number of build modes will be limited, and should be expected to remain
    at the named three.

    -   Most developers will use two build modes in their work: debug for
        development, and either performance or hardened for releases.

    -   It's important to focus on checks that are cheap enough to run as part
        of normal development. Users are not expected to want to run additional
        development build modes for additional sanitizers.

    -   Limiting the number of build modes simplifies support for both Carbon
        maintainers, who can focus on a more limited set of configurations, and
        Carbon developers, who can easily choose which is better for their
        use-case.

-   Each distinct safety-related build mode (debug, performance, and hardened)
    cannot be combined with others in the same binary.

    -   Cross-binary interfaces will exist in Carbon, and will need to be used
        by developers interested in combining libraries built under different
        build modes.

-   Although runtime safety checks should prevent logic errors from turning into
    security vulnerabilities, the underlying logic errors will still be bugs.
    For example, some safety checks would result in application termination;
    this prevents execution of unexpected code and still needs to be fixed.

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
will adopt a strategy based on three build modes that target key use-cases.

#### Debug

The debug build mode targets developers who are iterating on code and running
tests. It will emphasize detection and debugability, especially for safety
issues.

It needs to perform well enough to be run frequently by developers, but will
make performance sacrifices to catch more safety issues. This mode should have
runtime checks for the most common safety issues, but it can make trade-offs
that improve performance in exchange for less frequent, but still reliable,
detection. Developers should do most of their testing in this build mode.

The debug build mode will place a premium on the debugability of safety
violations. Where safety checks rely on hardening instead of guaranteed safety,
violations should be detected with a high probability per single occurrence of
the bug. Detected bugs will be accompanied by a detailed diagnostic report to
ease classification and root cause identification.

#### Performance

The performance build mode targets the typical application that wants high
performance from Carbon code, where performance considers processing time,
memory, and disk space. Trade-offs will be made that maximize the performance.

Only safety techniques that don't measurably impact application hot path
performance will be enabled by default. This is a very high bar, but is crucial
for meeting Carbon's performance goals, as well as allowing migration of
existing C++ systems which may not have been designed with Carbon's safety
semantics in mind.

#### Hardened

The hardened build mode targets applications where developers want strong safety
against attacks in exchange for worse performance. It will work to prevent
attacks in ways that
[attackers cannot work around](#probabilistic-techniques-likely-cannot-stop-attacks),
even if it means using techniques that create significant performance costs.

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

-   Implementing coverage-directed fuzz testing to discover bugs outside of
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

## Caveats

### Probabilistic techniques likely cannot stop attacks

It's expected that probabilistic techniques that can be applied at the language
level are attackable through a variety of techniques:

-   The attacker might be able to attack repeatedly until it gets through.
-   The attacker may be able to determine when the attack would be detected and
    only run the attack when it would not be.
-   The attacker might be able control the test condition to make detection much
    less likely or avoid detection completely. For example, if detection is
    based on the last 4 bits of a memory address, an attacker may be able to
    generate memory allocations, viewing the address and only attacking when
    there's a collision.

Hardware vulnerabilities may make these attacks easier than they might otherwise
appear. Future hardware vulnerabilities are difficult to predict.

Note this statement focuses on what can be applied to the language level. Using
a secure hash algorithm, such as SHA256, may be used to offer probabilistic
defense in other situations. However, the overhead of a secure hash algorithm's
calculation is significant in the context of most things that Carbon may do at
the language level.

Combining these issues, although it may seem like a probabilistic safety check
could be proven to reliably detect attackers, it's likely infeasible to do so.
For the various build modes, this means:

-   The debug build mode will not typically be accessible to attackers, so where
    a probabilistic technique provides a better developer experience, it will be
    preferred.
-   The performance build mode will often avoid safety checks in order to reach
    peak performance. As a consequence, even the weak protection of a
    probabilistic safety check may be used in order to provide _some_
    protection.
-   The hardened build mode will prefer non-probabilistic techniques that
    _cannot_ be attacked.

## Alternatives considered

### Guaranteed memory safety programming models

Multiple approaches that would offer guaranteed memory safety have been
considered, mainly based on other languages which offer related approaches.
Carbon will likely rely more on error detection and hardening because of what
the models would mean for Carbon's performance and C++ migration language goals.

#### Guaranteed compile-time memory safety using borrow checking

Rust offers a good example of an approach for compile-time safety based on
borrow checking, which provides guaranteed safety. For code which can't
implement borrow checking, runtime safety using reference counting is available
and provides reliable error detection. This approach still allows for
[`unsafe` blocks](https://doc.rust-lang.org/rust-by-example/unsafe.html), as
well as types that offer runtime safety while wrapping `unsafe` interfaces.

Carbon could use a similar approach for guaranteed safety by default.

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

#### Guaranteed run-time memory safety using reference counting

[Reference counting](https://en.wikipedia.org/wiki/Reference_counting) is a
common memory safety model, with Swift as a popular example.

Advantages:

-   Simple model for safety, particularly as compared with Rust.
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

In order to mitigate the performance overhead, Swift does have a proposal to add
an option for unique ownership, although the specifics are not designed yet. The
unique ownership approach is expected to require unowned and unsafe access, so
it would not considered to improve the safety trade-offs.

Swift was designated by Apple as the replacement for Objective-C. The safety
versus performance trade-offs that it makes fit Apple's priorities. Carbon's
performance goals should lead to different trade-off decisions with a higher
priority on peak performance, which effectively rules out broad use of reference
counting.

#### Guaranteed run-time memory safety using garbage collection

[Garbage collection](<https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)>)
is a common memory safety model, with Java as a popular example.

Advantages:

-   This approach is among the most robust and well studied models, with decades
    of practical usage and analysis for security properties.
-   Extremely suitable for efficient implementation on top of a virtual machine,
    such as the JVM.

Disadvantages:

-   Extremely high complexity to fully understand the implications of complex
    cases like data races.
-   Performance overhead is significant in terms of what Carbon would like to
    consider.
    -   Garbage collection remains a difficult performance problem, even for the
        JVM and its extensive optimizations.
    -   The complexity of the implementation makes it difficult to _predict_
        performance; for example, Java applications experience latency spikes
        when garbage collection runs.

Java is a good choice for many applications, but Carbon is working to focus on a
set of performance priorities that would be difficult to achieve with a garbage
collector.

### Build mode names

The build mode concepts are difficult to name. Other names that were evaluated,
and are ultimately similar, are:

-   "Debug" is a common term for the intended use of this build mode. Also,
    tooling including Visual Studio frequently uses the debug term for
    describing similar.

    -   "Development" was also considered, but this term is less specific and
        would be better for describing all non-release builds together. For
        example, a "fast build" mode might be added that disables safety checks
        to improve iteration time, like might be controlled by way of C++'s
        `NDEBUG` option.

-   "Performance" aligns with the phrasing of the language performance goal.

    -   "Optimized" implies that other modes would not be fully optimized, but
        hardened should be optimized.

    -   "Fast" would suggest that speed is the only aspect of performance being
        optimizing for, but "performance" also optimizes for memory usage and
        binary size.

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

### Performance versus safety in the hardened build mode

The performance cost of safety techniques are expected to be non-linear with
respect to detection rates. For example, a particular vulnerability such as heap
use-after-free may be detectable with 99% accuracy at 20% performance cost, but
100% accuracy at 50% performance cost. At present, build modes should be
expected to evaluate such a scenario as:

-   The debug build mode would choose the 99% accurate approach.
    -   Detecting safety issues is valuable for debugging.
    -   The probabilistic detection rate won't meaningfully affect accuracy of
        tests.
    -   The lower performance cost improves developer velocity.
-   The performance build mode would decline detection.
    -   Safety checks with a measurable performance cost should be declined.
-   The hardened build mode would choose the 100% accurate approach.
    -   Safety must be non-probabilistic in order to reliably prevent attacks.
    -   Significant performance hits are acceptable.
    -   This means the hardened build mode may be slower than the debug build
        mode.

In order to achieve better performance, the hardened build mode could make
trade-offs closer to the debug build mode. Rather than relying on
non-probabilistic techniques, it could instead offer a probability-based chance
of detecting a given attack.

Advantages:

-   Probabilistic safety should come at lower performance cost (including CPU,
    memory, and disk space).
    -   This will sometimes be significant, and as a result of multiple checks,
        could be the difference between the hardened build mode being 50% slower
        than the performance build mode and being 200% slower.

Disadvantages:

-   [Probabilistic techniques likely cannot stop attacks](#probabilistic-techniques-likely-cannot-stop-attacks).
    -   Attackers may be able to repeat attacks until they succeed.
    -   The variables upon which the probability is based, such as memory
        addresses, may be manipulable by the attacker. As a consequence, a
        determined attacker may be able to manipulate probabilities and not even
        be detected.

Although performance is
[Carbon's top goal](../goals.md#language-goals-and-priorities), the hardened
build mode exists to satisfy developers and environments that value safety more
than performance. The hardened build mode will rely on non-probabilistic safety
at significant performance cost because other approaches will be insufficient to
guard against determined attackers.

### Add more build modes

More build modes could be added to this principle, or the principle could
encourage the idea that specific designs may add more.

To explain why three build modes:

-   The concept of debug and release (sometimes called opt) are common. For
    example, in
    [Visual Studio](https://docs.microsoft.com/en-us/visualstudio/debugger/how-to-set-debug-and-release-configurations?view=vs-2019).
    In Carbon, this could be considered to translate to the "debug" and
    "performance" build modes by default.

-   The hardened build mode is added in order to emphasize security. Although
    hardened could be implemented as a set of options passed to the standard
    release build mode, the preference is to focus on it as an important
    feature.

An example of why another build mode may be needed is
[ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html), which is
noted as having 5-15x slowdown and 5-10x memory overhead. This is infeasible for
normal use, but could be useful for some users in a separate build mode. A
trade-off that's possible for Carbon is instead using an approach similar to
[KCSAN](https://github.com/google/ktsan/wiki/KCSAN) which offers relatively
inexpensive but lower-probability race detection.

Although options to these build modes may be supported to customize deployments,
the preference is to focus on a small set and make them behave well. For
example, if a separate build mode is added for ThreadSanitizer, it should be
considered a temporary solution until it can be merged into the debug build
mode.

Advantages:

-   Grants more flexibility for using build modes as a solution to problems.
    -   With safety checks, this would allow providing safety checks that are
        high overhead but also high detection rate as separate build modes.
    -   With other systems, there could be non-safety performance versus
        behavior trade-offs.

Disadvantages:

-   Having standard modes simplifies validation of interactions between various
    safety checks.
    -   Safety is the only reason that's been considered for adding build modes.
-   As more build modes are added, the chance of developers being confused and
    choosing the wrong build mode for their application increases.

Any long-term additions to the set of build modes will need to update this
principle, raising the visibility and requiring more consideration of such an
addition. If build modes are added for non-safety-related reasons, this may lead
to moving build modes out of the safety strategy.

**Experiment**: This can be considered an experiment. Carbon may eventually add
more than the initial three build modes, although the reticence to add more is
likely to remain.
