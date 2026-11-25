# Defining program behavior

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8161)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   ["As-if" rule](#as-if-rule)
    -   [Erroneous behavior](#erroneous-behavior)
        -   [Mutating an object bound to a value](#mutating-an-object-bound-to-a-value)
    -   [Undefined behavior](#undefined-behavior)
    -   [Safety assumptions, safety modes, and C++ interoperability](#safety-assumptions-safety-modes-and-c-interoperability)
        -   [Mixing safety modes and safety annotations](#mixing-safety-modes-and-safety-annotations)
        -   [Reasonable C++ code and compromised safety checks](#reasonable-c-code-and-compromised-safety-checks)
    -   [Changes to existing undefined behavior cases](#changes-to-existing-undefined-behavior-cases)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow the compiler to optimize assuming erroneous behavior does not occur](#allow-the-compiler-to-optimize-assuming-erroneous-behavior-does-not-occur)
    -   [Allow safety assumptions and strict mode to introduce undefined behavior](#allow-safety-assumptions-and-strict-mode-to-introduce-undefined-behavior)
    -   [Eliminate undefined behavior completely](#eliminate-undefined-behavior-completely)
    -   [Make erroneous behavior ordinary defined behavior](#make-erroneous-behavior-ordinary-defined-behavior)

<!-- tocstop -->

## Abstract

This proposal defines Carbon's foundational model for program behavior in terms
of three concepts: the _"as-if" rule_, _erroneous behavior_, and _undefined
behavior_, along with how undefined behavior interacts with Carbon's safety
modes and C++ interoperability. Fully strict Carbon code (without `unsafe`
operations or C++) has no undefined behavior; execution within Carbon without
C++ can only have undefined behavior in permissive mode or after passing through
an `unsafe`-marked operation. Erroneous behavior identifies bugs that
nonetheless have defined, bounded behavior: unlike undefined behavior, the
compiler must not optimize under the assumption that erroneous behavior does not
occur, and may optimize only on the basis of a proof under the "as-if" rule.
Similarly, adding safety annotations or switching to strict mode never
introduces undefined behavior -- the compiler may optimize based on safety
information, but not in ways that introduce undefined behavior if unsafe,
permissive, or C++ code compromises those safety assumptions -- and Carbon code
always has at most the undefined behavior of corresponding C++. Accordingly,
this proposal changes existing cases previously specified as undefined behavior
in performance builds -- signed integer arithmetic overflow, integer division by
zero, out-of-range bit shifts, and operations on unformed objects -- into
erroneous behavior.

## Problem

Carbon's language design and safety strategy rely on the "as-if" rule, erroneous
behavior, and undefined behavior, but the project lacks a single explicit
definition of these concepts and the constraints they place on compilers:

-   The [value expressions design](/docs/design/values.md#value-expressions),
    introduced in
    [proposal #2006](https://github.com/carbon-language/carbon-lang/pull/2006),
    specifies that mutating an object bound to an active value expression is
    "erroneous" rather than undefined behavior, and recorded an open issue
    noting that Carbon needs a common definition of erroneous behavior to cite
    there and elsewhere.
-   During
    [review of proposal #5434](https://github.com/carbon-language/carbon-lang/pull/5434/files#r2217033493),
    contributors noted that Carbon had only defined erroneous behavior
    implicitly and agreed that a follow-up proposal should establish an explicit
    definition.
-   Earlier proposals that predated Carbon's updated
    [safety strategy](/proposals/p005914-updating-carbon-s-safety-strategy.md)
    and
    [leads issue #5524](https://github.com/carbon-language/carbon-lang/issues/5524)
    allowed optimizers to treat several programming errors as undefined behavior
    in "performance builds" -- including signed integer overflow and integer
    division by zero
    ([proposal #1083](https://github.com/carbon-language/carbon-lang/pull/1083)),
    out-of-range bit shifts
    ([proposal #1191](https://github.com/carbon-language/carbon-lang/pull/1191)),
    and operations on unformed objects
    ([proposal #257](https://github.com/carbon-language/carbon-lang/pull/257)).
    Those design documents also carried `TODO` comments to unify their
    descriptions in a common place.
-   The [safety design](/docs/design/safety/README.md) and
    [safety terminology](/docs/design/safety/terminology.md) refer to both
    erroneous behavior and undefined behavior when defining safe code, unsafe
    code, and build modes.
-   During incremental migration from C++ through
    [permissive Carbon](/docs/design/safety/README.md#safety-modes) to
    [strict Carbon](/docs/design/safety/README.md#safety-modes), codebases mix
    strict Carbon, permissive Carbon, `unsafe` operations, and C++. Without
    explicit rules specifying where undefined behavior can arise and prohibiting
    safety annotations or strict-mode checks from introducing new undefined
    behavior, adding safety checks could introduce unfamiliar undefined-behavior
    hazards across mixed-mode and C++ boundaries rather than strictly improving
    safety.

Without a shared definition and consistent classification of these error cases,
language proposals risk specifying conflicting rules for buggy programs,
compiler implementers lack a clear boundary between permitted and forbidden
optimizations, and developers lack a uniform mental model for what safe and
unsafe Carbon code guarantee.

## Background

Carbon's approach builds on concepts from C++ and prior Carbon design decisions:

-   **C++ "as-if" rule**:
    [C++'s "as-if" rule](https://en.cppreference.com/w/cpp/language/as_if.html)
    permits any code transformation that does not change the observable behavior
    of a program.
-   **C++ undefined behavior**:
    [C++ undefined behavior](https://en.cppreference.com/w/cpp/language/ub.html)
    places no requirements on the program, allowing compilers to assume that
    undefined behavior never occurs and to optimize accordingly.
-   **C++ erroneous behavior**:
    [C++ Proposal P2795: Erroneous behaviour for uninitialized reads](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2795r2.html)
    introduced _erroneous behavior_ to describe program behavior that is
    well-defined yet still considered a bug that implementations and tools are
    encouraged to diagnose.
-   **Prior Carbon proposals and leads decisions**:
    -   [Proposal #257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257),
        [proposal #1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083),
        and
        [proposal #1191: Bitwise and shift operators](https://github.com/carbon-language/carbon-lang/pull/1191)
        predated Carbon's current safety strategy and allowed performance builds
        to treat operations on unformed objects, signed integer overflow,
        integer division by zero, and out-of-range bit shifts as undefined
        behavior.
    -   [Proposal #2006: Values, variables, pointers, and references](https://github.com/carbon-language/carbon-lang/pull/2006)
        used "erroneous" in the
        [value expressions design](/docs/design/values.md#value-expressions) to
        describe bugs that should be diagnosable without introducing unbounded
        undefined behavior.
    -   [Leads issue #5524](https://github.com/carbon-language/carbon-lang/issues/5524)
        ([decision comment](https://github.com/carbon-language/carbon-lang/issues/5524#issuecomment-2923988598))
        decided that mutating objects bound to an active value binding should
        remain erroneous behavior rather than undefined behavior, and
        articulated the rule that Carbon should never optimize on erroneous
        behavior unless the compiler proves it does not occur.
    -   [Proposal #5434: `ref` parameters, arguments, returns and `val` returns](https://github.com/carbon-language/carbon-lang/pull/5434)
        recorded this principle under
        [no optimization on erroneous behavior](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md#no-optimization-on-erroneous-behavior),
        and its
        [review discussion](https://github.com/carbon-language/carbon-lang/pull/5434/files#r2217033493)
        deferred a general definition of erroneous behavior to a dedicated
        proposal.
    -   [Proposal #5914: Updating Carbon's safety strategy](https://github.com/carbon-language/carbon-lang/pull/5914)
        established Carbon's [safety design](/docs/design/safety/README.md),
        introduced
        [strict and permissive safety modes](/docs/design/safety/README.md#safety-modes)
        for incremental migration from C++, simplified build modes to Debug and
        Release, and noted that restrictions on unsafe code should prefer
        erroneous behavior over undefined behavior to reduce risk during
        incremental migration from C++.

## Proposal

We expect the design and implementation of the Carbon language to define the
behavior of valid programs and to reject invalid programs. This definition must
be complete for valid, fully
[strict Carbon](/docs/design/safety/README.md#safety-modes) code (without
`unsafe` operations or C++), with no undefined behavior. Execution within Carbon
(with no C++) can only encounter [undefined behavior](#undefined-behavior) in
[permissive mode](/docs/design/safety/README.md#safety-modes) or after execution
passes through an operation marked `unsafe`, and we work to minimize the set of
undefined behavior.

This proposal establishes standard definitions for three foundational concepts
used across the Carbon language design:

-   The **["as-if" rule](#as-if-rule)** permits optimizations and code
    transformations that preserve observable behavior, subject to a bound on
    peak memory growth while allowing reductions in resource usage.
-   **[Erroneous behavior](#erroneous-behavior)** classifies behavior that is
    incorrect and a bug, yet remains defined. Compilers must not optimize under
    the assumption that erroneous behavior does not occur (they will not
    optimize on unvalidated assumptions that could introduce undefined
    behavior); they may optimize only when they prove it does not occur, under
    the "as-if" rule.
-   **[Undefined behavior](#undefined-behavior)** classifies behavior the
    compiler may assume never occurs. It is always a bug and a security
    vulnerability if reachable, and is admitted only when defining the behavior
    is constructively impossible without undermining the feature or incremental
    C++ migration.

This proposal also establishes the undefined-behavior invariants for
[safety assumptions, safety modes, and C++ interoperability](#safety-assumptions-safety-modes-and-c-interoperability):

-   **At most the undefined behavior of C++**: Permissive Carbon has a subset of
    the undefined behavior of corresponding C++ even with no safety annotations
    (some C++ undefined behavior is diagnosed at compile time, and some becomes
    erroneous behavior), and Carbon code always has at most the undefined
    behavior of corresponding C++.
-   **No undefined behavior from safety assumptions**: Adding safety annotations
    to permissive code or switching to strict mode never introduces undefined
    behavior. The compiler may optimize based on safety information, but not in
    ways that introduce undefined behavior if unsafe, permissive, or C++ code
    compromises those safety assumptions.
-   **Reasonable C++ interop boundary**: Reasonable C++ code following a
    sanitizable set of rules does not compromise Carbon's safety checks. If C++
    undefined behavior or unreasonable (even if well-defined) C++ compromises
    safety checks, Carbon still does not introduce additional undefined behavior
    from its safety assumptions, so Carbon code retains less undefined behavior
    than the equivalent C++.

This proposal also explicitly changes the existing language cases that were
previously allowed to produce undefined behavior in performance builds -- signed
integer arithmetic overflow, integer division by zero, out-of-range bit shifts,
and operations on unformed objects -- to be **erroneous behavior** instead,
eliminating undefined behavior from these operations across all build modes.

To keep the living documentation current, this proposal adds a stub
[program behavior design document](/docs/design/program_behavior.md) with a
`TODO` to fill in the full definitions, updates small affected passages in
existing design documents, and adds `TODO`s where larger updates remain.

## Details

### "As-if" rule

The _"as-if" rule_ is:

> Optimizations and code transformations are allowed if they do not change the
> observable behavior of the program.

This [matches C++](https://en.cppreference.com/w/cpp/language/as_if.html).

The peak memory usage of a program must not be increased beyond a constant
amount that does not vary by program input.

However, optimizations are allowed to decrease resource usage, including
reducing the asymptotic complexity of that usage. For example, it is legal to
collapse a linear-time calculation to a constant-time one:

```carbon
// A compiler may replace this O(n) loop with an O(1) formula because it does
// not change observable behavior (no overflow can occur for any `i16` input).
fn SumUpTo(n: i16) -> i64 {
  let limit: i64 = n;
  var total: i64 = 0;
  var i: i64 = 1;
  while (i <= limit) {
    total += i;
    ++i;
  }
  return total;
}
```

### Erroneous behavior

_Erroneous behavior_ is behavior that is incorrect and a bug, but unlike
[undefined behavior](#undefined-behavior), the compiler must not optimize under
the assumption that it does not occur. If the compiler can prove that erroneous
behavior does not occur, then it may optimize on the basis of that proof under
[the "as-if" rule](#as-if-rule). Turning C++ undefined behavior (such as signed
integer arithmetic overflow) into erroneous behavior ensures that the compiler
does not optimize based on unvalidated assumptions that could introduce
undefined behavior.

Examples of erroneous behavior:

-   Dividing by zero.
-   Signed integer arithmetic overflow.
-   Out-of-range bit-shift amounts (shifting by a negative amount or by at least
    the bit-width of the first operand).
-   Dereferencing a null pointer.
-   Performing operations on an object in an
    [unformed state](/docs/design/README.md#unformed-state) that are only
    defined for fully formed states, after an unsafe cast.
-   Mutating or destroying an object whose type uses a pointer value
    representation, while that object is bound to a value that is in use.

Examples of acceptable results of erroneous behavior:

-   The compiler detects the erroneous behavior at compile time and issues a
    diagnostic.
-   The erroneous behavior is detected at runtime and causes the program to
    halt, possibly with a message.
-   A defined value is produced.
    -   The defined value may or may not be in the domain for its type.
-   A defined value is produced, but a later runtime check detects that
    erroneous behavior occurred and halts the program.

#### Mutating an object bound to a value

When erroneous behavior occurs because code mutates a value that the safety
model specifies is currently immutable, the defined behavior still permits the
compiler to optimize away reloads or to introduce reloads to free up registers:

-   The value is either not reloaded, or if it is reloaded, it is the value in
    memory when the reload occurs.

This illustrates Carbon's general rule for
[optimizing on safety information](#mixing-safety-modes-and-safety-annotations):
the compiler can optimize using the safety model's immutability information
while bounding the possible outcomes to concrete values, without permitting the
compiler to assume that the mutation cannot happen elsewhere in the program or
to introduce undefined behavior if unsafe, permissive, or C++ code compromises
that assumption.

### Undefined behavior

_Undefined behavior_ is behavior the compiler may assume never occurs. It is
always a bug and must be assumed to be a security vulnerability if reachable.

Carbon makes an active effort to define as much behavior as possible. We admit
undefined behavior only because of a constructive inability to define the
behavior -- for example, when defining the behavior would undermine the very
effect of the feature, such as an explicit optimization assumption like "assume
this pointer is never null".

Within Carbon's [safety modes](/docs/design/safety/README.md#safety-modes) and
relative to C++:

-   **Fully strict Carbon code has no undefined behavior**: valid code written
    entirely in [strict Carbon](/docs/design/safety/README.md#safety-modes)
    without `unsafe` operations or C++ has completely defined behavior.
-   **Pure Carbon undefined behavior requires permissive mode or `unsafe`**:
    execution within Carbon, with no C++, can only have undefined behavior in
    [permissive mode](/docs/design/safety/README.md#safety-modes) or after
    execution passes through an operation marked `unsafe`.
-   **Permissive Carbon has a subset of C++'s undefined behavior**: even without
    any safety annotations, migrating C++ to permissive Carbon reduces undefined
    behavior:
    -   Some C++ undefined behavior is diagnosed at compile time (for example,
        One Definition Rule (ODR) violations).
    -   Some C++ undefined behavior becomes
        [erroneous behavior](#erroneous-behavior) (for example, signed integer
        arithmetic overflow), meaning the compiler will not optimize based on
        unvalidated assumptions that could introduce undefined behavior.
-   **At most the undefined behavior of corresponding C++**: Carbon code always
    has at most the undefined behavior of the corresponding C++ code.

Examples of undefined behavior:

-   Violating an explicitly stated optimization assumption.
-   A data race.

### Safety assumptions, safety modes, and C++ interoperability

#### Mixing safety modes and safety annotations

Adding safety annotations to
[permissive Carbon](/docs/design/safety/README.md#safety-modes) code or
switching from permissive mode to
[strict Carbon](/docs/design/safety/README.md#safety-modes) never introduces
undefined behavior.

The compiler may optimize based on safety information (such as safety
annotations or compile-time safety checks), but **not in ways that introduce
undefined behavior** if `unsafe` Carbon, permissive Carbon, or C++ code
compromises those safety assumptions.

This differs deliberately from Rust, where unsafe code must uphold additional
unchecked safety invariants unfamiliar to many C++ developers -- such as the
[Rustonomicon](https://doc.rust-lang.org/nomicon/) rule making it undefined
behavior to alias a mutable reference. Carbon does not turn violations of its
safety annotations into undefined behavior, because incremental C++ migration
requires large codebases to mix strict Carbon, permissive Carbon, `unsafe`
operations, and C++ without introducing new undefined behavior at every
boundary.

This choice intentionally gives up some performance optimization opportunities
that would depend on assuming safety annotations are never violated. Bounded
optimizations that cannot introduce undefined behavior -- such as eliding or
inserting reloads when
[mutating an object bound to a value](#mutating-an-object-bound-to-a-value)
-- remain allowed, and developers can still write explicit `unsafe`
optimization assumptions when peak performance requires an unproven invariant.

#### Reasonable C++ code and compromised safety checks

When Carbon code interoperates with C++, Carbon expects interoperating C++ code
to follow a set of rules so that it does not compromise Carbon's safety checks:

-   Code violating these rules should be recognizable as either buggy or
    dangerous by C++ developers. For example,
    [`delete this;`](https://isocpp.org/wiki/faq/freestore-mgmt#delete-this) is
    considered dangerous in C++, and Carbon assumes interoperating C++ code will
    not do it.
-   It should be reasonable to build a sanitizer to detect violations of these
    rules (similar to Clang's `-fbounds-safety`). Such a sanitizer can be built
    reactively if needed in practice.

C++ undefined behavior, or unreasonable (rule-violating) C++ code even if
well-defined in C++, may compromise Carbon's safety checks. However, because
Carbon never introduces undefined behavior based on its safety assumptions, even
when safety checks have been compromised, Carbon code still has less undefined
behavior than the equivalent C++.

### Changes to existing undefined behavior cases

Earlier Carbon proposals defined several runtime programming errors using the
pre-[#5914](https://github.com/carbon-language/carbon-lang/pull/5914)
three-mode model (`development`, `performance`, and `hardened`), allowing
compilers in `performance` builds to assume those errors never occurred
(undefined behavior). Because fully strict Carbon code must have no undefined
behavior, permissive Carbon converts C++ undefined behavior into erroneous
behavior where practical instead of optimizing on unvalidated assumptions, and
Carbon's release build mode does not opt into undefined behavior for these
errors, this proposal explicitly changes each of those existing cases from
undefined behavior to [erroneous behavior](#erroneous-behavior):

-   **Signed integer arithmetic overflow** (including `char` arithmetic
    overflow) and **signed or unsigned integer division and remainder by zero**,
    previously specified as undefined behavior in performance builds by
    [proposal #1083](https://github.com/carbon-language/carbon-lang/pull/1083)
    and documented in
    [arithmetic](/docs/design/expressions/arithmetic.md#overflow-and-other-error-conditions)
    and [integer types](/docs/design/README.md#integer-types).
-   **Out-of-range bit-shift amounts**, previously specified as undefined
    behavior in performance builds by
    [proposal #1191](https://github.com/carbon-language/carbon-lang/pull/1191)
    and documented in
    [bitwise and shift operators](/docs/design/expressions/bitwise.md#integer-types).
-   **Operations on an object in an unformed state** other than assignment or
    destruction, previously allowed to lead to undefined behavior in performance
    builds by
    [proposal #257](https://github.com/carbon-language/carbon-lang/pull/257) and
    documented in [unformed state](/docs/design/README.md#unformed-state).

In all of these cases, the compiler must no longer optimize under the assumption
that the error does not occur in any build mode unless it proves that the error
cannot occur (under [the "as-if" rule](#as-if-rule)). In
[debug builds](/docs/design/safety/README.md#build-modes), detectable erroneous
behavior halts with a diagnostic; in
[release builds](/docs/design/safety/README.md#build-modes), it produces one of
the acceptable bounded results for erroneous behavior (such as halting or
producing a defined value).

## Rationale

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    -   Carbon aims to maintain a formal specification alongside its reference
        implementation. Explicitly defining the "as-if" rule, erroneous
        behavior, and undefined behavior establishes shared foundational
        semantics for both the specification and the toolchain.
    -   Separating erroneous behavior from valid defined behavior lets
        compilers, static analyzers, and sanitizers distinguish bugs from
        intentional code and diagnose them.
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software):
    -   The "as-if" rule allows aggressive optimizations, including asymptotic
        reductions in resource usage, while bounding peak memory growth to a
        constant independent of program input to keep performance predictable.
    -   Although disallowing optimizations that introduce undefined behavior
        based on safety assumptions gives up some optimization opportunities,
        bounded optimizations based on safety information (such as eliding or
        inserting reloads for values treated as immutable by the safety model)
        remain permitted, and explicit `unsafe` optimization assumptions (backed
        by undefined behavior if violated) preserve full performance control
        when the compiler cannot statically prove an invariant.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    -   Carbon's goals explicitly call for clearly and simply specified behavior
        whenever possible, avoiding undefined behavior due to its high cost, and
        requiring a strong rationale whenever behavior is left undefined.
    -   Minimizing undefined behavior and ensuring fully strict Carbon code has
        no undefined behavior makes Carbon programs easier to reason about and
        debug.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms):
    -   Ensuring fully strict Carbon code has no undefined behavior, restricting
        pure-Carbon undefined behavior to permissive mode or execution after an
        `unsafe`-marked operation, and preventing safety annotations from
        introducing new undefined behavior uphold the requirement that unsafe or
        risky aspects of Carbon code are explicit and auditable.
    -   Changing arithmetic overflow, division by zero, out-of-range shifts, and
        unformed-object errors from undefined behavior to erroneous behavior
        prevents compiler optimizations from amplifying bugs into unpredictable,
        non-local vulnerabilities, while preserving the ability to catch those
        bugs through compile-time diagnostics, debug builds, and fuzzing.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    -   Matching C++'s "as-if" rule and building on C++'s erroneous behavior
        concept
        ([P2795](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2795r2.html))
        keeps Carbon's execution model familiar to C++ developers.
    -   Guaranteeing that Carbon code always has at most the undefined behavior
        of corresponding C++, and that migrating to permissive Carbon, adding
        safety annotations, or switching to strict Carbon never introduces new
        undefined behavior, ensures every incremental migration step improves
        safety without introducing unfamiliar unsafe undefined-behavior rules
        (as in Rust).
    -   Scoping C++ interop safety assumptions to rules whose violations are
        recognizable as buggy or dangerous to C++ developers and checkable by a
        sanitizer keeps mixed C++/Carbon codebases practical while bounding the
        blast radius if C++ code compromises safety checks.
    -   Preferring erroneous behavior over undefined behavior reduces the risk
        of subtle bugs across unsafe Carbon and C++ interop boundaries during
        incremental migration, aligning with
        [proposal #5914](/proposals/p005914-updating-carbon-s-safety-strategy.md).

## Alternatives considered

### Allow the compiler to optimize assuming erroneous behavior does not occur

We could allow the compiler to optimize under the assumption that erroneous
behavior does not occur (for example, in release builds), retaining the earlier
approach taken for arithmetic overflow, division by zero, out-of-range bit
shifts, and unformed objects in
[proposal #257](https://github.com/carbon-language/carbon-lang/pull/257),
[proposal #1083](https://github.com/carbon-language/carbon-lang/pull/1083), and
[proposal #1191](https://github.com/carbon-language/carbon-lang/pull/1191).

-   **Advantages**: Allows optimizations that depend on unproven invariants --
    such as assuming signed integer arithmetic never overflows or assuming an
    object bound to a value expression is never mutated -- without requiring the
    compiler to prove them or developers to write an explicit assumption.
-   **Disadvantages**: Allowing "as-if" optimizations without a proof is
    precisely what makes undefined behavior dangerous. If the erroneous
    condition is reachable, the optimizer can remove checks or produce arbitrary
    non-local misbehavior and security vulnerabilities, erasing the practical
    distinction between erroneous behavior and undefined behavior and violating
    the guarantee that safe Carbon code has no undefined behavior.
-   **Core of the decision**: As decided in
    [leads issue #5524](https://github.com/carbon-language/carbon-lang/issues/5524#issuecomment-2923988598)
    and recorded in
    [proposal #5434](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md#no-optimization-on-erroneous-behavior),
    if the compiler can prove that erroneous behavior does not occur, it can
    already optimize on that proof under the "as-if" rule. When no proof exists,
    optimizing on an unproven assumption contradicts Carbon's
    [practical safety](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    and
    [understandability](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    goals. When unproven assumptions are needed for peak performance, they
    should be stated explicitly as `unsafe` optimization assumptions.

### Allow safety assumptions and strict mode to introduce undefined behavior

We could treat violations of Carbon's safety annotations or strict-mode
invariants (such as aliasing constraints when reached from permissive Carbon,
`unsafe` Carbon, or C++) as undefined behavior, similar to Rust's aliasing rules
in the [Rustonomicon](https://doc.rust-lang.org/nomicon/), so the compiler could
freely emit optimizations backed by undefined behavior from safety information
alone.

-   **Advantages**: Unlocks additional compiler optimization opportunities
    driven by safety annotations and strict-mode invariants without requiring
    explicit `unsafe` optimization assumptions or whole-program proofs.
-   **Disadvantages**: Adding safety annotations to permissive code or switching
    to strict mode could introduce new undefined behavior whenever
    interoperating C++, permissive Carbon, or `unsafe` code compromises a safety
    assumption -- even when the corresponding C++ code had no undefined
    behavior. This forces C++ developers migrating incrementally to track
    unfamiliar, unchecked safety constraints across mixed-mode and C++
    boundaries and undermines the property that every migration step strictly
    improves safety.
-   **Core of the decision**: Carbon's
    [C++ interoperability and incremental migration](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    and
    [practical safety](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    goals require that adding safety checks or switching modes never introduces
    new undefined behavior and that Carbon code always has at most the undefined
    behavior of corresponding C++. Carbon therefore accepts giving up those
    optimization opportunities that depend on undefined behavior, while still
    allowing bounded optimizations based on safety information that cannot
    introduce undefined behavior and providing explicit `unsafe` optimization
    assumptions when needed.

### Eliminate undefined behavior completely

We could disallow undefined behavior entirely across both safe and unsafe Carbon
code, requiring every operation in the language to have defined (or erroneous)
behavior.

-   **Advantages**: Provides the simplest possible execution model with no
    unbounded undefined behavior anywhere in the language, eliminating the risk
    of miscompilation based on undefined behavior even in `unsafe` code.
-   **Disadvantages**: Constructively precludes language features whose entire
    purpose depends on undefined behavior if violated -- most notably explicit
    optimization assumptions such as assuming a pointer is never null -- as well
    as low-overhead unsynchronized concurrency where preventing data races
    everywhere without synchronization is infeasible. This conflicts with
    Carbon's top-priority
    [performance-critical software](/docs/project/goals.md#performance-critical-software)
    goal of giving developers full performance control without needing a
    lower-level language.
-   **Core of the decision**: Carbon eliminates undefined behavior from fully
    strict Carbon code (without `unsafe` operations or C++) and minimizes it
    everywhere else, but retains undefined behavior where necessary in
    permissive mode and for the narrow set of `unsafe` operations where defining
    the behavior would constructively undermine the feature.

### Make erroneous behavior ordinary defined behavior

Rather than introducing _erroneous behavior_ as a distinct category, we could
give all such operations ordinary, valid defined behavior (for example, always
defining signed integer overflow to wrap or defining operations on unformed
objects to yield a valid value).

-   **Advantages**: Reduces the number of behavioral categories in the language
    specification to two (defined behavior and undefined behavior) and gives
    every operation without undefined behavior deterministic semantics that
    programs can rely on.
-   **Disadvantages**: Makes programming bugs indistinguishable from intentional
    code. Compilers, static analyzers, fuzzers, and debug builds could not
    diagnose or halt on these mistakes without rejecting or aborting valid
    programs, weakening Carbon's
    [practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    and
    [language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem).
    It also degrades readability because readers cannot tell whether an
    operation such as signed overflow was intended or accidental, as previously
    noted in the
    [value expressions design](/docs/design/values.md#value-expressions) and
    [proposal #1083](/proposals/p001083-arithmetic-expressions.md#guarantee-that-all-integer-arithmetic-is-twos-complement).
-   **Core of the decision**: Separating erroneous behavior from valid defined
    behavior lets Carbon bound the runtime impact of bugs without undefined
    behavior while still classifying them as bugs that tools and checked builds
    are expected to catch and report.
