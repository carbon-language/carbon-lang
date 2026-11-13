# Memory safety intro

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8118)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Ante-inspired Rust-like model with non-exclusive mutable borrows](#ante-inspired-rust-like-model-with-non-exclusive-mutable-borrows)
    -   [Safety-effects model with flow-sensitive checking](#safety-effects-model-with-flow-sensitive-checking)

<!-- tocstop -->

## Abstract

This proposal introduces a high-level overview of Carbon's planned memory safety
model in `/docs/design/safety/README.md`, building on the safety strategy
established in
[proposal #5914](https://github.com/carbon-language/carbon-lang/pull/5914).

Detailed designs and syntax for individual mechanisms will follow in subsequent
proposals.

## Problem

Carbon needs to enforce memory safety to achieve its
["practical safety and testing mechanisms" goal](/docs/project/goals.md#practical-safety-and-testing-mechanisms).

To meet our goals, this strategy needs to support:

-   High performance, by checking as much as possible at compile-time instead of
    runtime, and supporting allocations without a runtime garbage collector.
-   Expressiveness for representing C++ code patterns so they may be
    migrated, including non-exclusive mutable pointers.
-   Interop with unsafe code including C++ without introducing new
    sources of undefined behavior.

## Background

Our safety strategy has been laid out in two proposals:

-   [#196: Language-level safety strategy](https://github.com/carbon-language/carbon-lang/pull/196)
-   [#5914: Updating Carbon's safety strategy](https://github.com/carbon-language/carbon-lang/pull/5914)

We have been iterating on memory safety designs to achieve this strategy in docs
in
[the "Memory safety" folder in Carbon's Google Drive](https://drive.google.com/drive/folders/1Wenw_38k-dKKk81feVzhsbaRYDrOvJag).

## Proposal

We propose documenting the high-level architecture of Carbon's memory safety
model in
[`/docs/design/safety/README.md`](/docs/design/safety/README.md#carbons-memory-safety-model),
replacing the provisional direction in
[#5914](p005914-updating-carbon-s-safety-strategy.md#direction-for-temporal-and-data-race-memory-safe-type-system-model)
with the hybrid model introduced in
[Safety Unit No. 25: hybrid flow-sensitive and fixed-by-types permissions](https://docs.google.com/document/d/1snTRAXs8AYGw0TCmKeiZQMR8suhLbpuhWJFbEohOf9Y/edit).

This solves the three core requirements from the ["problem" section](#problem):

-   **High performance without garbage collection**: Type, temporal, and
    data-race safety, along with immutability of borrowed values, are enforced
    at compile time without requiring reference counting or a runtime garbage
    collector. Initialization safety combines compile-time tracking with
    per-type unformed states, and spatial safety uses low-overhead runtime
    bounds checking in standard library types.
-   **Expressivity for C++ migration**: Restricting exclusivity to single
    ownership rather than mutable pointer types allows non-exclusive mutable
    pointers without multiplying incompatible pointer kinds. Flow-sensitivity
    increases precision allowing more code to be shown safe. Data-race safety
    builds on Clang's `-Wthread-safety` model of mutexes and guarded places.
-   **Interop with unsafe C++ without new undefined behavior**: Compile-time
    safety annotations are erased before lowering to preserve separate
    compilation and ABI compatibility, and code generation does not assume
    non-local aliasing or exclusivity invariants that unsafe or unannotated C++
    code could violate.

This proposal establishes the overall model and how its mechanisms fit
together; concrete rules and syntax for each component will be specified in
follow-up proposals.

## Details

See the
[Carbon's memory safety model](/docs/design/safety/README.md#carbons-memory-safety-model)
section added to
[`/docs/design/safety/README.md`](/docs/design/safety/README.md) in this pull
request for the specifics of the proposed model, including:

-   [Compile-time enforcement](/docs/design/safety/README.md#compile-time-enforcement)
    mechanisms (places and place sets, alias tracking, safety effects,
    ownership, type and place capabilities, flow-sensitivity, and erasure before
    lowering).
-   How Carbon enforces each category of safety:
    [type safety](/docs/design/safety/README.md#type-safety),
    [initialization safety](/docs/design/safety/README.md#initialization-safety),
    [spatial safety](/docs/design/safety/README.md#spatial-safety),
    [temporal safety](/docs/design/safety/README.md#temporal-safety),
    [data-race safety](/docs/design/safety/README.md#data-race-safety), and
    [protecting immutable borrows from mutation](/docs/design/safety/README.md#protecting-immutable-borrows-from-mutation).

## Rationale

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    Compiler-checked memory safety provides important guard rails for code
    generated by AI, reducing bugs and security vulnerabilities.
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software):
    Using compile-time annotations and checks where possible instead of runtime
    checks and garbage collection reduces the performance penalty for safety.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Safety contracts written in the code need to be understandable. Having them
    be in the code rather than comments means they are enforced by the compiler
    and remain up to date.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms):
    This proposal's primary goal is to deliver practical safety.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    The memory safety model prioritizes expressivity in order to represent C++
    code patterns to support migration to strict Carbon. This is in
    addition to the permissive mode that was already proposed in
    [proposal #5914 "Updating Carbon's safety strategy"](https://github.com/carbon-language/carbon-lang/pull/5914)
    to allow adding safety annotations incrementally in that migration. It also
    must not introduce undefined behavior when interoperating with unsafe code,
    to avoid regressing safety during the transition.

## Alternatives considered

Broader alternatives such as relying on runtime garbage collection or adopting
Rust's exclusive-mutability borrow model unchanged were previously considered
and rejected in
[#5914](p005914-updating-carbon-s-safety-strategy.md#alternatives-considered).
Before arriving at the proposed hybrid model in
[Safety Unit No. 25: hybrid flow-sensitive and fixed-by-types permissions](https://docs.google.com/document/d/1snTRAXs8AYGw0TCmKeiZQMR8suhLbpuhWJFbEohOf9Y/edit),
we explored two other compile-time models in depth in the
["Memory safety" Google Drive folder](https://drive.google.com/drive/folders/1Wenw_38k-dKKk81feVzhsbaRYDrOvJag).

### Ante-inspired Rust-like model with non-exclusive mutable borrows

Initially developed for the July 2025 safety summit in
[Carbon Language - Temporal memory safety (2025 summit)](https://docs.google.com/document/d/1ta7ttfezNxsYdNmxx_cE7t5oMEfGgAN8jkoA_rZBfsc/edit)
and outlined as a directional sketch in
[#5914](p005914-updating-carbon-s-safety-strategy.md#direction-for-temporal-and-data-race-memory-safe-type-system-model),
this model extended a Rust-like borrow checker with non-exclusive mutable
borrows inspired by
[safe, shared mutability in the Ante language](https://antelang.org/blog/safe_shared_mutability/).
Permissions were encoded in distinct pointer and reference capability kinds
spanning three primary modes: exclusive mutable (required for invalidating
operations such as reallocation), non-exclusive mutable, and shared immutable.

Advantages:

-   Builds closely on Rust's proven borrow-checking foundations, giving high
    confidence in soundness and making it a viable backup plan while exploring
    other designs.
-   Encodes permissions directly in pointer and reference types, which composes
    naturally with checked generics, interfaces, and
    [type erasure](/docs/design/generics/terminology.md#type-erasure) without
    separate effect annotations.

Disadvantages:

-   Transitions between exclusive mutable, non-exclusive mutable, and shared
    immutable modes were rigid and difficult to use ergonomically.
-   Combining mutability, exclusivity, and thread-sharing axes led to too many
    distinct pointer and reference types.
-   Lacked expressivity and precision for common C++ patterns: field-level
    borrowing granularity required additional mechanisms such as view types, and
    a single reference kind could not cleanly express different methods on an
    object requiring different capabilities from its fields.

These led to worries about code compositionality. So we put this idea on hold to
be used as a backup in case other ideas did not pan out. If we were to revisit
this approach, we would consider something like
[The Algebra of Loans in Rust](https://nadrieril.github.io/blog/2025/12/21/the-algebra-of-loans-in-rust.html),
which came out later. It explores a promising related direction for increasing
expressivity in Rust-like loan models.

In this proposal's model, exclusivity is only the domain of single ownership
rather than pointer types, allowing fewer and more compatible pointer types
while still gating invalidation on write access to an owner.

### Safety-effects model with flow-sensitive checking

Inspired by group-borrowing and write-tracking proposals
([nmsmith's "An alternative model for 'lifetimes' in Mojo"](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b),
[Evan Ovadia's "Group Borrowing: Zero-Cost Memory Safety with Fewer Restrictions"](https://verdagon.dev/blog/group-borrowing),
and
[Araq's "Write tracking for Nim"](https://nim-lang.org/araq/writetracking.html)),
this model was developed in
[Carbon Language - Memory safety effects](https://docs.google.com/document/d/1VTZYFerXB1gH4e7YARwuHBz-a7qSqyw5omcSw1tTxO0/edit)
and Safety Units 1 through 24 in the
["Memory safety" Google Drive folder](https://drive.google.com/drive/folders/1Wenw_38k-dKKk81feVzhsbaRYDrOvJag).
It tracked aliasing, mutation, initialization, and invalidation flow-sensitively
on places and place sets, and annotated function signatures with fine-grained
safety effects describing what operations a call performed. For example,
mutation was modeled as a write effect on calls rather than restricting types
with `const`, delaying commitment to what capabilities were needed on an object
until calling a method requiring those capabilities.

Advantages:

-   High precision in concrete, non-type-erased code, including distinguishing
    ordinary writes from pointer-invalidating operations and natural field-level
    granularity without view types.
-   Tracking writes as effects rather than `const` type qualifiers reduced code
    duplication for `const`-correctness.
-   Precision could be extended incrementally by introducing additional effect
    kinds.

Disadvantages:

-   Precision degraded sharply across
    [type erasure](/docs/design/generics/terminology.md#type-erasure), including
    checked generics, interfaces, inheritance, and indirect calls
    ([Safety Unit No. 11: type erasure](https://docs.google.com/document/d/1nuNDjDOWhSAsSZ4aWOOYNwwjvDNjG573-1lJ-q0bj8Y/edit)
    and
    [Safety Unit No. 14: interface effects](https://docs.google.com/document/d/1vBgR-1hPExxj5QkqTPBRKIHg6WAY1gH-kfJlx1eGX1g/edit)).
    Workarounds such as `call` and `template` effects
    ([Safety Unit No. 21: call effect](https://docs.google.com/document/d/1BPx9uG8e2LqpzT-z0SJasimbs39MhOtdhwarcWu2418/edit))
    ended up reproducing much of a function's body in its signature (such as
    declaring that `F` calls `G` and therefore has whatever effects `G` has),
    exposed implementation details like call ordering to callers, or weakened
    [definition checking](/docs/design/generics/terminology.md#definition-checking)
    and diagnostics.
-   Expressing full safety contracts by way of effects required a large catalog
    of annotations
    ([Safety Unit No. 19: effect catalog](https://docs.google.com/document/d/1_RemG4UWRir9K5CchAzBtLfJxHdbBo0D3L80mBNqYzY/edit)
    and
    [Safety Unit No. 24: reducing effects](https://docs.google.com/document/d/1sAMSF7rC9XqX2HtoieswvVygjcyzTcVmc4Bsg4XUtuU/edit)),
    making function signatures verbose and complex.
-   Flow-sensitive effect information did not integrate cleanly as an input to
    type-system decisions that determine program semantics, such as overload
    resolution and `impl` selection
    ([Safety Unit No. 23: Interaction of Types and Flow-Checking](https://docs.google.com/document/d/1E1w-bI7LG5geUTcgUA2057CygufYIrv7YGSZMJbDThs/edit)).

The problem with type erasure is that the effects of an operation on a generic
type were not tightly bounded. The type capabilities model of Rust addresses
this by bounding what can be done with the type at all. This allows the calling
code to assume that everything that can be done has been done without too much
loss of precision.

The hybrid model from this proposal attempts to strike a happy medium between
the type capability and effect approaches:

-   We use `const` instead of write effects in order to reduce the proliferation
    of effects in the catalog and how much needs to be written in API contracts.
    It also gives clients of a generic API a way to limit what that API can do
    by what type it is instantiated with.
-   We use ownership as a way to divide up responsibility for places, instead of
    every function being able to produce an effect on anything. This also helps
    the compiler to determine that places can't overlap, limiting invalidations.

The capability to invalidate comes from both of these: invalidation of a place
is only possible if you can write to its owner. This means you can limit
capabilities of a generic API by giving it something `const`, or if it needs to
write but not resize a container, a non-owning view type can be given instead.
