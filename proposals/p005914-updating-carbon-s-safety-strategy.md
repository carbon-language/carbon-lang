# Updating Carbon's safety strategy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5914)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Goals and requirements](#goals-and-requirements)
-   [Proposal](#proposal)
-   [Direction for temporal and data-race memory-safe type system model](#direction-for-temporal-and-data-race-memory-safe-type-system-model)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Defer beginning the concrete safety design](#defer-beginning-the-concrete-safety-design)
    -   [Pursue an alternative strategy towards memory safety specifically](#pursue-an-alternative-strategy-towards-memory-safety-specifically)
    -   [Adopt a memory safety strategy more closely based on Rust](#adopt-a-memory-safety-strategy-more-closely-based-on-rust)

<!-- tocstop -->

## Abstract

Carbon is accelerating and adjusting its safety strategy, specifically to flesh
out its memory safety strategy and reflect simplifying developments in the
safety space.

This proposal replaces the previous directional safety strategy with a new
concrete and updated framework for the safety design. It includes a specific
framework for memory safety, simplified build modes, specific "safety modes",
and terminology.

This proposal also provides a _directional_ suggestion for temporal and
data-race safety specifically.

In addition to fully building out the above directional component, there are
several other aspects of our safety design that will follow in subsequent
proposals. The hope is to establish the initial framework here.

## Problem

Carbon's safety strategy pre-dates the increased importance and urgency of
having a memory safety strategy, and our subsequent efforts to accelerate this
part of Carbon's design. It also was written at a time with deep uncertainty
about the performance costs of both safety and hardening efforts. And it left
major areas as future work that are now needed such as a specific model for
delineating when Carbon code has memory safety enforced.

We need an updated, more concrete, and more precise strategy now that we're
beginning to directly develop safety designs.

## Background

-   [Basic Concepts and Taxonomy of Dependable and Secure Computing](https://doi.org/10.1109%2FTDSC.2004.2)
-   [Secure by Design: Google's Perspective on Memory Safety](https://research.google/pubs/secure-by-design-googles-perspective-on-memory-safety/)
-   [Story-time: C++, bounds checking, performance, and compilers](https://chandlerc.blog/posts/2024/11/story-time-bounds-checking/)

## Goals and requirements

Our end goal for Carbon's memory safety strategy is to allow large-scale,
existing C++ codebases to _incrementally and scalably_ begin writing new code in
a memory-safe language without loss of their existing legacy codebase. Handling
real-world C++ codebases also means handling large-scale dependencies on C++,
but also on C and even Rust.

We plan to support a two step migration process:

1. Highly automated, minimal supervision migration from C++ to a dialect of
   Carbon designed for C++ interop and migration.
2. Incremental refactoring of the Carbon code to adopt memory safe designs,
   patterns, and APIs.

From this framing of our problem statement and end-goal, we can extract detailed
requirements on the memory safety design in Carbon:

-   The ability to write Carbon as a "rigorously memory safe programming
    language" (defined in the "Secure by Design" paper).
    -   Requires temporal, spatial, and type errors to be rejected at compile
        time or detected at runtime with fail-stop behavior.
    -   Requires initialization errors to not have undefined behavior, but
        allows, for example, zero-initialization rather than fail-stop behavior.
    -   This does not require complete elimination of data races, but does
        require
        [addressing temporal memory safety even in the face of concurrency](/docs/design/safety#data-races-versus-unsynchronized-temporal-safety).
    -   We directly refer to this as simply a
        ["memory-safe language"](/docs/design/safety/terminology.md#memory-safe-language)
        in Carbon.
-   Seamless integration between Carbon's memory safe code and the memory unsafe
    code resulting from migrating existing C++.
    -   Safe and unsafe code should be freely mixed, with incremental checking
        of the strictly safe aspects of the code.
-   A smooth, incremental path for this memory unsafe Carbon code to become more
    memory safe over time.
    -   Must resonate with users as _significantly_ more incremental than
        migrating directly to Rust. That is, the granularity of increasing
        safety needs to be significantly smaller than moving an entire file from
        C++ to Rust.
-   Similar level of runtime spatial safety checks as Rust with similar runtime
    overhead.
-   No reference counting or garbage collection by default -- we want to
    preserve C++ (and Rust) precise and explicit control over the lifetime of
    resources.
-   Minimal temporal safety runtime overhead: in aggregate and in real-world
    application benchmarks, any performance overhead would need to be under 1%.
    -   We expect to spend 1-2% of runtime overhead achieving spatial safety,
        and these would be cumulative.
    -   There has been strong, persistent resistance to over 2% runtime overhead
        in broad contexts.
    -   Rust shows that this is achievable using primarily compile time
        techniques.

We also have some less critical but still _highly desirable_ goals:

-   Seamless _safe_ Rust interop -- able to reliably and consistently avoid
    unsafe on the boundary with Rust code
    -   Should be no worse than the best Rust/C++ interop which uses extra
        safety annotations for the C++ API
-   Data race prevention

Whether we fully achieve these secondary goals or not, we need to clearly
demonstrate where we end up and what any remaining gap there is towards these
goals.

## Proposal

See the new [proposed safety design](/docs/design/safety/README.md) and
[safety terminology](/docs/design/safety/terminology.md) for the core proposed
strategy and model.

## Direction for temporal and data-race memory-safe type system model

> Note: this is a _directional_ component of the proposal, and should be treated
> as provisional until dedicated proposals fully establish our model here.

We expect Carbon to tackle temporal and data-race safety through its type
system, much as Rust and strict Swift do. However, we believe there are
substantial opportunities to select a model that:

-   provides comparable and compatible levels of safety;
-   improves the ergonomics of interop with existing C++ APIs; and
-   enables significantly more incremental adoption when starting from C++ code.

However, these benefits aren't free: they come at the cost of added complexity
in the type system itself. Fundamentally, while we will strive to keep Carbon as
simple as we can within its design constraints, we expect Rust to be a simpler
language than Carbon, and for Carbon's improved interop model and incremental
adoption to always come at the cost of complexity. We also don't expect this
complexity to be something that can be fully hidden in implementation details of
standard libraries; some of it will be fundamental and visible to user code in
order to allow the greater degree of flexibility.

There are at least five relevant places where we expect Carbon to differ from
Rust in this category:

-   Including constructs parallel to C++ constructs that are not present in
    Rust.
-   More precise modeling using more pointer types, to allow _safe_ and
    _ergonomic_ mutable aliasing.
-   More incremental enforcement of data-race safety.
-   Aligning concepts and idioms with C++.
-   Having fewer assumptions about unsafe code.

First, Carbon is going to include language constructs to match C++ that Rust
avoids completely, and this comes at a complexity cost. A canonical example is
inheritance: Carbon will have inheritance and virtual dispatch built into the
language, allowing straightforward C++ interop and migration for APIs that use
it. These features increase the size of the language overall, and through
interaction increase the complexity of safety features.

The second of these is the most significant change to safety: supporting safe,
shared mutation. We think this will open up the door to significantly more
flexible code while still providing memory safety. With Rust, there are two
kinds of safe pointers: shared borrows (`&`), and mutable borrows (`&mut`). The
latter are required to be exclusive -- no other safe pointers alias the mutable
object. We expect to have more distinguished types of pointers in Carbon to be
able to model more C++ coding patterns safely, including both shared mutable
pointers, and exclusive mutable pointers -- this is where the complexity
increases. While the exact design of this needs to be carefully fleshed out, our
current idea is to build on the concepts of
[safe, shared mutability in the Ante Language](https://antelang.org/blog/safe_shared_mutability/).
Building illustrative examples of how this impacts C++ code migrating towards
memory safety is an important part of our planned deliverables for 2025.

Third, we expect to have more incremental enforcement of data-race safety in
Carbon, similar to how
[Swift allowed code to incrementally adopt the necessary patterns and restrictions](https://www.swift.org/documentation/concurrency/)
to be data-race safe. The necessary infrastructure was provided as independently
adoptable APIs, and enforcement was then available optionally prior to Swift 6
making this a requirement.

Fourth, we will express Carbon's core concepts around mutation in terms of C++
idioms and concepts. For example, in a container, we expect iterators can use
shared mutable pointers, and the C++ concept of "iterator invalidation" will be
captured by methods that require exclusive mutable access, precluding any
outstanding shared mutable pointers.

Lastly, Carbon's safety model will make fewer assumptions about what unsafe code
is allowed to do than Rust. To avoid undefined behavior, the optimization of
safe code will not rely on properties that are not established locally. For
example, there won't be an assumption that a call to C++ code won't save a copy
of pointers to passed-in objects. When there are restrictions on what unsafe
code can do, we will endeavor to make violations into
[erroneous behavior](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2795r2.html#bigpic),
where the behavior is well-defined while still considered a diagnosable
programming mistake. This reduces the risks of unsafe code during its
incremental migration to memory safe code. It also leaves unsafe Carbon code as
reasonable to work in and maintain over a longer period, which can enable more
widespread migration of C++ to unsafe Carbon. The amount of change and effort
required to convert a piece of code from unsafe Carbon to safe Carbon is much
less than the amount of change required to also change languages, introduce an
interop boundary, and make any API changes needed to enable memory safety all at
once. We believe this will significantly reduce churn effort along the
boundaries of memory safety during incremental migration from unsafe Carbon to
memory safe Carbon code as opposed to moving from C++ directly to a memory safe
language.

Some of these differences may eventually be compelling directions for Rust to
improve its C++ interop, and we plan to develop these in active collaboration
with the Rust community. For these areas of potential overlap, Carbon largely
provides an open field for experimentation and proving out ideas. However, we
expect the majority of the differences we can achieve here would introduce
complexity and might change the nature of Rust making it difficult to retrofit.
As a consequence, there are likely to remain durable tradeoff differences
between Carbon and Rust going forward.

## Rationale

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   Without memory safety, it will not be reasonably secure to continue to
        write high performance software in low-level languages that provide the
        desired performance control.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Our understanding of practical safety techniques, especially new details
        of this uncovered over the past several years, directly form the need
        for an update to our safety strategy.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   A primary motivation and constraint across the surface of safety
        features is to better enable C++ code interop and migration.

## Alternatives considered

### Defer beginning the concrete safety design

The current directional strategy has served us well, and it would be technically
possible to continue to defer turning this into a concrete design. However, we
have had strong feedback from multiple interested users that they need the
safety component to be concrete to do any further evaluation of Carbon, and we
are accelerating our work to meet this need.

### Pursue an alternative strategy towards memory safety specifically

There are many different approaches to memory safety that we could pursue. We
haven't exhaustively listed them, however most of the major candidates are
excluded by our specific goals. For example, a GC-based strategy for solving
temporal memory safety is a non-starter for users who specifically need non-GC
memory management. Rather than exhaustively list the alternatives that are
excluded, we have tried to list the specific [goals](#goals-and-requirements)
that led to the proposed direction.

### Adopt a memory safety strategy more closely based on Rust

An especially appealing alternative is to base our memory safety model
_precisely_ on Rust's. It would give us a strong existence proof, a good basis
to understand soundness, etc. Many of the most difficult questions have already
been answered. This was a seriously considered direction.

However, Rust already exists and is an excellent language. Replicating Rust is
_not_ a goal of Carbon. Our goal is to specifically address use cases where Rust
is not viable at the moment, specifically due to the need for pervasive C++
interop or automated migration from a large, existing C++ codebase. This
directly motivates pursuing a memory safety model that attempts to further
optimize the ergonomics and incrementality of adopting memory safety when
starting in this position.
