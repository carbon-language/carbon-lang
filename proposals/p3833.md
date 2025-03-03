# SemIR fidelity when representing rewrite semantics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3833)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Work to incorporate needed simplifications directly into the design](#work-to-incorporate-needed-simplifications-directly-into-the-design)
    -   [Keep a full fidelity mode for any other optimizations](#keep-a-full-fidelity-mode-for-any-other-optimizations)
    -   [Example: overloaded operators](#example-overloaded-operators)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Strictly adhere to a full-fidelity model](#strictly-adhere-to-a-full-fidelity-model)
    -   [Directly implement the optimized model](#directly-implement-the-optimized-model)

<!-- tocstop -->

## Abstract

The toolchain's [Semantic IR][semir] should start off modeling the full,
complex, and rich library-based and generic extension point semantics of Carbon
without eliding any layers or rewrites for compile time efficiency. We shouldn't
front-load elision or optimization when implementing the designs.

Once we have a full-fidelity implementation, we should work to build an
efficient elision, short-circuit, or common case simplification into the design
itself sufficient to make the SemIR model efficient. Only if we cannot find a
reasonable approach for that should we diverge the SemIR model to optimize its
efficiency, and we should preserve full fidelity in an optional mode.

[semir]:
    https://docs.google.com/document/d/1RRYMm42osyqhI2LyjrjockYCutQ5dOf8Abu50kTrkX0/edit?resourcekey=0-kHyqOESbOHmzZphUbtLrTw#heading=h.503m6lfcnmui

## Problem

Carbon's design heavily leverages types and APIs defined in the `Core` package
and implicitly imported with the prelude. It also defines rich extension points
and generic semantic models for the language, typically through rewrites from
the user-facing syntax into more complex syntax that operates through
`interface`s and `impl`s that provide the generic model and customization
points.

Modeling this at full fidelity in the toolchain's [Semantic IR][semir] is
expected to create an unreasonably expensive representation for common and
pervasive patterns in code. On the other hand, every divergence between the
SemIR model and the design adds both risk and cost. When we can keep the two in
sync, we get the best engineering tradeoffs, provided it is reasonably
efficient.

## Background

Several proposals have started to surface these kinds of challenge, but to a
lesser degree. Some examples:

-   [Proposal #820 - Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
-   [Proposal #845 - `as` expressions](https://github.com/carbon-language/carbon-lang/pull/845)
-   [Proposal #1083 - Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

The most significant step that motivates capturing an explicit strategy here is
[proposal #3720 - Binding operators](https://github.com/carbon-language/carbon-lang/pull/3720).

## Proposal

Initially, the toolchain should have SemIR model the full semantics of Carbon,
with all rewrites, expressive hooks, and library references. This will serve two
purposes as we build out the toolchain's support for all of Carbon's features.

### Work to incorporate needed simplifications directly into the design

First, this will allow us to study the resulting SemIR, and understand exactly
how and where the efficiency explodes for expected coding patterns. We should
use this to try to adjust the design to build in targeted common-case
simplifications and elisions that make precise modeling of the design
sufficiently cheap in practice.

We see this pattern emerging in good ways already today in the toolchain with
implicit conversions: when calling functions, we need to allow for implicit
conversions. But we don't want to have an identity implicit conversion on all
the arguments which already are the correct type. Rather than notionally
"always" implicitly converting, we make the design itself specify that the
implicit conversion only occurs when needed and can then have the desired
minimal representation in SemIR.

Beyond simplifying the SemIR model, it also ends up addressing an important
aspect of the semantics -- we don't want to create unnecessary temporaries to
initialize rather than directly initializing. The toolchain in fact follows
similar patterns to model expression categories for similar reasons -- the
minimal semantic model is also the desired design.

We can even incorporate other kinds of design changes to allow the full fidelity
SemIR model to be efficient, such as assuring that we can build and reuse a
cached representation repeatedly and cheaply rather than requiring it to be
rebuilt on each instance of a specific pattern.

### Keep a full fidelity mode for any other optimizations

We may still ultimately end up unable to achieve the desired SemIR efficiency
even after making all the simplifications we can to the design itself. If and
when we reach this point, we will still want to have the full fidelity
implementation and be able to use it behind some option. We can use it to
understand how the design works in complex cases and to debug unexpected
behavior in the toolchain. This means starting with that implementation doesn't
result in sunk cost.

### Example: overloaded operators

> Note: this example isn't trying to capture all the nuance of the current
> design or suggest any change to the design. It is merely trying to provide an
> illustration of how we might follow the proposal in practice.

As an example to help illustrate this, how should the toolchain model overloaded
operators? This proposal suggests that _initially_, the implementation should
aim to fully model the rewrite-based dispatch through interfaces in the prelude.
That is, each use of `x + y` should turn into roughly equivalent SemIR as would
be used to model the rewritten semantics of
`x.(package#Core.AddWith(typeof(y)).Op)(y)`. This in turn would dispatch to an
exact-type implementation which would provide any implicit conversions, and so
on. And importantly, this would occur even when we know that the types of `x`
and `y` are exactly `i32` and we could alternatively directly hard code the
toolchain to use the desired builtin implementation.

To be precise, the expectation is that the SemIR for `x + y` should as a
consequence model all of:

-   Looking up the `package#Core.AddWith` interface, using the imagined syntax
    `package#<Name>` for directly looking up a specific package name (as opposed
    to doing unqualified name lookup).
-   Passing the type of `y` as a parameter to the interface, which is
    illustrated with the imagined syntax `typeof(<expression>)` but would likely
    directly use the type rather than forming an actual `typeof` expression (see
    below).
-   Looking up the `Op` method.
-   Calling that `Op` method on `x` with `y` as its parameter just as we would
    model any call to `object.(Interface.Method)(y)`.

> Note: this isn't a proposal for either the `package#<Name>` or
> `typeof(<expression>)` imagined syntaxes. If we want these syntaxes, that will
> be a separate proposal.

However, specific things that would still be reasonable without deviating from
the full semantic modeling here:

-   Not representing a user-written `typeof` style expression to compute the
    type, or duplicating the `y` expression.
-   Caching and reusing SemIR for immutable things like successfully resolved
    name lookup. Concretely, reusing a single cached lookup of
    `package#Core.AddWith` would still provide the full model. And similarly,
    reusing an already computed parameterization such as
    `package#Core.AddWith(i32)` repeatedly, and the lookup of `Op` within that
    parameterization.

The reason is that these don't change the semantics in any way -- they are
constructively the same results.

This should be the initial implementation _goal_, not a requirement for any
incremental progress on implementation. Short-circuiting or other approximations
used in incrementally building towards this goal should always be reasonable to
consider based on the tradeoffs at hand. But they should be seen as temporary,
incremental measures and where possible removed so we can evaluate the full
semantic model when the implementation is complete.

Based on the practical costs that this kind of model incurs once we have it
implemented, we should first look for ways to adjust the design itself to avoid
forming the complex semantic expansion in enough cases to make the model
scalable. And only when we've exhausted those options begin adding long-term
shortcuts in the SemIR model for common cases with a option to disable them for
testing and debugging.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   It's easy when specifying the design of the language with rewrite
        semantics in a way that accidentally has a cycle -- where the rewrite
        result is still a construct that should be rewritten. Without the
        proposed constraint, these accidental cycles are easily resolved in the
        implementation by exiting at a convenient place. However, this proposal
        will ensure that the rule for where the cycle is broken is actually
        surfaced into the design, making the design more robust and complete in
        describing the expected behavior.
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   Preserving a full fidelity model of the language design in SemIR helps
        ensure we can build rich and powerful tools that reason about even
        complex or typically "hidden" parts of Carbon's design. This has
        regularly proven critical in the design of effective tooling for C++ and
        other languages.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   We can't afford for full fidelity representation of Carbon's design to
        make the practical use of the toolchain have poor compile times and slow
        down the development experience with Carbon. We need to have a strategy
        for addressing the expected overhead of the level of customization and
        layered design that we're building into Carbon.

## Alternatives considered

### Strictly adhere to a full-fidelity model

We could simply insist on never deviating from the full-fidelity model. However,
this would force us to compromise at least one of our top-level goals for
Carbon: either we would have to regress the speed and scalability of
development, or lose some of the simplicity and cohesive design.

Both of those seem worse outcomes than having some limited divergence between
SemIR and the design, and there exist reasonable mitigations such as a option to
switch between modes.

### Directly implement the optimized model

We could directly jump to an optimized model that we expect to match the common
case behavior, and add opt-in complexity when needed for non-common-cases.
However, this would both remove a useful tool in driving simplification of the
design itself that can result in a overall superior tradeoff. It would also
create risk of divergence, and the need to eventually implement mode to disable
the optimization.

In some cases, we would end up discovering a need to simplify the design for
semantic reasons, and the divergence would have resulted in wasted effort in the
toolchain.

Starting with the full fidelity model seems to more reliably converge the design
early and result in a minimal set of effective optimizations.
