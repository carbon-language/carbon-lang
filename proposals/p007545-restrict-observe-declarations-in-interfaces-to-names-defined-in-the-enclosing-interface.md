# Restrict observe declarations in interfaces to names defined in the enclosing interface

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7545)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

This proposal restricts `observe` declarations in an `interface` to only
reference names that are part of the enclosing `interface`.

## Problem

Currently, the design does not state the scope of names an `observe`
declaration within an interface can reference to. This allows `observe`
declarations to be defined for types unrelated to the enclosing `interface`.

For example, this is currently syntactically possible:

```carbon
interface I1 {
    observe I2.A == I2.B == I2.C;
}
```

This creates coherence issues since a developer could get a different view of
types before and after an unrelated import. And violates Carbon's low
context-sensitivity goals by allowing actions at a distance.

## Background

-   [Observe declarations](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#observe-declarations)
-   [Coherence](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/goals.md#coherence)
-   [Low context-sensitivity principle](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/low_context_sensitivity.md)

## Proposal

Only allow referencing names brought into scope by the enclosing `interface`
in `observe` declarations. This solves the coherence issues by ensuring an
interface can only observe its own associated types and parameters,
preventing actions at a distance.

## Details

Referring to `.Self`, generic parameters, and associated constants defined in
the enclosing type is allowed.

```carbon
interface I(T:! P) {
    let A: Q where .Self == T;
    let B: R where .Self == A;
    let C: S where .Self == B;

    // Allowed, all names are associated constants defined in the enclosing
    // interface.
    observe A == B == C;

    // Allowed, both `T` and `A` are brought to scope by `I`, and `A`
    // implements `Q`.
    observe T == A impls Q;
}
```

An associated constant may implement an interface that defines its own
associated constants. Let's assume that the interface `Q` from the example
above defines three associated constants `X`, `Y` and `Z`.

In a function we can refer to these names in `observe` declarations.

```carbon
fn F[T: type, U: I(T)]() {
    observe U.A == U.B impls R;
}
```

This is allowed since the observation is made about the facet `U` rather than
the interface `I` itself.

Extending this logic to interfaces, an associated constant acts as a localized
binding. Therefore, we can refer to names accessed through associated
constants and generic parameters defined by the enclosing interface without
affecting global reasoning.

```carbon
interface I(T:! P) {
    let A: Q where .Self == T;
    let B: R where .Self == A;
    let C: S where .Self == B;

    // Allowed, observation is made about `A`, and does not affect the
    // interface `Q` itself.
    observe A.X == A.Y == A.Z;

    // Not allowed, `Q` is not brought to scope by `I`.
    observe Q.X == Q.Y == Q.Z;
}
```

## Rationale

TODO: How does this proposal effectively advance Carbon's goals? Rather than
re-stating the full motivation, this should connect that motivation back to
Carbon's stated goals and principles. This may evolve during review. Use links
to appropriate sections of [`/docs/project/goals.md`](/docs/project/goals.md),
and/or to documents in [`/docs/project/principles`](/docs/project/principles).
For example:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

## Alternatives considered

TODO: What alternative solutions have you considered?
