# Restrict observe declarations to names that are part of the enclosing interface

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
    -   [Allow observe declarations to reference global names](#allow-observe-declarations-to-reference-global-names)
    -   [Strictly restrict to enclosed names](#strictly-restrict-to-enclosed-names)

<!-- tocstop -->

## Abstract

This proposal restricts `observe` declarations in an `interface` to only
reference names dependent on `.Self`, generic parameters, and associated
constants that are part of the enclosing `interface`, with the following
exceptions:

-   Allow at most one unrelated value in an equivalence (`==`) chain.
-   Allow unrelated values that satisfy the `impls` constraint immediately in an
    `observe .. == .. impls`.

## Problem

Currently, the design does not state the scope of names an `observe`
declaration within an interface can reference. This allows `observe`
declarations to be defined for types unrelated to the enclosing `interface`.

For example, this is currently syntactically possible:

```carbon
interface I1 {
    observe I2.A == I2.B == I2.C;
}
```

This creates coherence issues since a developer could get a different view of
types before and after an unrelated import. It also violates Carbon's low
context-sensitivity goals by allowing actions at a distance.

## Background

-   [Observe declarations](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#observe-declarations)
-   [Coherence](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/goals.md#coherence)
-   [Low context-sensitivity principle](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/low_context_sensitivity.md)

## Proposal

Only allow referencing names dependent on `.Self`, or a parameter brought into
scope by the enclosing `interface` in `observe` declarations.

However, this rule permits at most one unrelated value in an equivalence (`==`)
chain, and allows unrelated values in `observe .. == .. impls` declarations
provided they immediately satisfy the impls constraint.

This solves the coherence issues by ensuring an interface can primarily observe
its own associated types and parameters, preventing actions at a distance.
This allows limited use of independent types, ensuring all cross-boundary
implementations remain immediately verifiable and locally bounded.

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

In a function, we can refer to these names in `observe` declarations.

```carbon
fn F[T: type, U: I(T)]() {
    observe U.A == U.B impls R;
}
```

This is allowed since the observation is made about the facet `U` rather than
the interface `I` itself, keeping the `observe` declaration locally bounded.

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

To support constraining associated constants to concrete types and evaluating
their implementations, we must permit at most one unrelated value that
immediately implements the `impls` constraint in an `observe .. == .. impls`
chain. With this exception, the unrelated value can act as a bridge proving
the local associated constants in the chain implement an `interface`.

```carbon
interface A {
    let T: type;
}

interface B {
    let X: A where .T == i32;
    let Y: A where .T == i32;

    // Allowed, even though `i32` doesn't depend on `.Self`, an associated
    // constant, or an interface parameter, we need it to deduce `X.T` and
    // `Y.T` implement `Core.AddWith`.
    observe .X.T == i32 == .Y.T impls AddWith;
}
```

## Rationale

By restricting `observe` declarations to names brought into scope by way of
generic parameters, `.Self`, and associated constants with the aforementioned
exceptions, we guarantee that an interface's requirements and constraints
remain primarily self-contained. This preserves [coherence][1] and aligns with
the [low context-sensitivity principle][2].

[1]: https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/goals.md#coherence
[2]: https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/low_context_sensitivity.md

## Alternatives considered

### Allow observe declarations to reference global names

We considered allowing `observe` declarations to reference arbitrary global
names, such as an external interface that is not strictly bound to the current
interface's scope.

This approach was rejected because it directly violates the principle of low
context-sensitivity. If an interface is permitted to observe external, unbound
types, its semantics become dependent on non-local information. A structural
change in a distant part of the codebase could silently alter the interface's
meaning or break coherence.

### Strictly restrict to enclosed names

We considered strictly restricting `observe` declarations to only reference
names dependent on values brought into scope by the enclosing `interface`,
without any exceptions for unrelated types.

This approach was rejected because it prevents from observing implementations
when associated constants are constrained by concrete types. Without allowing
a bridge value, it becomes impossible to deduce that local associated
constants implement specific interfaces, which limits the usage of associated
constants.
