# Identification of a named constraint during definition

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6902)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Example with `Self`](#example-with-self)
    -   [Motivating the partially identified state](#motivating-the-partially-identified-state)
    -   [Disallowing conversions to incomplete named constraint](#disallowing-conversions-to-incomplete-named-constraint)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Considering a facet type of a named constraint to be identified in its definition](#considering-a-facet-type-of-a-named-constraint-to-be-identified-in-its-definition)
    -   [Restricting to `Self`](#restricting-to-self)
    -   [Allowing limited conversions to partially identified facet types](#allowing-limited-conversions-to-partially-identified-facet-types)

<!-- tocstop -->

## Abstract

This proposal updates the criteria for when a facet type is considered
"identified." Specifically, it relaxes the requirement for named constraints,
allowing them to be incrementally identified inside the definition, rather than
requiring them to be fully complete, when used through the `Self` facet. This
change enables impl lookups with `Self` within a constraint's definition to
correctly resolve witnesses based on prior `require impls` statements in the
definition.

## Problem

Under the rules established in
[Proposal #5168](/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md),
a facet type is identified only if all its referenced interfaces are declared
and all its referenced named constraints are complete.

This definition creates a circularity problem during the definition of a named
constraint. If a `require impls` statement inside a named constraint definition
relies on an impl lookup with `Self`, that lookup will fail because the facet
type of `Self` is not identified before the named constraint is complete. This
prevents `require impls` statements in a named constraint from depending on
earlier ones.

## Background

-   [Proposal #5168](/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md):
    Introduced rules for facet type identification and completion.

## Proposal

We propose redefining the identification criteria for facet types by introducing
a new partially identified state.

A facet type can be in one of three states: unidentified, partially identified,
or identified. A facet type's identifiedness is the minimum of that of its
constituents:

-   When a facet type refers to an interface, the facet type is not identified
    until the interface is declared, and is fully identified after. This
    includes inside the definition of the interface.
-   When a facet type refers to a named constraint, the facet type is not
    identified until the named constraint is declared. It is partially
    identified during the definition of the named constraint, and it is fully
    identified after.

The change from previous rules is that a facet type containing a named
constraint is now partially identified inside the definition of that named
constraint.

As in
[#5168](/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md),
an `impl` declaration and `require` statement each requires its constraint to be
identified.

We define the rules for facets in impl lookups, which are representable as
`<self> as <target facet type>` conversions as follows:

-   The target facet type of an impl lookup must be defined.
    -   This allows the full set of interfaces to be known, which allows a
        stable ordering of witnesses for those interfaces to be produced by the
        impl lookup.
-   If the self is a facet, its facet type may be in any state of
    identifiedness.
    -   The impl lookup may provide a witness from the facet type of self, using
        the known constraints of any partially identified or identified
        constituent of the facet type.

In particular, this means that `Self as I` inside the definition of a named
constraint `N` may use any `require impls` statements before that use of `Self`
in order to provide a witness for `I`.

To improve diagnostics, we also propose to disallow using a named constraint
inside its own definition, except through the type of `Self`. Any other use is
diagnosed as an error. This provides a clear error when a named constraint
appears in the constraint of a `require` statement inside its definition.

### Example with `Self`

This change allows the compiler to treat a named constraint as partially
identified for the purposes of impl lookup while it is still being typechecked.

As the compiler processes a series of `require impls` statements within a named
constraint, the partially identified facet type of the named constraint, which
can be accessed through `Self`, is built up incrementally. The partially
identified facet type of `Self` will contain interfaces provided by
`require impls` statements written prior to that use of `Self`. Thus later
`require impls` statements can use the partially identified facet type of `Self`
to find witnesses provided by earlier `require impls` statements during impl
lookup.

The following example demonstrates a scenario that is currently invalid but
would be enabled by this proposal:

```carbon
interface Y {}

interface NeedsY(T:! Y) {}

constraint W {
  require impls Y;

  // This requires an impl lookup where the query self value is `Self`
  // (which is of type `W`) and the query interface is `Y`. The lookup
  // requires identifying the facet type of `Self` to find a witness.
  // After this proposal, W is partially identified because it has begun being
  // defined. The lookup for `Self as Y` can now succeed due to the previous
  // `require impls` statement.
  require impls NeedsY(Self);
}
```

In this example, identifying `W` while it is being defined allows the lookup for
`Self as Y` in order to form a facet value for `NeedsY` to succeed because the
compiler knows `Self` (of type `W`) implements `Y` from the previous
`require impls` statement.

### Motivating the partially identified state

If a facet type for a named constraint was considered identified (not partially
identified) inside its definition, the following becomes possible:

```carbon
constraint W {
  require C impls W;
  require impls Z;
}
```

This says that `C` must implement `W`, yet `W` is not fully defined. At that
line `W` is still empty, so it places no requirements on `C`. The next line
requires that anything implementing `W` must implement `Z`.

To prevent this, the constraint of a `require` statement must still be
identified, and the facet type of the being-defined named constraint is only
partially-identified.

Note this also disallows the use of `W` through an alias:

```carbon
constraint W;
alias X = W;
constraint W {
  // Error: X refers to named constraint `W` that is not identified.
  require impls X;
}
```

### Disallowing conversions to incomplete named constraint

By requiring the target facet type of an impl lookup to be identified, we
disallow an incomplete named constraint from being part of the target of an impl
lookup.

The result of an impl lookup stores witnesses for the target facet type. If the
target facet type was partially identified, the same facet type may have a
different set of interfaces later. A change in the set of interfaces would
invalidate the set of stored witnesses.

For example, if this was allowed:

```carbon
interface Z {}
interface X {}

constraint W;
class C(T:! W) {}
class D {}

interface Y(T:! type) {}

constraint W {
  require impls Z;
  // Constructs a `D as W` facet value.
  require impls Y(C(D));
  require impls X;
}
```

In this example the argument to `C` will be `D as W` which will store a witness
for each interface in the identified facet type `W`. If we allow a partially
identified facet type, then it will store a witness for the interface `Z`. But
later uses of the facet value in `Y(C(D))` would expect witnesses for `Z`, `Y`
and `X`, leading to unsoundness.

## Rationale

This change aligns with Carbon's goal of
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
by allowing expressive generics as they have been designed.

This follows the
[Information accumulation](/docs/project/principles/information_accumulation.md)
principle by increasing the information available to the program with each
statement.

## Alternatives considered

### Considering a facet type of a named constraint to be identified in its definition

Initial versions of this proposal did not differentiate between partially
identified and identified. This led to unsoundness by allowing conversion to a
facet with the incomplete named constraint in the facet's type, as described in
[Disallowing conversions to incomplete named constraint](#disallowing-conversions-to-incomplete-named-constraint).

### Restricting to `Self`

We considered restricting the use of the partially identified facet type to only
be on the `Self` facet value. This provided a way to reduce exposure of the
partially identified facet type. But by differentiating the the partially
identified state from identified, we can form the rules around the state of the
facet type instead of the identity of the facet.

### Allowing limited conversions to partially identified facet types

We considered allowing conversions from `N & J` to `N & K` where `N` is
partially identified, and `J` and `K` are identified.

It seems possible to support this for symbolic facets, by not storing the set of
witnesses in the facet value. If the facet type is partially identified, we can
defer the collection of witnesses, and just store the facet types that the facet
was converted from. In this model, the facet type itself acts as a type of
witness that we will will be able to find a witness later once the facet type is
identified.

We leave this to a future proposal if and when we find this additional
complexity worth adding to the language model.
