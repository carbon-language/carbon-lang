# Implicit conversions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Built-in types](#built-in-types)
    -   [Data types](#data-types)
    -   [Equivalent types](#equivalent-types)
    -   [Pointer conversions](#pointer-conversions)
    -   [Type-of-types](#type-of-types)
-   [Semantics](#semantics)
-   [Extensibility](#extensibility)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

When an expression appears in a context in which an expression of a specific
type is expected, the expression is implicitly converted to that type if
possible.

For [built-in types](#built-in-types), implicit conversions are permitted when:

-   The conversion is _lossless_: every possible value for the source expression
    converts to a distinct representation in the target type.
-   The conversion is _semantics-preserving_: corresponding values in the source
    and destination type have the same abstract meaning.

These rules aim to ensure that implicit conversions are unsurprising: the value
that is provided as the operand of an operation should match how that operation
interprets the value, because the identity and abstract meaning of the value are
preserved by any implicit conversions that are applied.

It is possible for user-defined types to [extend](#extensibility) the set of
valid implicit conversions. Such extensions are expected to also follow these
rules.

## Built-in types

### Data types

The following implicit numeric conversions are available:

-   `iN` or `uN` -> `iM` if `M` > `N`
-   `uN` -> `uM` if `M` > `N`
-   `fN` -> `fM` if `M` > `N`
-   `iN` or `uN` -> `fM` if every value of type `iN` or `uN` can be represeted
    in `fM`:
    -   `i11` or `u11` (or smaller) -> `f16`
    -   `i24` or `u24` (or smaller) -> `f32`
    -   `i53` or `u53` (or smaller) -> `f64`
    -   `i64` or `u64` (or smaller) -> `f80` (x86 only)
    -   `i113` or `u113` (or smaller) -> `f128` (if available)
    -   `i237` or `u237` (or smaller) -> `f256` (if available)

In each case, the numerical value is the same before and after the conversion.
An integer zero is translated into a floating-point positive zero.

An integer constant can be implicitly converted to any type `iM`, `uM`, or `fM`
in which that value can be exactly represented. A floating-point constant can be
implicitly converted to any type `fM` in which that value is between the least
representable finite value and the greatest representable finite value
(inclusive), and does not fall exactly half-way between two representable
values, and converts to the nearest representable finite value.

The above conversions are precisely those that C++ considers non-narrowing,
except:

-   Carbon also permits integer to floating-point conversions in more cases. The
    most important of these is that Carbon permits `i32` to be implicitly
    converted to `f64`. Lossy conversions, such as from `i32` to `f32`, are not
    permitted.

-   What Carbon considers to be an integer constant or floating-point constant
    may differ from what C++ considers to be a constant expression.

    **Note:** We have not yet decided what will qualify as a constant in this
    context, but it will include at least integer and floating-point literals,
    with optional enclosing parentheses. It is possible that such constants will
    have singleton types; see
    [#508](https://github.com/carbon-language/carbon-lang/issues/508).

### Equivalent types

The following conversion is available:

-   `T` -> `U` if `T` is equivalent to `U`

Two types are equivalent if they can represent the same set of values and can be
used interchangeably, implicitly. This refines the notion of types being
compatible, where the representation is the same but an explicit cast may be
required to view a value of one type with a compatible but non-equivalent type.

`T` is equivalent to `U` if:

-   `T` is `A*`, `U` is `B*`, and `A` is equivalent to `B`, or
-   `T` is the facet type `U as SomeInterface`, or
-   `U` is the facet type `T as SomeInterface`, or
-   for some type `V`, `T` is equivalent to `V` and `V` is equivalent to `U`.

**Note:** More type equivalence rules are expected to be added over time.

### Pointer conversions

The following pointer conversion is available:

-   `T*` -> `U*` if `T` is a subtype of `U`.

`T` is a subtype of `U` if:

-   `T` is equivalent to `U`, as described above, or
-   `T` is a class derived from `U`.

**Note:** More type subtyping rules are expected to be added over time.

Note that `T*` is not a subtype of `U*` even if `T` is a subtype of `U`. For
example, we can convert `Derived*` to `Base*`, but cannot convert `Derived**` to
`Base**` because that would allow storing a `Derived2*` into a `Derived*`:

```
abstract class Base {}
class Derived extends Base {}
class Derived2 extends Base {}
var d2: Derived2 = {};
var p: Derived*;
var q: Derived2* = &d2;
var r: Base** = &p;
// Bad: would store q to p.
*r = q;
```

**Note:** If we add `const` qualification, we should treat `const T*` as a
subtype of `const U*` if `T` is a subtype of `U`, and should treat `T` as a
subtype of `const T`.

### Type-of-types

A type `T` with [type-of-type](../generics/terminology.md#type-of-type) `TT1`
can be implicitly converted to the type-of-type `TT2` if `T`
[satisfies the requirements](../generics/details.md#subtyping-between-type-of-types)
of `TT2`.

## Semantics

An implicit conversion of an expression `E` of type `T` to type `U`, when
permitted, always has the same meaning as the explicit cast expression `E as U`.
Moreover, such an implicit conversion is expected to exactly preserve the value.
For example, `(E as U) as T`, if valid, should be expected to result in the same
value as produced by `E`.

**Note:** The explicit cast expression syntax has not yet been decided. The use
of `E as T` in this document is provisional.

## Extensibility

Implicit conversions can be defined for user-defined types such as
[classes](#../classes.md) by implementing the `ImplicitAs` interface:

```
interface As(Type:! Dest) {
  fn Convert[me: Self]() -> Dest;
}
interface ImplicitAs(Type:! Dest) extends As(Dest) {}
```

When attempting to implicitly convert an expression `x` of type `T` to type `U`,
the expression is rewritten to `(x as (T as ImplicitAs(U))).Convert()`.

Note that implicit conversions are not transitive. Even if an
`impl A as ImplicitAs(B)` and an `impl B as ImplicitAs(C)` are both provided, an
expression of type `A` cannot be implicitly converted to type `C`. Allowing
transitivity would introduce the risk of ambiguity issues as code evolves and
would in general require a search of a potentially unbounded set of intermediate
types.

## Alternatives considered

-   [Provide lossy and non-semantics-preserving implicit conversions from C++](/docs/proposals/p0820.md#c-conversions)
-   [Provide no implicit conversions](/docs/proposals/p0820.md#no-conversions)
-   [Provide no extensibility](/docs/proposals/p0820.md#no-extensibility)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#820: implicit cnoversions](https://github.com/carbon-language/carbon-lang/pull/820).
