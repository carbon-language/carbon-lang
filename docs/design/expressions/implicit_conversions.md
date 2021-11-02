# Implicit conversions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Properties of implicit conversions](#properties-of-implicit-conversions)
    -   [Lossless](#lossless)
    -   [Semantics-preserving](#semantics-preserving)
    -   [Examples](#examples)
-   [Built-in types](#built-in-types)
    -   [Data types](#data-types)
    -   [Equivalent types](#equivalent-types)
    -   [Pointer conversions](#pointer-conversions)
        -   [Pointer conversion examples](#pointer-conversion-examples)
    -   [Type-of-types](#type-of-types)
-   [Consistency with `as`](#consistency-with-as)
-   [Extensibility](#extensibility)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

When an expression appears in a context in which an expression of a specific
type is expected, the expression is implicitly converted to that type if
possible.

For [built-in types](#built-in-types), implicit conversions are permitted when:

-   The conversion is [_lossless_](#lossless): every possible value for the
    source expression converts to a distinct value in the target type.
-   The conversion is [_semantics-preserving_](#semantics-preserving):
    corresponding values in the source and destination type have the same
    abstract meaning.

These rules aim to ensure that implicit conversions are unsurprising: the value
that is provided as the operand of an operation should match how that operation
interprets the value, because the identity and abstract meaning of the value are
preserved by any implicit conversions that are applied.

It is possible for user-defined types to [extend](#extensibility) the set of
valid implicit conversions. Such extensions are expected to also follow these
rules.

## Properties of implicit conversions

### Lossless

We expect implicit conversion to never lose information: if two values are
distinguishable before the conversion, they should generally be distinguishable
after the conversion. It should be possible to define a conversion in the
opposite direction that restores the original value, but such a conversion is
not expected to be provided in general, and might be computationally expensive.

Because an implicit conversion is converting from a narrower type to a wider
type, implicit conversions do not necessarily preserve static information about
the source value.

### Semantics-preserving

We expect implicit conversions to preserve the meaning of converted values. The
assessment of this criterion will necessarily be subjective, because the
meanings of values generally live in the mind of the programmer rather than in
the program text. However, the semantic interpretation is expected to be
consistent from one conversion to another, so we can provide a test: if multiple
paths of implicit conversions from a type `A` to a type `B` exist, and the same
value of type `A` would convert to different values of type `B` along different
paths, then at least one of those conversions must not be semantics-preserving.

A semantics-preserving conversion does not necessarily preserve the meaning of
particular syntax when applied to the value. The same syntax may map to
different operations in the new type. For example, division may mean different
things in integer and floating-point types, and member access may find different
members in a derived class pointer versus in a base class pointer.

### Examples

Conversion from `i32` to `Vector(int)` by forming a vector of N zeroes is
lossless but not semantics-preserving.

Conversion from `i32` to `f32` by rounding to the nearest representable value is
semantics-preserving but not lossless.

Conversion from `String` to `StringView` is lossless, because we can compute the
`String` value from the `StringView` value, and semantics-preserving because the
string value denoted is the same. Conversion in the other direction may or may
not be semantics-preserving depending on whether we consider the address to be a
salient part of a `StringView`'s value.

## Built-in types

### Data types

The following implicit numeric conversions are available:

-   `iN` or `uN` -> `iM` if `M` > `N`
-   `uN` -> `uM` if `M` > `N`
-   `fN` -> `fM` if `M` > `N`
-   `iN` or `uN` -> `fM` if every value of type `iN` or `uN` can be represeted
    in `fM`:
    -   `i12` or `u11` (or smaller) -> `f16`
    -   `i25` or `u24` (or smaller) -> `f32`
    -   `i54` or `u53` (or smaller) -> `f64`
    -   `i65` or `u64` (or smaller) -> `f80` (x86 only)
    -   `i114` or `u113` (or smaller) -> `f128` (if available)
    -   `i238` or `u237` (or smaller) -> `f256` (if available)

In each case, the numerical value is the same before and after the conversion.
An integer zero is translated into a floating-point positive zero.

An integer constant can be implicitly converted to any type `iM`, `uM`, or `fM`
in which that value can be exactly represented. A floating-point constant can be
implicitly converted to any type `fM` in which that value is between the least
representable finite value and the greatest representable finite value
(inclusive), and converts to the nearest representable finite value, with ties
broken by picking the value for which the mantissa is even.

The above conversions are also precisely those that C++ considers non-narrowing,
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
    have singleton types; see issue
    [#508](https://github.com/carbon-language/carbon-lang/issues/508).

### Equivalent types

The following conversion is available:

-   `T` -> `U` if `T` is equivalent to `U`

Two types are equivalent only if they have the same set of values with the same
meaning and the same representation, with the same set of capabilities and
constraints, where the only difference is how the type interprets operations on
values of that type.

`T` is equivalent to `U` if:

-   `T` is the same type as `U`, or
-   `T` is the facet type `U as SomeInterface`, or
-   `U` is the facet type `T as SomeInterface`, or
-   `T` is `A*`, `U` is `B*`, and `A` is equivalent to `B`, or
-   for some type `V`, `T` is equivalent to `V` and `V` is equivalent to `U`.

**Note:** More type equivalence rules are expected to be added over time.

A prerequisite for types being equivalent is that they are
[compatible](../generics/terminology.md#compatible-types), and in particular
that they have the same set of values and the same representation for those
values. However, types being compatible does not imply that an implicit
conversion, or even an explicit cast, between those types is necessarily valid.
This is because the type of a value models not only the representation of the
value but also the capabilities that a user of the value has to interact with
the value. Two compatible types may expose different capabilities, such as the
capability to mutate the object or to access its implementation details, and
conversions between such types may require an explicit cast if the conversion is
possible at all.

### Pointer conversions

The following pointer conversion is available:

-   `T*` -> `U*` if `T` is a subtype of `U`.

`T` is a subtype of `U` if:

-   `T` is equivalent to `U`, as described above, or
-   `T` is equivalent to a class derived from a class equivalent to `U`.

**Note:** More type subtyping rules are expected to be added over time.

`T*` is not necessarily a subtype of `U*` even if `T` is a subtype of `U`. For
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

**Note:** If we add `const` qualification, we could treat `const T*` as a
subtype of `const U*` if `T` is a subtype of `U`, and could treat `T` as a
subtype of `const T`.

#### Pointer conversion examples

With these classes:

```
base class C;
let F: auto = C as Hashable;
class D extends C;
```

These implicit pointer conversions are permitted:

-   `D*` -> `C*`: `D` is a subtype of `C`
-   `F*` -> `C*`: `F` is equivalent to `C`, so `F` is a subtype of `C`
-   `C*` -> `F*`: `C` is equivalent to `F`, so `C` is a subtype of `F`
-   `F**` -> `C**`: `F` is equivalent to `C`, so `F*` is equivalent to `C*`, so
    `F*` is a subtype of `C*`
-   `D*` -> `F*`: `D` is derived from `C` and `C` is equivalent to `D`, so `D`
    is a subtype of `F`

These implicit pointer conversions are disallowed:

-   `C*` -> `D*`: `C` is not a subtype of `D`
-   `D**` -> `C**`: `D*` is not a subtype of `C*`

Note that "equivalent to" means we can freely convert back and forwards; the
difference in the types is just changing which operations are surfaced, not
changing anything about the interpretation or switching between different
abstractions. In contrast, "subtype of" permits conversion from a more specific
type to a more general type, so the reverse conversion is not necessarily valid.

### Type-of-types

A type `T` with [type-of-type](../generics/terminology.md#type-of-type) `TT1`
can be implicitly converted to the type-of-type `TT2` if `T`
[satisfies the requirements](../generics/details.md#subtyping-between-type-of-types)
of `TT2`.

## Consistency with `as`

An implicit conversion of an expression `E` of type `T` to type `U`, when
permitted, always has the same meaning as the
[explicit cast expression `E as U`](as_expressions.md). Moreover, because such
an implicit conversion is expected to exactly preserve the value,
`(E as U) as T`, if valid, should be expected to result in the same value as
produced by `E` even if the `as T` cast cannot be performed as an implicit
conversion.

## Extensibility

Implicit conversions can be defined for user-defined types such as
[classes](../classes.md) by implementing the `ImplicitAs` interface, which
extends
[the `As` interface used to implement `as` expressions](as_expressions.md#extensibility):

```
interface ImplicitAs(Dest:! Type) extends As(Dest) {
  // Inherited from As(Dest):
  // fn Convert[me: Self]() -> Dest;
}
```

When attempting to implicitly convert an expression `x` to type `U`, the
expression is rewritten to `x.(ImplicitAs(U).Convert)()`.

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
-   [Apply implicit conversions transitively](/docs/proposals/p0820.md#transitivity)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820).
-   Proposal
    [#866: Allow ties in floating literals](https://github.com/carbon-language/carbon-lang/pull/866).
