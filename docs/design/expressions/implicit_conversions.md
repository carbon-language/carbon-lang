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
    -   [Same type](#same-type)
    -   [Pointer conversions](#pointer-conversions)
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

Conversion from `i32` to `Vector(i32)` by forming a vector of N zeroes is
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
-   `iN` or `uN` -> `fM` if every value of type `iN` or `uN` can be represented
    in `fM`:
    -   `i8` or `u8` -> `f16`
    -   `i24` or `u24` (or smaller) -> `f32`
    -   `i48` or `u48` (or smaller) -> `f64`
    -   `i64` or `u64` (or smaller) -> `f80` (x86 only)
    -   `i112` or `u112` (or smaller) -> `f128` (if available)
    -   `i232` or `u232` (or smaller) -> `f256` (if available)

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

In addition to the above rules, a negative integer constant `k` can be
implicitly converted to the type `uN` if the value `k` + 2<sup>N</sup> can be
exactly represented, and converts to that value. Note that this conversion
violates the "semantics-preserving" test. For example, `(-1 as u8) as i32`
produces the value `255` whereas `-1 as i32` produces the value `-1`. However,
this conversion is important in order to allow bitwise operations with masks, so
we allow it:

```
// We allow ^0 == -1 to convert to `u32` to represent an all-ones value.
var a: u32 = ^0;
// ^4 == -5 is negative, but we want to allow it to convert to u32 here.
var b: u32 = a & ^4;
```

### Same type

The following conversion is available for every type `T`:

-   `T` -> `T`

### Pointer conversions

The following pointer conversion is available:

-   `T*` -> `U*` if `T` is a class derived from the class `U`.

Even though we can convert `Derived*` to `Base*`, we cannot convert `Derived**`
to `Base**` because that would allow storing a `Derived2*` into a `Derived*`:

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
interface ImplicitAs(Dest:! type) {
  extends As(Dest);
  // Inherited from As(Dest):
  // fn Convert[self: Self]() -> Dest;
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

-   [Provide lossy and non-semantics-preserving implicit conversions from C++](/proposals/p0820.md#c-conversions)
-   [Provide no implicit conversions](/proposals/p0820.md#no-conversions)
-   [Provide no extensibility](/proposals/p0820.md#no-extensibility)
-   [Apply implicit conversions transitively](/proposals/p0820.md#transitivity)
-   [Do not allow negative constants to convert to unsigned types](/proposals/p1191.md#converting-complements-to-unsigned-types)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820).
-   Proposal
    [#866: Allow ties in floating literals](https://github.com/carbon-language/carbon-lang/pull/866).
