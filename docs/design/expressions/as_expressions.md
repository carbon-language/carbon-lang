# `as` expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Built-in types](#built-in-types)
    -   [Data types](#data-types)
    -   [Compatible types](#compatible-types)
-   [Extensibility](#extensibility)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An expression of one type can be explicitly cast to another type by using an
`as` expression:

```
var n: i32 = Get();
var f: f32 = n as f32;
```

An `as` expression can be used to perform any implicit conversion, either when
the context does not imply a destination type or when it is valuable to a reader
of the code to make the conversion explicit. In addition, `as` expressions can
perform safe conversions that nonetheless should not be performed implicitly,
such as lossy conversions or conversions that lose capabilities or change the
way a type would be interpreted.

As guidelines, an `as` conversion should be permitted when:

-   The conversion is _complete_: it produces a well-defined output value for
    each input value.
-   The conversion is _unsurprising_: the resulting value is the expected value
    in the destination type.

For example:

-   A conversion from `fM` to `iN` is not complete, because it is not defined
    for input values that are out of the range of the destination type, such as
    infinities or, if `N` is too small, large finite values.
-   A conversion from `iM` to `iN`, where `N` < `M`, is either not complete or
    not unsurprising, because there is more than one possible expected behavior
    for an input value that is not within the destination type, and those
    behaviors are not substantially the same -- we could perform two's
    complement wrapping, saturate, or produce undefined behavior analogous to
    arithmetic overflow.
-   A conversion from `iM` to `fN` can be unsurprising, because even though
    there may be a choice of which way to round, the possible values are
    substantially the same.

It is possible for user-defined types to [extend](#extensibility) the set of
valid explicit casts that can be performed by `as`. Such extensions are expected
to follow these guidelines.

## Precedence and associativity

`as` expressions are non-associative.

```
var b: bool = true;
// OK
var n: i32 = (b as i1) as i32;
var m: auto = b as (bool as Hashable);
// Error, ambiguous
var m: auto = b as T as U;
```

**Note:** `b as (bool as Hashable)` is valid but not useful, because
[the second operand of `as` is implicitly converted to type `type`](#extensibility).
This expression therefore has the same interpretation as `b as bool`.

**TODO:** We should consider making `as` expressions left-associative now that
facet types have been removed from the language.

The `as` operator has lower precedence than operators that visually bind
tightly:

-   prefix symbolic operators
    -   dereference (`*a`)
    -   negation (`-a`)
    -   complement (`~a`)
-   postfix symbolic operators
    -   pointer type formation (`T*`),
    -   function call (`a(...)`),
    -   array indexing (`a[...]`), and
    -   member access (`a.m`).

The `as` operator has higher precedence than assignment and comparison. It is
unordered with respect to binary arithmetic, bitwise operators, and unary `not`.

```
// OK
var x: i32* as Eq;
// OK, `x as (U*)` not `(x as U)*`.
var y: auto = x as U*;

var a: i32;
var b: i32;
// OK, `(a as i64) < ((*x) as i64)`.
if (a as i64 < *x as i64) {}
// Ambiguous: `(a + b) as i64` or `a + (b as i64)`?
var c: i32 = a + b as i64;
// Ambiguous: `(a as i64) + b` or `a as (i64 + b)`?
var d: i32 = a as i64 + b;

// OK, `(-a) as f64`, not `-(a as f64)`.
// Unfortunately, the former is undefined if `a` is `i32.MinValue()`;
// the latter is not.
var u: f64 = -a as f64;

// OK, `i32 as (GetType())`, not `(i32 as GetType)()`.
var e: i32 as GetType();
```

## Built-in types

### Data types

In addition to the [implicit conversions](implicit_conversions.md#data-types),
the following numeric conversions are supported by `as`:

-   `iN`, `uN`, or `fN` -> `fM`, for any `N` and `M`. Values that cannot be
    exactly represented are suitably rounded to one of the two nearest
    representable values. Very large finite values may be rounded to an
    infinity. NaN values are converted to NaN values.

-   `bool` -> `iN` or `uN`. `false` converts to `0` and `true` converts to `1`
    (or to `-1` for `i1`).

Conversions from numeric types to `bool` are not supported with `as`; instead of
using `as bool`, such conversions can be performed with `!= 0`.

Lossy conversions between `iN` or `uN` and `iM` or `uM` are not supported with
`as`, and similarly conversions from `fN` to `iM` are not supported.

**Future work:** Add mechanisms to perform these conversions.

### Compatible types

The following conversion is supported by `as`:

-   `T` -> `U` if `T` is
    [compatible](../generics/terminology.md#compatible-types) with `U`.

**Future work:** We may need a mechanism to restrict which conversions between
adapters are permitted and which code can perform them. Some of the conversions
permitted by this rule may only be allowed in certain contexts.

## Extensibility

Explicit casts can be defined for user-defined types such as
[classes](../classes.md) by implementing the `As` interface:

```
interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}
```

The expression `x as U` is rewritten to `x.(As(U).Convert)()`.

**Note:** This rewrite causes the expression `U` to be implicitly converted to
type `type`. The program is invalid if this conversion is not possible.

## Alternatives considered

-   [Allow `as` to perform some unsafe conversions](/proposals/p0845.md#allow-as-to-perform-some-unsafe-conversions)
-   [Allow `as` to perform two's complement truncation](/proposals/p0845.md#allow-as-to-perform-twos-complement-truncation)
-   [`as` only performs implicit conversions](/proposals/p0845.md#as-only-performs-implicit-conversions)
-   [Integer to bool conversions](/proposals/p0845.md#integer-to-bool-conversions)
-   [Bool to integer conversions](/proposals/p0845.md#bool-to-integer-conversions)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#845: `as` expressions](https://github.com/carbon-language/carbon-lang/pull/845).
