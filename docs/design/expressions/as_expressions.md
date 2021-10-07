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
    -   [Pointer conversions](#pointer-conversions)
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

-   The conversion is _safe_: it produces a well-defined output value for each
    input value.
-   The conversion is _unsurprising_: the resulting value is the expected value
    in the destination type.

In cases where a cast is only defined for a subset of the possible inputs, an
`assume_as` expression can be used. An `assume_as` expression behaves like an
`as` expression, except that the domain of the conversion is narrower than the
entire input type, so the conversion is not safe as defined above.

It is possible for user-defined types to [extend](#extensibility) the set of
valid explicit casts that can be performed by `as` and `assume_as`. Such
extensions are expected to follow these guidelines.

## Precedence and associativity

`as` and `assume_as` expressions are non-associative.

```
var b: bool = true;
// OK
var n: i32 = (b as i1) as i32;
var m: auto = b as (bool as Hashable);
// Error, ambiguous
var m: auto = b as T as U;
```

The `as` and `assume_as` operators have lower precedence than operators that
visually bind tightly:

-   prefix symbolic operators
    -   dereference (`*a`)
    -   negation (`-a`)
    -   complement (`~a`)
-   postfix symbolic operators
    -   pointer type formation (`T*`),
    -   function call (`a(...)`),
    -   array indexing (`a[...]`), and
    -   member access (`a.m`).

The `as` and `assume_as` operators have higher precedence than assignment and
comparison. They are unordered with respect to binary arithmetic, bitwise
operators, unary `not`, and each other.

```
// OK
var x: i32* as Comparable;
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

// OK, `(-a) assume_as u64`, not `-(a assume_as u64)`.
var u: u64 = -a assume_as u64;

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

The following additional numeric conversions are supported by `assume_as`:

-   `iN` or `uN` -> `iM` or `uM`, for any `N` and `M`. It is a programming error
    if the source value cannot be represented in the destination type.

    **TODO:** Once we have a two's complement truncation operation with defined
    behavior on overflow, link to it from here as an alternative.

-   `fN` -> `iM`, for any `N` and `M`. Values that cannot be exactly represented
    are suitably rounded to one of the two nearest representable values. It is a
    programming error if the source value does not round to an integer that can
    be represented in the destination type.

Conversions from numeric types to `bool` are not supported with `as`; instead of
using `as bool`, such conversions can be performed with `!= 0`.

**Note:** The precise rounding rules for these conversions have not yet been
decided.

### Compatible types

The following conversion is supported by `as`:

-   `T` -> `U` if `T` is
    [compatible](../generics/terminology.md#compatible-types) with `U`.

**Future work:** We may need a mechanism to restrict which conversions between
adapters are permitted and which code can perform them. Some of the conversions
permitted by this rule may only be allowed in certain contexts.

### Pointer conversions

The following pointer conversion is supported by `assume_as`:

-   `T*` -> `U*` if `U` is a subtype of `T`.

This cast converts in the opposite direction to the corresponding
[implicit conversion](implicit_conversions.md#pointer-conversions). It is a
programming error if the source pointer does not point to a `U` object.

**Note:** `assume_as` cannot convert between unrelated pointer types, because
there are no input values for which the conversion would produce a well-defined
output value. Separate facilities will be provided for reinterpreting memory as
a distinct type.

## Extensibility

Explicit casts can be defined for user-defined types such as
[classes](../classes.md) by implementing the `As` or `AssumeAs` interface:

```
interface AssumeAs(Dest:! Type) {
  fn Convert[me: Self]() -> Dest;
}
interface As(Dest:! Type) extends AssumeAs(Dest) {
  // Inherited from AssumeAs(Dest):
  // fn Convert[me: Self]() -> Dest;
}
```

The expression `x as U` is rewritten to `x.(As(U).Convert)()`. The expression
`x assume_as U` is rewritten to `x.(AssumeAs(U).Convert)()`.

## Alternatives considered

-   [Do not distinguish between safe and unsafe casts](/docs/proposals/p0845.md#merge-as-and-assume_as)
-   [Do not distinguish between `as` and implicit conversions](/docs/proposals/p0845.md#as-only-performs-implicit-conversions)
-   [Use a different name for `assume_as`](/docs/proposals/p0845.md#different-name-for-assume_as)
-   [Allow `iN as bool`](/docs/proposals/p0845.md#integer-to-bool-conversions)
-   [Disallow `bool as iN`](/docs/proposals/p0845.md#bool-to-integer-conversions)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#845: `as` expressions](https://github.com/carbon-language/carbon-lang/pull/845).
