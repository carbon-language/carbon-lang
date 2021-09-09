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
    -   `i113` or `u113` (or smaller) -> `f128`
    -   `i237` or `u237` (or smaller) -> `f256`

Further, a constant expression of type `iN`, `uN`, or `fN` can be converted to
any other type `iM`, `uM`, or `fM` in which that value can be exactly
represented.

In each case, the numerical value is the same before and after the conversion.
An integer zero is translated into a floating-point positive zero.

The above conversions are precisely those that C++ considers non-narrowing,
except that Carbon also permits integer to floating-point conversions in more
cases. The most important of these is that Carbon permits `i32` to be implicitly
converted to `f64`.

The following pointer conversion is available:

-   `T*` -> `U*` if `T` is a subtype of `U`

### Type-of-types

A type `T` with [type-of-type](../generics/terminology.md#type-of-type) `TT1`
can be implicitly converted to the type-of-type `TT2` if `T`
[satisfies the requirements](../generics/details.md#subtyping-between-type-of-types)
of `TT2`.

## Semantics

An implicit conversion of an expression `E` to type `T`, when permitted, always
has the same meaning as the explicit cast expression `E as T`.

**Note:** The explicit cast expression syntax has not yet been decided. The use
of `E as T` here is provisional.

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

## Alternatives considered

-   [Provide lossy and non-semantics-preserving implicit conversions from C++](/docs/proposals/p0820.md#c-conversions)
-   [Provide no implicit conversions](/docs/proposals/p0820.md#no-conversions)
-   [Provide no extensibility](/docs/proposals/p0820.md#no-extensibility)

## References

-   [Implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion)
-   Proposal
    [#820: implicit cnoversions](https://github.com/carbon-language/carbon-lang/pull/820).
