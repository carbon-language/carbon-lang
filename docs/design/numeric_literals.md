# Numeric Literal Semantics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

> **STATUS:** Up-to-date on 23-Aug-2022.

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [TODO](#todo)
    -   [Numeric literal syntax](#numeric-literal-syntax)
    -   [Defined Types](#defined-types)
    -   [Implicit conversions](#implicit-conversions)
-   [Examples](#examples)
-   [Alternatives Considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Numeric Literals are defined on Wikipedia
[here](<https://en.wikipedia.org/wiki/Literal_(computer_programming)>).

In Carbon, numeric literals have a type derived from their value. Two integer
literals have the same type if and only if they represent the same integer. Two
real number literals have the same type if and only if they represent the same
real number.

That is:

-   For every integer, there is a type representing literals with that integer
    value.
-   For every rational number, there is a type representing literals with that
    real value.
-   The types for real numbers are distinct from the types for integers, even
    for real numbers that represent integers. For example, `1 / 2` results in
    `0`, due to integer arithmetic, whereas `1.0 / 2` results in `0.5`. This is
    due to `1` having an integral type, while `1.0` has a real number type, even
    though it represents the same numeric value.

Primitive operators are available between numeric literals, and produce values
with numeric literal types. For example, the type of `1 + 2` is the same as the
type of `3`.

Numeric types can provide conversions to support initialization from numeric
literals. Because the value of the literal is carried in the type, a type-level
decision can be made as to whether the conversion is valid.

The integer types defined in the standard library permit conversion from integer
literal types whose values are representable in the integer type. The
floating-point types defined in the standard library permit conversion from
integer and rational literal types whose values are between the minimum and
maximum finite value representable in the floating-point type.

### TODO

This document needs to be updated once we have resolved how to reference things
brought in by the prelude. `BigInt`, `Rational`, `IntLiteral`, and
`FloatLiteral` will likely be accessed through a package prefix like
`Carbon.BigInt` or `Core.BigInt`, and the [Defined Types](#defined-types)
section will need to be updated to reflect those.

### Numeric literal syntax

Numeric Literal syntax is covered in the
[numeric literal lexical conventions](lexical_conventions/numeric_literals.md)
doc. Both Integer and Real-Number syntax is defined, with decimal, hexadecimal
and binary integer literals, and decimal and hexadecimal real number literals.

### Defined Types

The following types are defined in the Carbon prelude:

-   An arbitrary-precision integer type.

    ```
    class BigInt;
    ```

-   A rational type, parameterized by a type used for its numerator and
    denominator.

    ```
    class Rational(T:! Type);
    ```

    The exact constraints on `T` are not yet decided.

-   A type representing integer literals.

    ```
    class IntLiteral(N:! BigInt);
    ```

-   A type representing floating-point literals.

    ```
    class FloatLiteral(X:! Rational(BigInt));
    ```

All of these types are usable during compilation. `BigInt` supports the same
operations as `Int(n)`. `Rational(T)` supports the same operations as
`Float(n)`.

The types `IntLiteral(n)` and `FloatLiteral(x)` also support primitive integer
and floating-point operations such as arithmetic and comparison, but these
operations are typically heterogeneous: for example, an addition between
`IntLiteral(n)` and `IntLiteral(m)` produces a value of type
`IntLiteral(n + m)`.

### Implicit conversions

`IntLiteral(n)` converts to any sufficiently large integer type, as if by:

```
impl forall [template N:! BigInt, template M:! BigInt]
    IntLiteral(N) as ImplicitAs(Carbon.Int(M))
    if N >= Carbon.Int(M).MinValue as BigInt and N <= Carbon.Int(M).MaxValue as BigInt {
  ...
}
impl forall [template N:! BigInt, template M:! BigInt]
    IntLiteral(N) as ImplicitAs(Carbon.UInt(M))
    if N >= Carbon.UInt(M).MinValue as BigInt and N <= Carbon.UInt(M).MaxValue as BigInt {
  ...
}
```

The above is for exposition purposes only; various parts of this syntax are not
yet decided.

Similarly, `IntLiteral(x)` and `FloatLiteral(x)` convert to any sufficiently
large floating-point type, and produce the nearest representable floating-point
value.

Conversions in which `x` lies exactly half-way between two values are rounded to
the value in which the mantissa is even, as defined in the IEEE 754 standard and
as was decided in
[proposal #866](https://github.com/carbon-language/carbon-lang/pull/866).

Conversions in which `x` is outside the range of finite values of the
floating-point type are rejected rather than saturating to the finite range or
producing an infinity.

## Examples

```carbon
// This is OK: the initializer is of the integer literal type with value
// -2147483648 despite being written as a unary `-` applied to a literal.
var x: i32 = -2147483648;

// This initializes y to 2^60.
var y: i64 = 1 << 60;

// This forms a rational literal whose value is one third, and converts it to
// the nearest representable value of type `f64`.
var z: f64 = 1.0 / 3.0;

// This is an error: 300 cannot be represented in type `i8`.
var c: i8 = 300;

fn F[template T:! Type](v: T) {
  var x: i32 = v * 2;
}

// OK: x = 2_000_000_000.
F(1_000_000_000);

// Error: 4_000_000_000 can't be represented in type `i32`.
F(2_000_000_000);

// No storage required for the bound when it's of integer literal type.
struct Span(template T:! Type, template BoundT:! Type) {
  var begin: T*;
  var bound: BoundT;
}

// Returns 1, because 1.3 can implicitly convert to f32, even though conversion
// to f64 might be a more exact match.
fn G() -> i32 {
  match (1.3) {
    case _: f32 => { return 1; }
    case _: f64 => { return 2; }
  }
}

// Can only be called with a literal 0.
fn PassMeZero(_: IntLiteral(0));

// Can only be called with integer literals in the given range.
fn ConvertToByte[template N:! BigInt](_: IntLiteral(N)) -> i8
    if N >= -128 and N <= 127 {
  return N as i8;
}

// Given any int literal, produces a literal whose value is one higher.
fn OneHigher(L: IntLiteral(template _:! BigInt)) -> auto {
  return L + 1;
}
// Error: 256 can't be represented in type `i8`.
var v: i8 = OneHigher(255);
```

## Alternatives Considered

-   [Use an ordinary integer or floating-point type for literals](/proposals/p0144.md#use-an-ordinary-integer-or-floating-point-type-for-literals)
-   [Use same type for all literals](/proposals/p0144.md#use-same-type-for-all-literals)
-   [Allow leading `-` in literal tokens](/proposals/p0144.md#allow-leading---in-literal-tokens)
-   [Forbidding floating-point ties](/proposals/p0866.md/#alternatives-considered)

## References

> -   Proposal
>     [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
> -   Proposal
>     [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
> -   Proposal
>     [#866: Allow ties in floating literals](https://github.com/carbon-language/carbon-lang/pull/866)
> -   Issue
>     [#1998: Make proposal for numeric type literal syntax](https://github.com/carbon-language/carbon-lang/issues/1998#issuecomment-1212644291)
