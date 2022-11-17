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
    -   [Numeric literal syntax](#numeric-literal-syntax)
    -   [Defined Types](#defined-types)
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

### Numeric literal syntax

Numeric Literal syntax is covered in the
[numeric literal lexical conventions](lexical_conventions/numeric_literals.md)
doc. Both Integer and Real-Number syntax is defined, with decimal, hexadecimal
and binary integer literals, and decimal and hexadecimal real number literals.

### Defined Types

The syntax for a two's complement signed integer, the unsigned integer, and the
floating-point number corresponds to a lowercase 'i', 'u', or 'f' character,
respectively, indicating the type followed by a numeric value specifying the
width.

As a regular expression, this can be illustrated as:

```re
([iuf])([1-9][0-9]*)
```

Capture group 1 indicates either an 'i' for a two's complement signed integer
type, a 'u' for an unsigned integer type, or an 'f' for an
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) binary floating-point number
type. Capture group 2 specifies the width in bits. Note that this bit width is
restricted to a multiple of 8.

Conversions in which `x` lies exactly half-way between two values are rounded to
the value in which the mantissa is even, as defined in the IEEE 754 standard and
as was decided in
[proposal #866](https://github.com/carbon-language/carbon-lang/pull/866).

Conversions in which `x` is outside the range of finite values of the
floating-point type are rejected rather than saturating to the finite range or
producing an infinity.

Examples of this syntax include:

-   `i16` - A 16-bit two's complement signed integer type
-   `u32` - A 32-bit unsigned integer type
-   `f64` - A 64-bit IEEE-754 binary floating-point number type

## Examples

```carbon
package sample api;

fn Sum(x: i32, y: i32) -> i32 {
  return x + y;
}

fn Main() -> i32 {
  return Sum(4, 2);
}
```

In the above example, `Sum` has parameters `x` and `y`, each of which is typed
as a 32-bit two's complement signed integer. `Main` then returns the output of
`Sum` as a 32-bit two's complement signed integer.

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
