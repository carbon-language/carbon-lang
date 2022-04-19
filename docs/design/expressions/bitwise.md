# Bitwise and shift operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Integer types](#integer-types)
-   [Extensibility](#extensibility)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides a conventional set of operators for operating on bits:

```
var a: u8 = 5;
var b: u8 = 3;
var c: i8 = -5;

// 250
var complement: u8 = ^a;
// 1
var bitwise_and: u8 = a & b;
// 7
var bitwise_or: u8 = a | b;
// 6
var bitwise_xor: u8 = a ^ b;
// 40
var left_shift: u8 = a << b;
// 2
var logical_right_shift: u8 = a >> 1;
// -3
var arithmetic_right_shift: i8 = c >> 1;
```

These operators have [predefined meanings](#integer-types) for Carbon's integer
types.

User-defined types can define the meaning of these operations by
[implementing an interface](#extensibility) provided as part of the Carbon
standard library.

## Precedence and associativity

```mermaid
graph TD
    complement["^x"]
    bitwise_and>"x & y"]
    bitwise_or>"x | y"]
    bitwise_xor>"x ^ y"]
    shift["x << y<br>
           x >> y"]
    bitwise_and & bitwise_or & bitwise_xor & shift --> complement
```

<small>[Instructions for reading this diagram.](README.md#precedence)</small>

Parentheses are required when mixing different bitwise and bit-shift operators.
Binary `&`, `|`, and `^` are left-associative. The bit-shift operators `<<` and
`>>` are non-associative.

```
// ✅ Same as (1 | 2) | 4, evaluates to 7.
var a: i32 = 1 | 2 | 4;

// ❌ Error, parentheses are required to distinguish between
//    (3 | 5) & 6, which evaluates to 6, and
//    3 | (5 & 6), which evaluates to 7.
var b: i32 = 3 | 5 & 6;

// ❌ Error, parentheses are required to distinguish between
//    (1 << 2) << 3, which evaluates to 1 << 5 == 32, and
//    1 << (2 << 3), which evaluates to 1 << 16 == 65536.
var c: i32 = 1 << 2 << 3;

// ❌ Error, can't repeat the `^` operator. Use `^(^4)` or simply `4`.
var d: i32 = ^^4;
```

## Integer types

Bitwise and bit-shift operators are supported for Carbon's built-in integer
types, and, unless that behavior is [overridden](#extensibility), for types that
can be implicitly converted to integer types, as follows:

For binary bitwise operators, if one operand has an integer type and the other
operand can be implicitly converted to that type, then it is. If both operands
are of integer type, this results in the following conversions:

-   If the types are `uN` and `uM`, or they are `iN` and `iM`, the operands are
    converted to the larger type.
-   If one type is `iN` and the other type is `uM`, and `M` < `N`, the `uM`
    operand is converted to `iN`.

A built-in bitwise operation is performed if, after the above conversion step,
the operands have the same integer type. The result type is that type, and the
result value is produced by applying the relevant operation -- AND, OR, or XOR
-- to each pair of corresponding bits in the input, including the sign bit for a
signed integer type.

A built-in complement operation is performed if the operand can be implicitly
converted to an integer type. The result type is that type, and the result value
is produced by flipping all bits in the input, including the sign bit for a
signed integer type. `^a` is equivalent to `a ^ x`, where `x` is the
all-one-bits value of the same type as `a`.

A built-in bit-shift operation is performed if both operands are, or can be
implicitly converted to, integer types. The result type is the converted type of
the first operand. The result value is produced by shifting the first operand
left for `<<` or right for `>>` a number of positions equal to the second
operand. Vacant positions are filled with `0` bits, except for a right shift
where the first operand is of a signed type and has a negative value, in which
case they are filled with `1` bits.

The second operand of a bit-shift is required to be between zero (inclusive) and
the bit-width of the first operand (exclusive); it is a programming error if the
second operand is not within that range. In a hardened build, the result will
either be a trap or a correct shift by an unspecified number of bits, which
might still be wider than the first operand, resulting in 0 or -1. In a
performance build, the optimizer may assume that this programming error does not
occur.

## Extensibility

!!! HERE !!!

Arithmetic operators can be provided for user-defined types by implementing the
following family of interfaces:

```
// Unary `-`.
interface Negatable {
  let Result:! Type = Self;
  fn Negate[me: Self]() -> Result;
}
```

```
// Binary `+`.
interface AddableWith(U:! Type) {
  let Result:! Type = Self;
  fn Add[me: Self](other: U) -> Result;
}
constraint Addable {
  extends AddableWith(Self) where .Result = Self;
}
```

```
// Binary `-`.
interface SubtractableWith(U:! Type) {
  let Result:! Type = Self;
  fn Subtract[me: Self](other: U) -> Result;
}
constraint Subtractable {
  extends SubtractableWith(Self) where .Result = Self;
}
```

```
// Binary `*`.
interface MultipliableWith(U:! Type) {
  let Result:! Type = Self;
  fn Multiply[me: Self](other: U) -> Result;
}
constraint Multipliable {
  extends MultipliableWith(Self) where .Result = Self;
}
```

```
// Binary `/`.
interface DividableWith(U:! Type) {
  let Result:! Type = Self;
  fn Divide[me: Self](other: U) -> Result;
}
constraint Dividable {
  extends DividableWith(Self) where .Result = Self;
}
```

```
// Binary `%`.
interface ModuloWith(U:! Type) {
  let Result:! Type = Self;
  fn Mod[me: Self](other: U) -> Result;
}
constraint Modulo {
  extends ModuloWith(Self) where .Result = Self;
}
```

Given `x: T` and `y: U`:

-   The expression `-x` is rewritten to `x.(Negatable.Negate)()`.
-   The expression `x + y` is rewritten to `x.(AddableWith(U).Add)(y)`.
-   The expression `x - y` is rewritten to
    `x.(SubtractableWith(U).Subtract)(y)`.
-   The expression `x * y` is rewritten to
    `x.(MultipliableWith(U).Multiply)(y)`.
-   The expression `x / y` is rewritten to `x.(DividableWith(U).Divide)(y)`.
-   The expression `x % y` is rewritten to `x.(ModuloWith(U).Mod)(y)`.

Implementations of these interfaces are provided for built-in types as necessary
to give the semantics described above.

## Alternatives considered

-   [Use a sufficiently wide result type to avoid overflow](/proposals/p1083.md#use-a-sufficiently-wide-result-type-to-avoid-overflow)
-   [Guarantee that the program never proceeds with an incorrect value after overflow](/proposals/p1083.md#guarantee-that-the-program-never-proceeds-with-an-incorrect-value-after-overflow)
-   [Guarantee that all integer arithmetic is two's complement](/proposals/p1083.md#guarantee-that-all-integer-arithmetic-is-twos-complement)
-   [Treat overflow as an error but don't optimize on it](/proposals/p1083.md#treat-overflow-as-an-error-but-dont-optimize-on-it)
-   [Don't let `Unsigned` arithmetic wrap](/proposals/p1083.md#dont-let-unsigned-arithmetic-wrap)
-   [Provide separate wrapping types](/proposals/p1083.md#provide-separate-wrapping-types)
-   [Do not provide an ordering or division for `uN`](/proposals/p1083.md#do-not-provide-an-ordering-or-division-for-un)
-   [Give unary `-` lower precedence](/proposals/p1083.md#give-unary---lower-precedence)
-   [Include a unary plus operator](/proposals/p1083.md#include-a-unary-plus-operator)
-   [Floating-point modulo operator](/proposals/p1083.md#floating-point-modulo-operator)
-   [Provide different division operators](/proposals/p1083.md#provide-different-division-operators)
-   [Use different division and modulo semantics](/proposals/p1083.md#use-different-division-and-modulo-semantics)
-   [Use different precedence groups for division and multiplication](/proposals/p1083.md#use-different-precedence-groups-for-division-and-multiplication)
-   [Use the same precedence group for modulo and multiplication](/proposals/p1083.md#use-the-same-precedence-group-for-modulo-and-multiplication)
-   [Use a different spelling for modulo](/proposals/p1083.md#use-a-different-spelling-for-modulo)

## References

-   Proposal
    [#1083: arithmetic](https://github.com/carbon-language/carbon-lang/pull/1083).
