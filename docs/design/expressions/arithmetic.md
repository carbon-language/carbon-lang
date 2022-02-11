# Arithmetic

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
    -   [Integer types](#integer-types)
    -   [Floating-point types](#floating-point-types)
    -   [Overflow and other error conditions](#overflow-and-other-error-conditions)
    -   [Strings](#strings)
-   [Extensibility](#extensibility)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides a conventional set of arithmetic operators:

```
var a: i32 = 5;
var b: i32 = 3;

// -5
var negative: i32 = -a;
// 8
var sum: i32 = a + b;
// 2
var difference: i32 = a - b;
// 15
var product: i32 = a * b;
// 1
var quotient: i32 = a / b;
// 2
var remainder: i32 = a % b;
```

These operators have predefined meanings for some of Carbon's
[built-in types](#built-in-types).

User-defined types can define the meaning of these operations by
[implementing an interface](#extensibility) provided as part of the Carbon
standard library.

## Precedence and associativity

![Precedence diagram for arithmetic operators](arithmetic-precedence.svg)

Binary `+` and `-` can be freely mixed, and are left-associative.

```
// -2, same as `((1 - 2) + 3) - 4`.
var n: i32 = 1 - 2 + 3 - 4;
```

Binary `*` and `/` can be freely mixed, and are left-associative.

```
// 0.375, same as `((1.0 / 2.0) * 3.0) / 4.0`.
var m: f32 = 1.0 / 2.0 * 3.0 / 4.0;
```

Unary `-` has higher precedence than binary `*`, `/`, and `%`. Binary `*` and
`/` have higher precedence than binary `+` and `-`.

```
// 5, same as `(-1) + ((-2) * (-3))`.
var x: i32 = -1 + -2 * -3;
// Error, parentheses required: no precedence order between `+` and `%`.
var y: i32 = 2 + 3 % 5;
```

## Built-in types

For binary operators, if the operands have different built-in types, they are
converted as follows:

-   If the types are `uN` and `uM`, or they are `iN` and `iM`, the operands are
    converted to the larger type.
-   If one type is `iN` and the other type is `uM`, and `M` < `N`, the `uM`
    operand is converted to `iN`.
-   If one type is `fN` and the other type is `iM` or `uM`, and there is an
    [implicit conversion](implicit_conversions.md#data-types) from the integer
    type to `fN`, then the integer operand is converted to `fN`.

More broadly, if one operand is of built-in type and the other operand can be
implicitly converted to that type, then it is.

A built-in arithmetic operation is performed if, after the above conversion
step, the operands have the same built-in type. The result type is that type.
The result type is never wider than the operands, and the conversions applied to
the operands are always lossless, so arithmetic between a wider unsigned integer
type and a narrower signed integer is not defined.

Although the conversions are always lossless, the arithmetic may still
[overflow](#overflow-and-other-error-conditions).

### Integer types

Signed integer types support all the arithmetic operators. Unsigned integer
types support all arithmetic operators other than unary `-`.

Signed integer arithmetic produces the usual mathematical result. Unsigned
integer arithmetic in `uN` wraps around modulo 2<sup>`N`</sup>.

Division truncates towards zero. The result of the `%` operator is defined by
the equation `a % b == a - a / b * b`.

### Floating-point types

Floating-point types support all the arithmetic operators other than `%`.
Floating-point types in Carbon have IEEE 754 semantics, use the round-to-nearest
rounding mode, and do not set any floating-point exception state.

### Overflow and other error conditions

Integer arithmetic is subject to two classes of error condition:

-   Overflow, where the resulting value is too large to be represented in the
    type, or, for `%`, when the implied multiplication overflows.
-   Division by zero.

Unsigned integer arithmetic cannot overflow, but division by zero is still an
error.

Floating-point arithmetic follows IEEE 754 rules: overflow results in ±∞, and
division by zero results in either ±∞ or, for 0.0 / 0.0, a quiet NaN.

**Note:** All arithmetic operators can overflow for signed integer types. For
example, given a value `v: iN` that is the least possible value for its type,
`-v`, `v + v`, `v - 1`, `v * 2`, `v / -1`, and `v % -1` all result in overflow.

Signed integer overflow and signed or unsigned integer division by zero are
programming errors:

-   In a development build, they will be caught at runtime.
-   In a performance build, the optimizer can assume that such conditions don't
    occur.
-   In a hardened build, the behavior on overflow is defined.

**TODO:** In a hardened build, should we attempt to trap on overflow or simply
give a two's complement result?

### Strings

Binary `+` performs string concatenation.

## Extensibility

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
constraint Addable extends AddableWith(Self) where .Result = Self {}
```

```
// Binary `-`.
interface SubtractableWith(U:! Type) {
  let Result:! Type = Self;
  fn Subtract[me: Self](other: U) -> Result;
}
constraint Subtractable extends SubtractableWith(Self) where .Result = Self {}
```

```
// Binary `*`.
interface MultipliableWith(U:! Type) {
  let Result:! Type = Self;
  fn Multiply[me: Self](other: U) -> Result;
}
constraint Multipliable extends MultipliableWith(Self) where .Result = Self {}
```

```
// Binary `/`.
interface DividableWith(U:! Type) {
  let Result:! Type = Self;
  fn Divide[me: Self](other: U) -> Result;
}
constraint Dividable extends DividableWith(Self) where .Result = Self {}
```

```
// Binary `%`.
interface ModableWith(U:! Type) {
  let Result:! Type = Self;
  fn Mod[me: Self](other: U) -> Result;
}
constraint Modable extends ModableWith(Self) where .Result = Self {}
```

Given `x: T` and `y: U`:

-   The expression `-x` is rewritten to `x.(Negatable.Negate)()`.
-   The expression `x + y` is rewritten to `x.(AddableWith(U).Add)(y)`.
-   The expression `x - y` is rewritten to `x.(SubtractableWith(U).Add)(y)`.
-   The expression `x * y` is rewritten to `x.(MultipliableWith(U).Add)(y)`.
-   The expression `x / y` is rewritten to `x.(DividableWith(U).Add)(y)`.
-   The expression `x % y` is rewritten to `x.(ModableWith(U).Add)(y)`.

Implementations of these interfaces are provided for built-in types, with the
semantics described above.

## Alternatives considered

-   TODO

## References

-   TODO
-   Proposal
    [#???: arithmetic](https://github.com/carbon-language/carbon-lang/pull/???).
