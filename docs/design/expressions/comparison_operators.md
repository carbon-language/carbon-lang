# Comparison operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Precedence](#precedence)
    -   [Associativity](#associativity)
    -   [Built-in comparisons and implicit conversions](#built-in-comparisons-and-implicit-conversions)
        -   [Consistency with implicit conversions](#consistency-with-implicit-conversions)
        -   [Comparisons with constants](#comparisons-with-constants)
    -   [Overloading](#overloading)
    -   [Default implementations for basic types](#default-implementations-for-basic-types)
-   [Open questions](#open-questions)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides equality and relational comparison operators, each with a
standard mathematical meaning:

| Category   | Operator | Example  | Mathematical meaning | Description                |
| ---------- | -------- | -------- | -------------------- | -------------------------- |
| Equality   | `==`     | `x == y` | =                    | Equality or equal to       |
| Equality   | `!=`     | `x != y` | ≠                    | Inequality or not equal to |
| Relational | `<`      | `x < y`  | <                    | Less than                  |
| Relational | `<=`     | `x <= y` | ≤                    | Less than or equal to      |
| Relational | `>`      | `x > y`  | >                    | Less than                  |
| Relational | `>=`     | `x >= y` | ≥                    | Less than or equal to      |

Comparison operators all return a `bool`; they evaluate to `true` when the
indicated comparison is true. All comparison operators are infix binary
operators.

## Details

### Precedence

The comparison operators are all at the same precedence level. This level is
lower than operators used to compute (non-`bool`) values, higher than the
[logical operators](logical_operators.md) `and` and `or`, and incomparable with
the precedence of `not`.

For example:

```carbon
// ✅ Valid: precedence provides order of evaluation.
if (n + m * 3 < n * n and 3 < m and m < 6) {
  ...
}
// The above is equivalent to:
if (((n + (m * 3)) < (n * n)) and ((3 < m) and (m < 6))) {
  ...
}

// ❌ Invalid due to ambiguity: `(not a) == b` or `not (a == b)`?
if (not a == b) {
  ...
}
// ❌ Invalid due to precedence: write `a == (not b)`.
if (a == not b) {
  ...
}
// ❌ Invalid due to precedence: write `not (f < 5.0)`.
if (not f < 5.0) {
  ....
}
```

### Associativity

The comparison operators are non-associative. For example:

```carbon
// ❌ Invalid: write `3 < m and m < 6`.
if (3 < m < 6) {
  ...
}
// ❌ Invalid: write `a == b and b == c`.
if (a == b == c) {
  ...
}
// ❌ Invalid: write `(m > 1) == (n > 1)`.
if (m > 1 == n > 1) {
  ...
}
```

### Built-in comparisons and implicit conversions

Built-in comparisons are permitted in three cases:

1.  When both operands are of standard Carbon integer types (`Int(n)` or
    `Unsigned(n)`).
2.  When both operands are of standard Carbon floating-point types (`Float(n)`).
3.  When one operand is of floating-point type and the other is of integer type,
    if all values of the integer type can be exactly represented in the
    floating-point type.

In each case, the result is the mathematically-correct answer. This applies even
when comparing `Int(n)` with `Unsigned(m)`.

For example:

```carbon
// ✅ Valid: Fits case #1. The value of `compared` is `true` because `a` is less
// than `b`, even though the result of a wrapping `i32` or `u32` comparison
// would be `false`.
fn Compare(a: i32, b: u32) -> bool { return a < b; }
let compared: bool = Compare(-1, 4_000_000_000);

// ❌ Invalid: Doesn't fit case #3 because `i64` values in general are not
// exactly representable in the type `f32`.
let float: f32 = 1.0e18;
let integer: i64 = 1_000_000_000_000_000_000;
let eq: bool = float == integer;
```

Comparisons involving integer and floating-point constants are not covered by
these rules and are [discussed separately](#comparisons-with-constants).

#### Consistency with implicit conversions

We support the following [implicit conversions](implicit_conversions.md):

-   From `Int(n)` to `Int(m)` if `m > n`.
-   From `Unsigned(n)` to `Int(m)` or `Unsigned(m)` if `m > n`.
-   From `Float(n)` to `Float(m)` if `m > n`.
-   From `Int(n)` to `Float(m)` if `Float(m)` can represent all values of
    `Int(n)`.

These rules can be summarized as: a type `T` can be converted to `U` if every
value of type `T` is a value of type `U`.

Implicit conversions are also supported from certain kinds of integer and
floating-point constants to `Int(n)` and `Float(n)` types, if the constant can
be represented in the type.

All built-in comparisons can be viewed as performing implicit conversions on at
most one of the operands in order to reach a suitable pair of identical or very
similar types, and then performing a comparison on those types. The target types
for these implicit conversions are, for each suitable value `n`:

-   `Int(n)` versus `Int(n)`
-   `Unsigned(n)` versus `Unsigned(n)`
-   `Int(n)` versus `Unsigned(n)`
-   `Unsigned(n)` versus `Int(n)`
-   `Float(n)` versus `Float(n)`

There will in general be multiple combinations of implicit conversions that will
lead to one of the above forms, but we will arrive at the same result regardless
of which is selected, because all comparisons are mathematically correct and all
implicit conversions are lossless. Implementations are expected to do whatever
is most efficient: for example, for `u16 < i32` it is likely that the best
choice would be to promote the `u16` to `i32`, not `u32`.

Because we only ever convert at most one operand, we never use an intermediate
type that is larger than both input types. For example, both `i32` and `f32` can
be implicitly converted to `f64`, but we do not permit comparisons between `i32`
and `f32` even though we could perform those comparisons in `f64`. If such
comparisons were permitted, the results could be surprising:

```carbon
// `i32` can exactly represent this value.
var integer: i32 = 2_000_000_001;
// This value is within the representable range for `f32`, but will be rounded
// to 2_000_000_000.0 due to the limited precision of `f32`.
var float: f32 = 2_000_000_001.0;

// ❌ Invalid: `f32` cannot exactly represent all values of `i32`.
if (integer == float) {
  ...
}

// ✅ Valid: An explicit cast to `f64` on either side makes the code valid, but
// will compare unequal because `float` was rounded to 2_000_000_000.0
// but `integer` will convert to exactly 2_000_000_001.0.
if (integer == float as f64) {
  ...
}
if (integer as f64 == float) {
  ...
}
```

The two kinds of mixed-type comparison may be
[less efficient](/proposals/p0702.md#performance) than the other kinds due to
the slightly wider domain.

Note that this approach diverges from C++, which would convert both operands to
a common type first, sometimes performing a lossy conversion potentially giving
an incorrect result, sometimes converting both operands, and sometimes using a
wider type than either of the operand types.

#### Comparisons with constants

We permit the following comparisons involving constants:

-   A constant can be compared with a value of any type to which it can be
    implicitly converted.
-   Any two constants can be compared, even if there is no type that can
    represent both.

As described in [implicit conversions](implicit_conversions.md#data-types),
integer constants can be implicitly converted to any integer or floating-point
type that can represent their value, and floating-point constants can be
implicitly converted to any floating-point type that can represent their value.

Note that this disallows comparisons between, for example, `i32` and an integer
literal that cannot be represented in `i32`. Such comparisons would always be
tautological. This decision should be revisited if it proves problematic in
practice, for example in templated code where the literal is sometimes in range.

### Overloading

Separate interfaces will be provided to permit overloading equality and
relational comparisons. The exact design of those interfaces is left to a future
proposal. As non-binding design guidance for such a proposal:

-   The interface for equality comparisons should primarily provide the ability
    to override the behavior of `==`. The `!=` operator can optionally also be
    overridden, with a default implementation that returns `not (a == b)`. This
    conversation was marked as resolved by chandlerc Show conversation
    Overriding `!=` separately from `==` is expected to be used to support
    floating-point NaN comparisons and for C++ interoperability.

-   The interface for relational comparisons should primarily provide the
    ability to specify a three-way comparison operator. The individual
    relational comparison operators can optionally be overridden separately,
    with a default implementation in terms of the three-way comparison operator.
    This facility is expected to be used primarily to support C++
    interoperability.

-   Overloaded comparison operators may wish to produce a type other than
    `bool`, for uses such as a vector comparison producing a vector of `bool`
    values. We should decide whether we wish to support such uses.

### Default implementations for basic types

In addition to being defined for standard Carbon numeric types, equality and
relational comparisons are also defined for all "data" types:

-   [Tuples](../tuples.md)
-   [Struct types](../classes.md#struct-types)
-   [Classes implementing an interface that identifies them as data classes.](../classes.md#interfaces-implemented-for-data-classes)

Relational comparisons for these types provide a lexicographical ordering. In
each case, the comparison is only available if it is supported by all element
types.

## Open questions

The `bool` type should be treated as a choice type, and so should support
equality comparisons and relational comparisons if and only if choice types do
in general. That decision is left to a future proposal.

## Alternatives considered

-   [Alternative symbols](/proposals/p0702.md#alternative-symbols)
-   [Chained comparisons](/proposals/p0702.md#chained-comparisons-1)
-   [Convert operands like C++](/proposals/p0702.md#convert-operands-like-c)
-   [Provide a three-way comparison operator](/proposals/p0702.md#provide-a-three-way-comparison-operator)
-   [Allow comparisons as the operand of `not`](/proposals/p0702.md#allow-comparisons-as-the-operand-of-not)

## References

-   Proposal
    [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
-   Issue
    [#710: Default comparison for data classes](https://github.com/carbon-language/carbon-lang/issues/710)
