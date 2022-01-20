# Logical operators

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
    -   [Conversions](#conversions)
    -   [Overloading](#overloading)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides three operators to support logical operations on `bool` values:

-   `and` provides a logical AND operation.
    -   `x and y` evaluates to `true` if both operands are `true`.
-   `or` provides a logical OR operation.
    -   `x or y` evaluates to `true` if either operand is `true`.
-   `not` provides a logical NOT operation.
    -   `not x` evaluates to `true` if the operand is `false`.

`and` and `or` are infix binary operators, and use
[short-circuit evaluation](https://en.wikipedia.org/wiki/Short-circuit_evaluation).
`not` is a prefix unary operator.

## Details

### Precedence

`and`, `or`, and `not` have very low precedence. When an expression appearing as
the condition of an `if` uses these operators unparenthesized, they are always
the lowest precedence operators in that expression.

These operators permit any reasonable operator that might be used to form a
`bool` value as a subexpression. In particular, comparison operators such as `<`
and `==` have higher precedence than all logical operators.

A `not` operator can be used within `and` and `or`, but `and` cannot be used
directly within `or` without parentheses, nor the other way around.

For example:

```carbon
// ✅ Valid: Operator precedence is unambiguous.
if (n + m == 3 and not n < m) {
  ...
}
// The above is equivalent to:
if (((n + m) == 3) and (not (n < m))) {
  ...
}

// ❌ Invalid: Combines `and` and `or` without parentheses.
if (cond1 and cond2 or cond3) {
  ...
}
// ✅ Valid: Parentheses remove ambiguity.
if (cond1 and (cond2 or cond3)) {
  ...
}

// ✅ Valid: `not` is lower precedence than `==` so this is okay.
if (not cond1 == cond2) {
  ...
}
// The above is equivalent to:
if (not (cond1 == cond2)) {
  ...
}
// ❌ Invalid: `not` precedence relative to `==` prevents this use.
if (cond1 == not cond2) {
  ...
}
// ✅ Valid: Parentheses address precedence.
if (cond1 == (not cond2)) {
  ...
}
```

### Associativity

`and` and `or` are left-associative. A `not` expression cannot be the operand of
another `not` expression; `not not b` is an error without parentheses.

```
// ✅ Valid: No associativity issues, and precedence is fine.
if (not a and not b and not c) {
  ...
}
// The above is equivalent to:
if ((not a) and (not b) and (not c)) {
  ...
}
// ✅ Valid: Parentheses avoid the `not` associativity error.
if (not (not a)) {
  ...
}

// ❌ Invalid: `not not` associativity requires parentheses.
if (not not a) {
  ...
}
```

### Conversions

> TODO: This should be addressed through a standard `bool` conversion design.

The operand of `and`, `or`, or `not` is converted to a `bool` value in the same
way as the condition of an `if` expression. In particular:

-   If we decide that certain values, such as pointers or integers, should not
    be usable as the condition of an `if` without an explicit comparison against
    null or zero, then those values will also not be usable as the operand of
    `and`, `or`, or `not` without an explicit comparison.
-   If an extension point is provided to determine how to branch on the truth of
    a value in an `if` (such as by supplying a conversion to a `bool` type),
    that extension point will also apply to `and`, `or`, and `not`.

### Overloading

The logical operators `and`, `or`, and `not` are not overloadable. As noted
above, any mechanism that allows types to customize how `if` treats them will
also customize how `and`, `or`, and `not` treats them.

## Alternatives considered

-   [Use punctuation spelling for all three operators](/proposals/p0680.md#use-punctuation-spelling-for-all-three-operators)
-   [Precedence of AND versus OR](/proposals/p0680.md#precedence-of-and-versus-or)
-   [Precedence of NOT](/proposals/p0680.md#precedence-of-not)
-   [Punctuation form of NOT](/proposals/p0680.md#punctuation-form-of-not)
-   [Two forms of NOT](/proposals/p0680.md#two-forms-of-not)
-   [Repeated NOT](/proposals/p0680.md#repeated-not)
-   [AND and OR produce the decisive value](/proposals/p0680.md#and-and-or-produce-the-decisive-value)

## References

-   Proposal
    [#680: And, or, not](https://github.com/carbon-language/carbon-lang/pull/680).
