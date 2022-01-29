# Expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Operators](#operators)
-   [Conversions and casts](#conversions-and-casts)
-   [`if` expressions](#if-expressions)

<!-- tocstop -->

## Overview

Expressions are the portions of Carbon syntax that produce values. Because types
in Carbon are values, this includes anywhere that a type is specified.

```
fn Foo(a: i32*) -> i32 {
  return *a;
}
```

Here, the parameter type `i32*`, the return type `i32`, and the operand `*a` of
the `return` statement are all expressions.

## Operators

Most expressions are modeled as operators:

| Category   | Operator                      | Syntax    | Function                                                            |
| ---------- | ----------------------------- | --------- | ------------------------------------------------------------------- |
| Conversion | [`as`](as_expressions.md)     | `x as T`  | Converts the value `x` to the type `T`.                             |
| Logical    | [`and`](logical_operators.md) | `x and y` | A short-circuiting logical AND: `true` if both operands are `true`. |
| Logical    | [`or`](logical_operators.md)  | `x or y`  | A short-circuiting logical OR: `true` if either operand is `true`.  |
| Logical    | [`not`](logical_operators.md) | `not x`   | Logical NOT: `true` if the operand is `false`.                      |

## Conversions and casts

When an expression appears in a context in which an expression of a specific
type is expected, [implicit conversions](implicit_conversions.md) are applied to
convert the expression to the target type.

Expressions can also be converted to a specific type using an
[`as` expression](as_expressions.md).

```
fn Bar(n: i32);
fn Baz(n: i64) {
  // OK, same as Bar(n as i32)
  Bar(n);
}
```

## `if` expressions

An [`if` expression](if.md) chooses between two expressions.

```
fn Run(args: Span(StringView)) {
  var file: StringView = if args.size() > 1 then args[1] else "/dev/stdin";
}
```

`if` expressions are analogous to `?:` ternary expressions in C and C++.
