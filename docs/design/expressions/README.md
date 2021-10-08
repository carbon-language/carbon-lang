# Expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Names](#names)
-   [Implicit conversions](#implicit-conversions)

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

## Names

Names are a primitive component of Carbon expressions. They come in two forms:

-   Any word that is not a keyword and not preceded by a `.` is an
    [unqualified name](unqualified_names.md), and is looked up in the enclosing
    scopes.
-   Any word that is preceded by a `.` is a qualified name, and is looked up in
    the value to the left of the `.`.

```
// `F`, `a`, and `b` are unqualified names. `c` is a qualified name.
F(a + b.c);
// `x` and `y` are unqualified names. `z` is a qualified name.
x.(y.z)();
```

## Implicit conversions

When an expression appears in a context in which an expression of a specific
type is expected, [implicit conversions](implicit_conversions.md) are applied to
convert the expression to the target type.

```
fn Bar(n: i32);
fn Baz(n: i64) {
  // OK, same as Bar(n as i32)
  Bar(n);
}
```
