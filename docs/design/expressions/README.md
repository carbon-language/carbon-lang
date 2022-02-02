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
    -   [Unqualified names](#unqualified-names)
    -   [Qualified names and member access](#qualified-names-and-member-access)
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

## Names

### Unqualified names

An _unqualified name_ is a [word](../lexical_conventions/words.md) that is not a
keyword and is not preceded by a period (`.`).

**TODO:** Name lookup rules for unqualified names.

### Qualified names and member access

A _qualified name_ is a word that is prefixed by a period. Qualified names
appear in the following contexts:

-   [Designators](/docs/design/classes.md#literals): `.` _word_
-   [Direct member access expressions](member_access.md): _expression_ `.`
    _word_

```
var x: auto = {.hello = 1, .world = 2};
                ^^^^^       ^^^^^ qualified name
               ^^^^^^      ^^^^^^ designator

x.hello = x.world;
  ^^^^^     ^^^^^ qualified name
^^^^^^^   ^^^^^^^ member access expression
```

Qualified names refer to members of an entity determined by the context in which
the expression appears. For a member access, the entity is named by the
expression preceding the period. In a struct literal, the entity is the struct
type. For example:

```
package Foo api;
namespace N;
fn N.F() {}

fn G() {
  // Same as `(Foo.N).F()`.
  // `Foo.N` names namespace `N` in package `Foo`.
  // `(Foo.N).F` names function `F` in namespace `N`.
  Foo.N.F();
}

// `.n` refers to the member `n` of `{.n: i32}`.
fn H(a: {.n: i32}) -> i32 {
  // `a.n` is resolved to the member `{.n: i32}.n`,
  // and names the corresponding subobject of `a`.
  return a.n;
}

fn J() {
  // `.n` refers to the member `n of `{.n: i32}`.
  H({.n = 5 as i32});
}
```

Member access expressions associate left-to-right. If the member name is more
complex than a single _word_, an indirect member access expression can be used,
with parentheses around the member name:

-   _expression_ `.` `(` _member-access-expression_ `)`

```
interface I { fn F[me: Self](); }
class X {}
external impl X as I { fn F[me: Self]() {} }

// `x.I.F()` would mean `(x.I).F()`.
fn Q(x: X) { x.(I.F)(); }
```

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
