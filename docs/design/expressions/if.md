# `if` expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Syntax](#syntax)
-   [Semantics](#semantics)
-   [Finding a common type](#finding-a-common-type)
    -   [Same type](#same-type)
    -   [Implicit conversions](#implicit-conversions)
    -   [Facet types](#facet-types)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An `if` expression is an expression of the form:

> `if` _condition_ `then` _value1_ `else` _value2_

The _condition_ is implicitly converted to `bool`. The _value1_ and _value2_ are
converted to a [common type](#finding-a-common-type), which is the type of the
`if` expression.

## Syntax

An `if` expression can appear anywhere a parenthesized expression can appear.
The _value1_ and _value2_ expressions are arbitrary expressions. _value2_
extends as far to the right as possible.

```
// OK, same as `3 * (if cond then (1 + 1) else (2 + (4 * 6)))`
var a: i32 = 3 * if cond then 1 + 1 else 2 + 4 * 6;
```

An `if` keyword at the start of a statement is always interpreted as an
[`if` statement](/docs/design/control_flow/conditionals.md), never as an `if`
expression, even if it is followed eventually by a `then` keyword.

## Semantics

The converted _condition_ is evaluated. If it evaluates to `true`, then the
converted _value1_ is evaluated and its value is the result of the expression.
Otherwise, the converted _value2_ is evaluated and its value is the result of
the expression.

## Finding a common type

The common type of two types `T` and `U` is determined by the
`Carbon.CommonType` interface:

```
interface CommonType(U:! Type) {
  let Result:! Type;
}
```

When attempting to find a common type in an `if` expression:

-   If the operands are lvalues of types `A` and `B`, an attempt to find the
    common type of `A*` and `B*` is made. If successful, the expression is
    rewritten as

    `*(if` _condition_ `then` `&(` _value1_ `)` `else` `&(` _value2_ `))`

-   Otherwise, or if the pointer types have no common type, the common types of
    the operand types is determined, and both operands are converted to that
    type.

### Same type

If `T` is the same type as `U`, the result is that type:

```
impl [T:! Type] T as CommonType(T) {
  let Result:! Type = T;
}
```

### Implicit conversions

If one of `T` and `U` converts to the other, the result is the destination type:

```
impl [U:! Type, T:! ImplicitAs(U)] T as CommonType(U) {
  let Result:! Type = U;
}
impl [T:! Type, U:! ImplicitAs(T)] T as CommonType(U) {
  let Result:! Type = T;
}
impl [template T:! Type, template U:! ImplicitAs(T) where T is ImplicitAs(U)]
    T as CommonType(U) {
  // Produce an ambiguity error.
}
```

**Note:** The intent is that the result is an ambiguity error if an implicit
conversion in both directions is possible and no more specific rule matches.

### Facet types

If `T` and `U` are both facets of the same type, corresponding to constraints
`C` and `D`, the result is the facet type corresponding to the constraint
`C & D`.

```
impl [T:! Type, U:! FacetOf(T)] T as CommonType(U) {
  let Result:! Type = T as (typeof(T) & typeof(U));
}
```

**Note:** The intent is that this should be considered more specialized than the
implicit conversion case above, because `U:! FacetOf(T)` implies that there are
implicit conversions in both directions.

## Alternatives considered

-   [Provide no conditional operator](/proposals/p0911.md#no-conditional-operator)
-   [Use `?:`, like in C and C++](/proposals/p0911.md#use-c-syntax)
-   [Use `if (...) expr1 else expr2`](/proposals/p0911.md#no-then)

## References

-   Proposal
    [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).
