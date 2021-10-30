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
    -   [Symmetry](#symmetry)
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
extends as far to the right as possible. An `if` expression can be parenthesized
if the intent is for _value2_ to end earlier.

```
// OK, same as `3 * (if cond then (1 + 1) else (2 + (4 * 6)))`
var a: i32 = 3 * if cond then 1 + 1 else 2 + 4 * 6;

// OK
var b: i32 = 3 * (if cond then 1 + 1 else 2) + 4 * 6;
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

The common type of two types `T` and `U` is `(T as CommonType(U)).Type`, where
`CommonType` is the `Carbon.CommonType` interface:

```
interface CommonTypeWith(U:! Type) {
  let Result:! Type
    where Self is ImplicitAs(.Self) and
          .Self is ImplicitAs(Self);
}
constraint CommonType(U:! CommonTypeWith(Self)) {
  extends CommonTypeWith(U) where .Result == U.Result;
}
```

_Note:_ It is required that both types implicitly convert to the common type.

`CommonTypeWith` is described in [symmetry](#symmety). Some blanket `impl`s for
these interfaces are provided as part of the prelude. These are described in the
following sections.

When attempting to find a common type in an `if` expression:

-   If the operands are lvalues of types `A` and `B`, an attempt to find the
    common type of `A*` and `B*` is made. If successful, the expression is
    rewritten as

    `*(if` _condition_ `then` `&(` _value1_ `)` `else` `&(` _value2_ `))`

-   Otherwise, or if the pointer types have no common type, the common types of
    the operand types is determined, and both operands are converted to that
    type.

_Note:_ The same mechanism is expected to eventually be used to compute common
types in other circumstances.

### Symmetry

The common type of `T` and `U` is the same as the common type of `U` and `T`.
This is guaranteed by providing two interfaces:

-   `CommonTypeWith` is a primitive interface by which a type can suggest how to
    find a common type with another type. It is not necessarily symmetric.
-   `CommonType` is a constraint expressing that two types have a single common
    type, and that common type doesn't depend on the order in which the types
    appear.

To avoid the need to define `CommonTypeWith` in both directions, a helper
blanket `impl` is provided to generate the reversed form:

```
impl [T:! Type, U:! CommonTypeWith(T)] T as CommonTypeWith(U) {
  let Result:! Type = U.Result;
}
```

For example, given:

```
impl [T:! Type] MyX as CommonTypeWith(T) { // #1
  let Result:! Type = MyX;
}
impl [T:! Type] MyY as CommonTypeWith(T) { // #2
  let Result:! Type = MyY;
}
```

`MyX as CommonTypeWith(MyY)` will select #1, and `MyY as CommonTypeWith(MyX)`
will select #2, but the constraints on `MyX as CommonType(MyY)` will not be met
because result types differ.

_Note:_ This `impl` is ambiguous with the other `impl`s described below.
Additional `impl`s will be provided to resolve the ambiguity in favor of the
other option.

### Same type

If `T` is the same type as `U`, the result is that type:

```
impl [T:! Type] T as CommonTypeWith(T) {
  let Result:! Type = T;
}
```

_Note:_ This rule is intended to be considered more specialized than the other
rules in this document.

### Implicit conversions

If one of `T` and `U` implicitly converts to the other, the result is the
destination type:

```
impl [U:! Type, T:! ImplicitAs(U)] T as CommonTypeWith(U) {
  let Result:! Type = U;
}
impl [T:! Type, U:! ImplicitAs(T)] T as CommonTypeWith(U) {
  let Result:! Type = T;
}
impl [template T:! Type, template U:! ImplicitAs(T) where T is ImplicitAs(U)]
    T as CommonTypeWith(U) {
  let Result:! Type = T;
}
```

_Note:_ If an implicit conversion is possible in both directions, and no more
specific implementation exists, the constraints on `T as CommonType(U)` will not
be met because `(T as CommonTypeWith(U)).Result` and
`(U as CommonTypeWith(T)).Result` will differ. In order to define a common type
for such a case, `CommonTypeWith` implementations in both directions must be
provided to override the blanket `impl`s in both directions:

```
impl MyString as CommonTypeWith(YourString) {
  let Result:! Type = MyString;
}
impl YourString as CommonTypeWith(MyString) {
  let Result:! Type = MyString;
}
var my_string: MyString;
var your_string: YourString;
var also_my_string: String = if cond then my_string else your_string;
```

### Facet types

If `T` and `U` are both facets of the same type, corresponding to constraints
`C` and `D`, the result is the facet type corresponding to the constraint
`C & D`.

```
impl [T:! Type, U:! FacetOf(T)] T as CommonTypeWith(U) {
  let Result:! Type = T as (typeof(T) & typeof(U));
}
```

**Note:** The intent is that this should be considered more specialized than the
implicit conversion case above, because `U:! FacetOf(T)` implies that there are
implicit conversions in both directions.

## Alternatives considered

-   [Provide no conditional expression](/proposals/p0911.md#no-conditional-expression)
-   [Use `?:`, like in C and C++](/proposals/p0911.md#use-c-syntax)
-   [Use `if (...) expr1 else expr2`](/proposals/p0911.md#no-then)

## References

-   Proposal
    [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).
