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

The common type of two types `T` and `U` is `(T as CommonType(U)).Result`, where
`CommonType` is the `Carbon.CommonType` constraint. `CommonType` is notionally
defined as follows:

```
constraint CommonType(U:! CommonTypeWith(Self)) {
  extend CommonTypeWith(U) where .Result == U.Result;
}
```

The actual definition is a bit more complex than this, as described in
[symmetry](#symmetry).

The interface `CommonTypeWith` is used to customize the behavior of
`CommonType`. The implementation `A as CommonTypeWith(B)` specifies the type
that `A` would like to result from unifying `A` and `B`:

```
interface CommonTypeWith(U:! Type) {
  let Result:! Type
    where Self is ImplicitAs(.Self) and
          U is ImplicitAs(.Self);
}
```

_Note:_ It is required that both types implicitly convert to the common type.

Some blanket `impl`s for `CommonTypeWith` are provided as part of the prelude.
These are described in the following sections.

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

The common type of `T` and `U` should always be the same as the common type of
`U` and `T`. This is enforced in two steps:

-   A `SymmetricCommonTypeWith` interface implicitly provides a
    `B as CommonTypeWith(A)` implementation whenever one doesn't exist but an
    `A as CommonTypeWith(B)` implementation exists.
-   `CommonType` is defined in terms of `SymmetricCommonTypeWith`, and requires
    that both `A as SymmetricCommonTypeWith(B)` and
    `B as SymmetricCommonTypeWith(A)` produce the same type.

The interface `SymmetricCommonTypeWith` is an implementation detail of the
`CommonType` constraint. It is defined and implemented as follows:

```
interface SymmetricCommonTypeWith(U:! Type) {
  let Result:! Type
    where Self is ImplicitAs(.Self) and
          U is ImplicitAs(.Self);
}
impl [T:! Type, U:! CommonTypeWith(T)] T as SymmetricCommonTypeWith(U) {
  let Result:! Type = U.Result;
}
impl [U:! Type, T:! CommonTypeWith(U)] T as SymmetricCommonTypeWith(U) {
  let Result:! Type = T.Result;
}
impl [U:! Type, T:! CommonTypeWith(U) where U is CommonTypeWith(T)]
    T as SymmetricCommonTypeWith(U) {
  let Result:! Type = T.Result;
}
```

The `SymmetricCommonTypeWith` interface is not exported, so user-defined `impl`s
can't be defined, and only the two blanket `impl`s above are used. The
`CommonType` constraint is then defined as follows:

```
constraint CommonType(U:! SymmetricCommonTypeWith(Self)) {
  extend SymmetricCommonTypeWith(U) where .Result == U.Result;
}
```

When computing the common type of `T` and `U`, if only one of the types provides
a `CommonTypeWith` implementation, that determines the common type. If both
types do, there is no common type, and the `CommonType` constraint is not met.
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
  final let Result:! Type = T;
}
```

_Note:_ This rule is intended to be considered more specialized than the other
rules in this document.

`T.(CommonType(T)).Result` is always assumed to be `T`, even in contexts where
`T` involves a generic parameter and so the result would normally be an unknown
type whose type-of-type is `Type`.

```
fn F[T:! Hashable](c: bool, x: T, y: T) -> HashCode {
  // OK, type of `if` expression is `T`.
  return (if c then x else y).Hash();
}
```

### Implicit conversions

If `T` implicitly converts to `U`, the common type is `U`:

```
impl [T:! Type, U:! ImplicitAs(T)] T as CommonTypeWith(U) {
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
// The type of `also_my_string` is `MyString`.
var also_my_string: auto = if cond then my_string else your_string;
```

## Alternatives considered

-   [Provide no conditional expression](/proposals/p0911.md#no-conditional-expression)
-   [`cond ? expr1 : expr2`, like in C and C++](/proposals/p0911.md#use-c-syntax)
-   [`if (cond) expr1 else expr2`](/proposals/p0911.md#no-then)
-   [`if (cond) then expr1 else expr2`](/proposals/p0911.md#require-parentheses-around-the-condition)
-   [`(if cond then expr1 else expr2)`](/proposals/p0911.md#require-enclosing-parentheses)
-   [Only require one `impl` to specify the common type if implicit conversions in both directions are possible](/proposals/p0911.md#implicit-conversions-in-both-directions)

## References

-   Proposal
    [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).
