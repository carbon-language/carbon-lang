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
    -   [Commutativity and associativity](#commutativity-and-associativity)
    -   [Same type](#same-type)
    -   [Implicit conversions](#implicit-conversions)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An `if` expression is an expression of the form:

> `if` _condition_ `then` _value1_ `else` _value2_

The _condition_ is converted to a `bool` value in the same way as the condition
of an `if` statement.

> **Note:** These conversions have not yet been decided.

The _value1_ and _value2_ are implicitly converted to their
[common type](#finding-a-common-type), which is the type of the `if` expression.

## Syntax

`if` expressions have very low precedence, and cannot appear as the operand of
any operator, except as the right-hand operand in an assignment. They can appear
in other context where an expression is permitted, such as within parentheses,
as the operand of a `return` statement, as an initializer, or in a
comma-separated list such as a function call.

The _value1_ and _value2_ expressions are arbitrary expressions, and can
themselves be `if` expressions. _value2_ extends as far to the right as
possible. An `if` expression can be parenthesized if the intent is for _value2_
to end earlier.

```
// ✅ OK, same as `if cond then (1 + 1) else (2 + (4 * 6))`
var a: i32 = if cond then 1 + 1 else 2 + 4 * 6;

// ✅ OK
var b: i32 = (if cond then 1 + 1 else 2) + 4 * 6;
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

The common type of two types `T` and `U` is `(T as CommonTypeWith(U)).Result`,
where `CommonTypeWith` is the `Carbon.CommonTypeWith` constraint.
`CommonTypeWith` is defined as follows:

```
interface CommonTypeWith(U:! Type) {
  let Result:! Type where
    Self is ImplicitAs(.Self) and
    U is ImplicitAs(.Self);
  impl U as CommonTypeWith(Self);
}
```

The implementation `A as CommonTypeWith(B)` specifies the type that `A` would
like to result from unifying `A` and `B` as its `Result`.

_Note:_ It is required that both types implicitly convert to the common type.
Some blanket `impl`s for `CommonTypeWith` are provided as part of the prelude.
These are described in the following sections.

_Note:_ The same mechanism is expected to eventually be used to compute common
types in other circumstances.

### Commutativity and associativity

The common type of `T` and `U` should always be the same as the common type of
`U` and `T`. When implementing `CommonTypeWith`, you should ensure this by
providing `impl`s in both directions:

```
class Duration {};
let ZeroType:! Type = // FIXME: type of 0 literal
impl ZeroType as ImplicitAs(Duration) {
  fn Convert[me: Self]() { return {}; }
}
impl Duration as CommonTypeWith(ZeroType) where .Result = Duration {}
impl ZeroType as CommonTypeWith(Duration) where .Result = Duration {}
var d1: Duration;
// ✅ OK, `0` is implicitly converted to `Duration`.
var d2: Duration = if false then d1 else 0;
```

Additionally, `CommonTypeWith` is should be associative where feasible: the
common type of (the common type of `T` and `U`) and `V` is expected to be the
same as the common type of `T` and (the common type of `U` and `V`. However,
this may not be possible to ensure in general: for example, if `T` and `V` come
from different libraries that both depend on `U`, they may have no way to ensure
that both queries produce the same result.

In order to help you remember to define `CommonTypeWith` in both directions, the
interface includes a constraint that a reverse `impl` exists. However, there is
no static enforcement that the reverse `impl` provides the same type, nor any
static enforcement of the associativity property.

### Same type

If `T` is the same type as `U`, the result is that type:

```
final impl [T:! Type] T as CommonTypeWith(T) {
  let Result:! Type = T;
}
```

_Note:_ This rule is intended to be considered more specialized than the other
rules in this document.

Because this `impl` is declared `final`, `T.(CommonTypeWith(T)).Result` is
always assumed to be `T`, even in contexts where `T` involves a generic
parameter and so the result would normally be an unknown type whose type-of-type
is `Type`.

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
specific implementation exists, the constraints on `T as CommonTypeWith(U)` will
not be met because `(T as CommonTypeWith(U)).Result` and
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
-   Use
    [`cond ? expr1 : expr2`, like in C and C++](/proposals/p0911.md#use-c-syntax)
    syntax
-   Use [`if (cond) expr1 else expr2`](/proposals/p0911.md#no-then) syntax
-   Use
    [`if (cond) then expr1 else expr2`](/proposals/p0911.md#require-parentheses-around-the-condition)
    syntax
-   Allow
    [`1 + if cond then expr1 else expr2`](/proposals/p0911.md#never-require-enclosing-parentheses)
-   [Only require one `impl` to specify the common type if implicit conversions in both directions are possible](/proposals/p0911.md#implicit-conversions-in-both-directions)
-   [Introduce special rules for lvalue conditionals](/proposals/p0911.md#support-lvalue-conditionals)

## References

-   Proposal
    [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).
