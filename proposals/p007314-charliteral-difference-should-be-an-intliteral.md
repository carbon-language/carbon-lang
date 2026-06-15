# CharLiteral difference should be an IntLiteral

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7314)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Example](#example)
-   [Rationale](#rationale)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)
    -   [Use a fixed result type](#use-a-fixed-result-type)

<!-- tocstop -->

## Abstract

Change the result type `CharLiteral - CharLiteral` from `i32` to
`Core.IntLiteral`.

## Problem

Per [p006710](/proposals/p006710-char-redesign.md), the subtraction of two
character literals (for example `'b' - 'a'`) is of type `i32`. This introduces
two main problems:

1.  **Inconsistency with other literals**: In Carbon, operations on literals
    (like `IntLiteral - IntLiteral`) yield other literal types (such as
    `IntLiteral`). This allows them to be used in template contexts and to be
    implicitly converted to other types as appropriate. Having
    `CharLiteral - CharLiteral` produce a concrete, fixed type like `i32` is an
    exception to this rule. For example, `'b' - 'a'` should be usable as a
    template argument of type `Core.IntLiteral` and should be implicitly
    convertible to any integer type (like `i64` or `u8`) depending on the value
    of the difference, rather than being locked to `i32`.
2.  **Layering issue in the toolchain**: In the toolchain implementation, `Core`
    types and arithmetic operators are defined in a layered, sequential fashion
    in the prelude, and this layering is enforced by the orphan rule.
    `Core.CharLiteral` and `Core.IntLiteral` are fundamental, primitive types
    that the compiler knows about intrinsically. However, `i32` is not a
    primitive type; it is a type alias for `Core.Int(32)`, which is defined in
    `core/prelude/types/int.carbon`. If `CharLiteral - CharLiteral` produces
    `i32`, then the definition of `CharLiteral` operations in
    `core/prelude/operators/arithmetic.carbon` must depend on `i32`, creating a
    circular dependency where the definition of primitive arithmetic operators
    needs to know about a specific concrete integer type that hasn't been fully
    defined yet. The only solution to this that satisfies the orphan rule is
    reverse the dependency edge, so `arithmetic.carbon` depends on `int.carbon`,
    and to define all `Core.Int` operations in `arithmetic.carbon`. By producing
    `Core.IntLiteral`, we keep the integer types self-contained.

## Background

In [p006710-char-redesign](/proposals/p006710-char-redesign.md), character
literals were redesigned. As part of that, the difference of two character
literals was defined as producing `i32`, with the rationale:

> The difference of two characters produces an `i32`. This is preferred even for
> `char` to be consistent with the range needed to represent the difference of
> two `Core.CharLiteral` values.

`CharLiteral` values are in the range [0, 0x10FFFF], so the smallest fixed-width
power-of-two-sized type their differences fit within is indeed `i32`. However, we
did not consider using a literal type, nor the layering impact of this choice.

## Proposal

Change the type of `CharLiteral - CharLiteral` to be `IntLiteral`.

### Example

With this change, subtracting two character literals results in an `IntLiteral`,
which is evaluated at compile time and can be implicitly converted to any
integer type that can represent the value:

```carbon
// 'b' - 'a' produces an IntLiteral with value 1.
// This can initialize a u8, i64, etc.
var offset_u8: u8 = 'b' - 'a';
var offset_i64: i64 = 'b' - 'a';

// Error, `u8` cannot represent -1.
var offset_u8: u8 = 'a' - 'b';

// OK, result fits in `u16`.
var table_size: u16 = '\u{10FFFF}' - '\0' + 1;
```

## Rationale

This proposal advances Carbon's goals and principles in the following ways:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    By resolving the layering issue, we simplify the prelude implementation.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    The change provides consistency across all literal types. Developers do not
    need to remember that subtracting character literals produces a concrete
    `i32` while other literal operations produce literal types. It also enables
    more use cases, such as using the difference of two character literals to
    initialize an unsigned type without explicit and potentially lossy casts.

## Future work

This proposal does not consider making any changes to the type of `char - char`,
which will therefore continue to produce `i32`.

This is somewhat arbitrary, especially as the old rationale of consistency with
`CharLiteral` is gone. The following options may be worth considering in a
future proposal:

-   Use `i9`, as the smallest type that can fit all possible results.
-   Use `i16`, as the smallest power-of-2 type that can fit all possible
    results.
-   Keep `i32`, as it is expected to be the "normal" integer type.

## Alternatives considered

### Use a fixed result type

We could keep the result type of `CharLiteral - CharLiteral` as `i32` as
originally specified in
[p006710-char-redesign](/proposals/p006710-char-redesign.md), or similarly pick
a different fixed type such as `i16`.

-   **Disadvantages**:
    -   Preserves the layering issue in the toolchain.
    -   Creates an inconsistency in the language design between character
        literals and other literal types.
    -   Requires explicit casts (for example, `as u8` or `as i64`) when the
        difference is used to initialize non-`i32` types, even though the value
        is a compile-time constant that is guaranteed to fit in the target type.
