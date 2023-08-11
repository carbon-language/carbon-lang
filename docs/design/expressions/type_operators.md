# Type operators

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
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides the following operators to transform types:

-   `const` as a prefix unary operator produces a `const`-qualified type.
-   `*` as a postfix unary operator produces a pointer _type_ to some other
    type.

The pointer type operator is also covered as one of the
[pointer operators](pointer_operators.md).

## Details

The semantic details of both `const`-qualified types and pointer types are
provided as part of the [values](/docs/design/values.md) design:

-   [`const`-qualified types](/docs/design/values.md#const-qualified-types)
-   [Pointers](/docs/design/values.md#pointers)

The syntax of these operators tries to mimic the most common appearance of
`const` types and pointer types in C++.

### Precedence

Because these are type operators, they don't have many precedence relationship
with non-type operators.

-   `const` binds more tightly than `*` and can appear unparenthesized in an
    operand, despite being both a unary operator and having whitespace
    separating it.
    -   This allows the syntax of a pointer to a `const i32` to be `const i32*`,
        which is intended to be familiar to C++ developers.
    -   Forming a `const` pointer type requires parentheses: `const (i32*)`.
-   All type operators bind more tightly than `as` so they can be used in its
    type operand.
    -   This also allows a desirable transitive precedence with `if`:
        `if condition then T* else U*`.

## Alternatives considered

-   [Alternative pointer syntaxes](/proposals/p2006.md#alternative-pointer-syntaxes)
-   [Alternative syntaxes for locals](/proposals/p2006.md#alternative-syntaxes-for-locals)
-   [Make `const` a postfix rather than prefix operator](/proposals/p2006.md#make-const-a-postfix-rather-than-prefix-operator)

## References

-   [Proposal #2006: Values, variables, and pointers](/proposals/p2006.md)
