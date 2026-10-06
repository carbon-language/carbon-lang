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
    -   [`typeof`](#typeof)
    -   [Precedence](#precedence)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides the following operators to transform or produce types:

-   `const` as a prefix unary operator produces a `const`-qualified type.
-   `*` as a postfix unary operator produces a pointer _type_ to some other
    type.
-   [`typeof(x)`](#typeof) produces the static type of the expression `x`.

The pointer type operator is also covered as one of the
[pointer operators](pointer_operators.md).

## Details

The semantic details of both `const`-qualified types and pointer types are
provided as part of the [values](/docs/design/values.md) design:

-   [`const`-qualified types](/docs/design/values.md#const-qualified-types)
-   [Pointers](/docs/design/values.md#pointers)

The syntax of these operators tries to mimic the most common appearance of
`const` types and pointer types in C++.

### `typeof`

We provisionally define `typeof(x)` to give the static type of the expression
`x` without any runtime evaluation of `x`.

`typeof(x)` has no runtime side effects, and produces a compile-time result.
This may involve compile-time evaluation, but all runtime effects from that
evaluation are discarded before code generation, as if the code is in an
`if (false)` block. For example:

```carbon
musteval fn P(T: type) -> type {
  return T*;
}

fn F[template T: type](ref x: T) -> P(T) {
  x += 1;
  return &x;
}

fn Call() {
  var y: i32 = 0;
  // Involves the compile-time evaluation of `P(i32)`,
  // and forming a specific instance of `F`. However,
  // `F(ref y)` is not called at runtime.
  StaticAssert(typeof(F(ref y)) == i32*);
  Assert(y == 0);
}
```

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

-   [Alternative pointer syntaxes](/proposals/p002006-values-variables-pointers-and-references.md#alternative-pointer-syntaxes)
-   [Alternative syntaxes for locals](/proposals/p002006-values-variables-pointers-and-references.md#alternative-syntaxes-for-locals)
-   [Make `const` a postfix rather than prefix operator](/proposals/p002006-values-variables-pointers-and-references.md#make-const-a-postfix-rather-than-prefix-operator)

## References

-   [Proposal #2006: Values, variables, and pointers](/proposals/p002006-values-variables-pointers-and-references.md)
-   [Proposal #7697: Updates to member access](/proposals/p007697-updates-to-member-access.md)
