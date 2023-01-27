<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Operator Tokens

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Tokens](#tokens)
-   [Alternatives Considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An _operator_ is a mathematical/logical symbol that is useful for operator
overloading like `+`, `*`, `<`, and more.

There are specific keywords operators (`and`, `or`, `not`, `yield`) that beyond
value computation and are used for control flow.

Operator sets are fixed and follow the
[maximal munch](https://en.wikipedia.org/wiki/Maximal_munch) principle that is
`a =- b` is read as invalid rather than `a = (-b)`

## Tokens

The initial tokens interpreted. More to be added as needed.

     Operators Tokens

| Token | Explanation                   |
| ----- | ----------------------------- |
| `*`   | Indirection or Multiplication |
| `&`   | Address-of or Bitwise AND     |
| `=`   | Assignment                    |
| `->`  | Return type                   |
| `=>`  | Match syntax                  |
| `[]`  | Subscript                     |
| `()`  | Function call                 |
| `{}`  | Initialization                |
| `,`   | Comma                         |
| `.`   | Member access                 |
| `:`   | Scope                         |

## Alternatives Considered

-   [Using the longest sequence of symbols rather than the longest known.](/proposals/p0601.md#alternatives-considered)
-   [Allow extensible operator set for developer created operators](/proposals/p0601.md/#alternatives-considered)

## References

-   Proposal
    [#601: Operator tokens](https://github.com/carbon-language/carbon-lang/pull/601)
