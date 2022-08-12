# Operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Symbolic tokens](#symbolic-tokens)
    -   [Details](#details)
    -   [List](#raw-string-literals)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon has a fixed set of tokens that represent operators, defined by the language specification. Developers cannot define new tokens to represent new operators - however, there may be facilities to overload operators in the future.

There are two kinds of tokens that represent operators: Symbolic tokens and keywords (for keyword operators, see [words](words.md)).

## Symbolic tokens

These tokens consist of one or more symbol characters. In particular, such a token contains no characters that are valid in identifiers, no quote characters, and no whitespace.

### Details

Symbolic tokens are lexed using a "max munch" rule: at each lexing step, the longest symbolic token defined by the language specification that appears starting at the current input position is lexed, if any.

Not all uses of symbolic tokens within the Carbon grammar will be as operators. For example, we will have `(` and `)` tokens that serve to delimit various grammar productions, and we may not want to consider `.` to be an operator, because its right "operand" is not an expression.

When a symbolic token is used as an operator, we use the presence or absence of whitespace around the symbolic token to determine its fixity, in the same way we expect a human reader to recognize them. For example, we want `a* - 4` to treat the `*` as a unary operator and the `-` as a binary operator, while `a * -4` results in the reverse. This largely requires whitespace on only one side of a unary operator and on both sides of a binary operator. However, we'd also like to support binary operators where a lack of whitespace reflects precedence such as `2*x*x + 3*x + 1` where doing so is straightforward. The rules we use to achieve this are:

-   There can be no whitespace between a unary operator and its operand.
-   The whitespace around a binary operator must be consistent: either there is
    whitespace on both sides or on neither side.
-   If there is whitespace on neither side of a binary operator, the token before the operator must be an identifier, a literal, or any kind of closing bracket (for example, `)`, `]`, or `}`), and the token after the operator must be an identifier, a literal, or any kind of opening bracket (for example, `(`, `[`, or `{`).

_This list should be extended by proposals that use additional symbolic tokens._

### Token list
The following symbolic tokens are recognized:

|     |      |      |     |     |     |
| --- | ---- | ---- | --- | --- | --- |
| `(` | `)`  | `{`  | `}` | `[` | `]` |
| `,` | `.`  | `;`  | `:` | `*` | `&` |
| `=` | `->` | `=>` |     |     |     |

## Alternatives considered

-   [Lex the longest sequence of symbolic characters.](/proposals/p0601.md#alternatives-considered)
-   [Support an extensible operator set.](/proposals/p0601.md#alternatives-considered)
-   [Apply different or no whitespace restrictions.](/proposals/p0601.md#alternatives-considered)
-   [Require whitespace around a binary operator followed by \[ or \{.](/proposals/p0601.md#alternatives-considered)

## References

-   Proposal
    [#601: Operator tokens](https://github.com/carbon-language/carbon-lang/pull/601)
