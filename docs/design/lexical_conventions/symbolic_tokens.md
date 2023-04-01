# Symbolic Tokens

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Symbolic token list](#symbolic-token-list)
    -   [Whitespace](#whitespace)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _symbolic token_ is one of a fixed set of
[tokens](https://en.wikipedia.org/wiki/Lexical_analysis#Token) that consist of
characters that are not valid in identifiers, that is they are tokens consisting
of symbols, not letters or numbers. Operators are one use of symbolic tokens,
but they are also used in patterns `:`, declarations (`->` to indicate return
type, `,` to separate parameters), statements (`;`, `=`, and so on), and other
places (`,` to separate function call arguments).

Carbon has a fixed set of tokens that represent operators, defined by the
language specification. Developers cannot define new tokens to represent new
operators.

Symbolic tokens are lexed using a "max munch" rule: at each lexing step, the
longest symbolic token defined by the language specification that appears
starting at the current input position is lexed, if any.

When a symbolic token is used as an operator, the surrounding whitespace must
follow certain rules:

-   There can be no whitespace between a unary operator and its operand.
-   The whitespace around a binary operator must be consistent: either there is
    whitespace on both sides or on neither side.
-   If there is whitespace on neither side of a binary operator, the token
    before the operator must be an identifier, a literal, or any kind of closing
    bracket (for example, `)`, `]`, or `}`), and the token after the operator
    must be an identifier, a literal, or any kind of opening bracket (for
    example, `(`, `[`, or `{`).

## Details

Symbolic tokens are intended to be used for widely-recognized operators, such as
the mathematical operators `+`, `*`, `<`, and so on. Those used as operators
would generally be expected to also be meaningful for some user-defined types,
and should be candidates for being made overloadable once we support operator
overloading.

### Symbolic token list

The following is the initial list of symbolic tokens recognized in a Carbon
source file:

| Token | Explanation                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------- |
| `*`   | Indirection, multiplication, and forming pointers                                                          |
| `&`   | Address-of or Bitwise AND                                                                                  |
| `=`   | Assignment                                                                                                 |
| `->`  | Return type and indirect member access                                                                     |
| `=>`  | Match syntax                                                                                               |
| `[`   | Subscript and used for deduced parameter lists                                                             |
| `]`   | Subscript and used for deduced parameter lists                                                             |
| `(`   | Separate tuple and struct elements                                                                         |
| `)`   | Separate tuple and struct elements                                                                         |
| `{`   | Struct literals, blocks of control flow statements and the bodies of definitions (classes, functions, etc) |
| `}`   | Struct literals, blocks of control flow statements and the bodies of definitions (classes, functions, etc) |
| `,`   | Separate tuple and struct elements                                                                         |
| `.`   | Member access                                                                                              |
| `:`   | Name bindings                                                                                              |
| `;`   | Name bindings                                                                                              |

This list is expected to grow over time as more symbolic tokens are required by
language proposals.

Note: The above list only covers up to
[#601](https://github.com/carbon-language/carbon-lang/pull/601) and more have
been added since that are not reflected here.

### Whitespace

Carbon's rule for whitespace around operators have been designed to allow the
same symbolic token to be used as a prefix operator, infix operator, and postfix
operator in some cases. To make parsing operators unambiguous, we require
whitespace to be present or absent around the operator to indicate its fixity,
with binary operators having whitespace on both sides, and unary operators
lacking whitespace between the operator and its operand. However, there are some
cases where omitting whitespace around a binary operator can aid readability,
such as in expressions like `2*x*x + 3*x + 1`. In such cases, the operator with
whitespace on neither side is treated as binary if the token immediately before
the operator indicates the end of an operand and the token immediately after
indicates the beginning of an operand.

Identifiers, literals, and brackets of any kind, facing away from the operator,
are defined as tokens that indicate the beginning or end of an operand. For
error recovery purposes, no expression context can be preceded by a token that
looks like the end of an operand, and no expression context can be followed by a
token that looks like the start of an operand, except in function definitions
where `{}` is the body of the function.

From the perspective of token formation, there are four variants of each
symbolic token: a binary variant with whitespace on both sides, a binary variant
with whitespace on neither side, a unary variant with whitespace on neither
side, and prefix and postfix variants with whitespace on the left and right
sides, respectively. In non-operator contexts, any variant of a symbolic token
is acceptable, but in operator contexts, only the appropriate variant can be
used.

The whitespace rule was designed to strike a balance between simplicity and
expressiveness for the programmer, and simplicity and good support for error
recovery in the implementation. The rule's allowance for omitting whitespace
around binary operators aids readability, but it can cause errors if not used
carefully, particularly in function definitions. Despite this, the rule provides
the necessary cues for human readers to understand the code, while still
allowing for unambiguous parsing of operators.

## Alternatives considered

-   [Proposal: p0601](/proposals/p0601.md#alternatives-considered)

## References

-   Proposal
    [#601: Symbolic tokens](https://github.com/carbon-language/carbon-lang/pull/601)
