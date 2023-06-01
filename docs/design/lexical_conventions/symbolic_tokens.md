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
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _symbolic token_ is one of a fixed set of
[tokens](https://en.wikipedia.org/wiki/Lexical_analysis#Token) that consist of
characters that are not valid in identifiers. That is, they are tokens
consisting of symbols, not letters or numbers. Operators are one use of symbolic
tokens, but they are also used in patterns `:`, declarations (`->` to indicate
return type, `,` to separate parameters), statements (`;`, `=`, and so on), and
other places (`,` to separate function call arguments).

Carbon has a fixed set of symbolic tokens, defined by the language
specification. Developers cannot define new symbolic tokens in their own code.

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

These rules enable us to use a token like `*` as a prefix, infix, and postfix
operator, without creating ambiguity.

## Details

Symbolic tokens are intended to be used for widely-recognized operators, such as
the mathematical operators `+`, `*`, `<`, and so on. Those used as operators
would generally be expected to also be meaningful for some user-defined types,
and should be candidates for being made overloadable once we support operator
overloading.

### Symbolic token list

The following is the initial list of symbolic tokens recognized in a Carbon
source file:

| Symbolic Tokens | Explanation                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `+`             | Addition                                                                                                   |
| `-`             | Subtraction and negation                                                                                   |
| `*`             | Indirection, multiplication, and forming pointer types                                                     |
| `/`             | Division                                                                                                   |
| `%`             | Modulus                                                                                                    |
| `=`             | Assignment                                                                                                 |
| `^`             | Complementing and Bitwise XOR                                                                              |
| `&`             | Address-of and Bitwise AND                                                                                 |
| `\|`            | Bitwise OR                                                                                                 |
| `<<`            | Arithmetic and Logical Left-shift                                                                          |
| `>>`            | Arithmetic and Logical Right-shift                                                                         |
| `==`            | Equality or equal to                                                                                       |
| `!=`            | Inequality or not equal to                                                                                 |
| `>`             | Greater than                                                                                               |
| `>=`            | Greater than or equal to                                                                                   |
| `<`             | Less than                                                                                                  |
| `<=`            | Less than or equal to                                                                                      |
| `->`            | Return type and indirect member access                                                                     |
| `=>`            | Match syntax                                                                                               |
| `[` and `]`     | Subscript and deduced parameter lists                                                                      |
| `(` and `)`     | Function call, function declaration and tuple literals                                                     |
| `{` and `}`     | Struct literals, blocks of control flow statements and the bodies of definitions (classes, functions, etc) |
| `,`             | Separate tuple and struct elements                                                                         |
| `.`             | Member access                                                                                              |
| `:`             | Name bindings                                                                                              |
| `:!`            | Generic binding                                                                                            |
| `;`             | Statement separator                                                                                        |

TODO: The assignment operators in
[#2511](https://github.com/carbon-language/carbon-lang/pull/2511) are still to
be added.

## Alternatives considered

-   [Proposal: p0601](/proposals/p0601.md#alternatives-considered)

## References

-   Proposal
    [#601: Symbolic tokens](https://github.com/carbon-language/carbon-lang/pull/601)
