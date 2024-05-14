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

### Symbolic token list

The following is the initial list of symbolic tokens recognized in a Carbon
source file:

| Symbolic Tokens | Explanation                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| `+`             | Addition                                                                                                     |
| `-`             | Subtraction and negation                                                                                     |
| `*`             | Indirection, multiplication, and forming pointer types                                                       |
| `/`             | Division                                                                                                     |
| `%`             | Modulus                                                                                                      |
| `^`             | Complementing and Bitwise XOR                                                                                |
| `&`             | Address-of and Bitwise AND                                                                                   |
| `\|`            | Bitwise OR                                                                                                   |
| `<<`            | Arithmetic and Logical Left-shift                                                                            |
| `>>`            | Arithmetic and Logical Right-shift                                                                           |
| `=`             | Assignment and initialization                                                                                |
| `++`            | Increment                                                                                                    |
| `--`            | Decrement                                                                                                    |
| `+=`            | Add-and-assign                                                                                               |
| `-=`            | Subtract-and-assign                                                                                          |
| `*=`            | Multiply-and-assign                                                                                          |
| `/=`            | Divide-and-assign                                                                                            |
| `%=`            | Modulus-and-assign                                                                                           |
| `&=`            | Bitwise-AND-and-assign                                                                                       |
| `\|=`           | Bitwise-OR-and-assign                                                                                        |
| `^=`            | Bitwise-XOR-and-assign                                                                                       |
| `<<=`           | Left-shift-and-assign                                                                                        |
| `>>=`           | Right-shift-and-assign                                                                                       |
| `==`            | Equality or equal to                                                                                         |
| `!=`            | Inequality or not equal to                                                                                   |
| `>`             | Greater than                                                                                                 |
| `>=`            | Greater than or equal to                                                                                     |
| `<`             | Less than                                                                                                    |
| `<=`            | Less than or equal to                                                                                        |
| `->`            | Return type and indirect member access                                                                       |
| `=>`            | Match syntax                                                                                                 |
| `[` and `]`     | Subscript and deduced parameter lists                                                                        |
| `(` and `)`     | Function call, function declaration, and tuple literals                                                      |
| `{` and `}`     | Struct literals, blocks of control flow statements, and the bodies of definitions (classes, functions, etc.) |
| `,`             | Separate tuple and struct elements                                                                           |
| `.`             | Member access                                                                                                |
| `:`             | Name binding patterns                                                                                        |
| `:!`            | Compile-time binding patterns                                                                                |
| `;`             | Statement separator                                                                                          |

## Alternatives considered

[Alternatives from proposal #601](/proposals/p0601.md#alternatives-considered):

-   lex the longest sequence of symbolic characters rather than lexing only the
    longest known operator
-   support an extensible operator set
-   different whitespace restrictions or no whitespace restrictions

## References

-   Proposal
    [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
-   Proposal
    [#339: Add `var <type> <identifier> [ = <value> ];` syntax for variables](https://github.com/carbon-language/carbon-lang/pull/339)
-   Proposal
    [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
-   Proposal
    [#561: Basic classes: use cases, struct literals, struct types, and future work](https://github.com/carbon-language/carbon-lang/pull/561)
-   Proposal
    [#601: Operator tokens](https://github.com/carbon-language/carbon-lang/pull/601)
-   Proposal
    [#676: `:!` generic syntax](https://github.com/carbon-language/carbon-lang/pull/676)
-   Proposal
    [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
-   Proposal
    [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   Proposal
    [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)
-   Proposal
    [#1191: Bitwise operators](https://github.com/carbon-language/carbon-lang/pull/1191)
-   Proposal
    [#2188: Pattern matching syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2188)
-   Proposal
    [#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
-   Proposal
    [#2511: Assignment statements](https://github.com/carbon-language/carbon-lang/pull/2511)
-   Proposal
    [#2665: Semicolons terminate statements](https://github.com/carbon-language/carbon-lang/pull/2665)
