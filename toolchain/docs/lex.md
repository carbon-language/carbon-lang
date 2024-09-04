# Lex

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Bracket matching](#bracket-matching)
-   [Alternatives considered](#alternatives-considered)
    -   [Bracket matching in parser](#bracket-matching-in-parser)

<!-- tocstop -->

## Overview

Lexing converts input source code into tokenized output. Literals, such as
string literals, have their value parsed and form a single token at this stage.

## Bracket matching

The lexer handles matching for `()`, `[]`, and `{}`. When a bracket lacks a
match, it will insert a "recovery" token to produce a match. As a consequence,
the lexer's output should always have matched brackets, even with invalid code.

While bracket matching could use hints such as contextual clues from
indentation, that is not yet implemented.

## Alternatives considered

### Bracket matching in parser

Bracket matching could have also been implemented in the parser, with some
awareness of parse state. However, that would shift some of the complexity of
recovery in other error situations, such as where the parser searches for the
next comma in a list. That needs to skip over bracketed ranges. We don't think
the trade-offs would yield a net benefit, so any change in this direction would
need to show concrete improvement, for example better diagnostics for common
issues.
