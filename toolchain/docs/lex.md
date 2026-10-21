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

<!-- tocstop -->

## Overview

Lexing converts input source code into tokenized output. Literals, such as
string literals, have their value parsed and form a single token at this stage.

## Bracket matching

The lexer handles matching for `()`, `[]`, and `{}`. When a bracket lacks a
match, it will insert a "recovery" token to produce a match. As a consequence,
the lexer's output should always have matched brackets, even with invalid code.

Where the missing bracket belongs is decided by a search over candidate repairs,
using indentation, structural facts about the language, and formatting
conventions as evidence. See
[mismatched bracket recovery](lex/mismatched_brackets.md).

## Alternatives considered

-   [Bracket matching in parser](/proposals/p006716-move-toolchain-alternatives-to-proposals.md#bracket-matching-in-parser)
