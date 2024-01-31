# Lexical conventions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Lexical elements](#lexical-elements)

<!-- tocstop -->

## Lexical elements

The first stage of processing a
[source file](/docs/design/code_and_name_organization/source_files.md) is the
division of the source file into lexical elements.

A _lexical element_ is one of the following:

-   a maximal sequence of [whitespace](whitespace.md) characters
-   a [word](words.md)
-   a literal:

    -   a [numeric literal](numeric_literals.md)
    -   a [string literal](string_literals.md)

-   a [comment](comments.md)
-   a [symbolic token](symbolic_tokens.md)

The sequence of lexical elements is formed by repeatedly removing the longest
initial sequence of characters that forms a valid lexical element, with the
following exception:

-   When a numeric literal immediately follows a `.` or `->` token, with no
    intervening whitespace, a real literal is never formed. Instead, the token
    will end no later than the next `.` character. For example, `tuple.1.2` is
    five tokens, `tuple` `.` `1` `.` `2`, not three tokens, `tuple` `.` `1.2`.
    However, `tuple . 1.2` is lexed as three tokens.
