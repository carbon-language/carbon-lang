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
-   TODO: operators, comments, ...

The sequence of lexical elements is formed by repeatedly removing the longest
initial sequence of characters that forms a valid lexical element.
