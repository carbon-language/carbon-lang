# Lexical conventions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Lexical elements](#lexical-elements)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support
[the overview](/docs/design/README.md). It should not be treated as accepted by
the core team; rather, it is a placeholder until we have more time to examine
this detail. Please feel welcome to rewrite and update as appropriate.

See [PR 17](https://github.com/carbon-language/carbon-lang/pull/17) for context
-- that proposal may replace this.

## Lexical elements

The first stage of processing a
[source file](/docs/design/code_and_name_organization/source_files.md) is the
division of the source file into lexical elements.

A _lexical element_ is one of the following:

-   a maximal sequence of [whitespace](whitespace.md) characters
-   a [word](words.md)
-   a literal:
    -   a [numeric literal](numeric_literals.md)
    -   TODO: string literals
-   TODO: operators, comments, ...

The sequence of lexical elements is formed by repeatedly removing the longest
initial sequence of characters that forms a valid lexical element.
