# Words

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Keywords](#keywords)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _word_ is a lexical element formed from a sequence of letters or letter-like
characters, such as `fn` or `Foo` or `Int`.

The exact lexical form of words has not yet been settled. However, Carbon will
follow lexical conventions for identifiers based on
[Unicode Annex #31](https://unicode.org/reports/tr31/). TODO: Update this once
the precise rules are decided; see the
[Unicode source files](/proposals/p0142.md#characters-in-identifiers) proposal.

## Keywords

The following words are interpreted as keywords:

-   `abstract`
-   `addr`
-   `alias`
-   `and`
-   `api`
-   `auto`
-   `base`
-   `break`
-   `case`
-   `class`
-   `continue`
-   `default`
-   `else`
-   `extends`
-   `external`
-   `fn`
-   `for`
-   `friend`
-   `if`
-   `impl`
-   `import`
-   `interface`
-   `let`
-   `library`
-   `match`
-   `namespace`
-   `not`
-   `or`
-   `override`
-   `package`
-   `partial`
-   `private`
-   `protected`
-   `return`
-   `returned`
-   `then`
-   `var`
-   `virtual`
-   `where`
-   `while`

## Alternatives considered

-   [Character encoding: We could restrict words to ASCII.](/proposals/p0142.md#character-encoding-1)

## References

-   Proposal
    [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
