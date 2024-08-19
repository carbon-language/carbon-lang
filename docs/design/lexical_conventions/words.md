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
    -   [Raw identifiers](#raw-identifiers)
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
[Unicode source files](/proposals/p0142.md#characters-in-identifiers-and-whitespace)
proposal.

Carbon source files, including comments and string literals, are required to be
in Unicode Normalization Form C (NFC).

## Keywords

<!--
Keep in sync:
- utils/textmate/Syntaxes/Carbon.plist
- utils/treesitter/queries/highlights.scm
-->

The following words are interpreted as keywords:

-   `abstract`
-   `adapt`
-   `addr`
-   `alias`
-   `and`
-   `as`
-   `auto`
-   `base`
-   `break`
-   `case`
-   `choice`
-   `class`
-   `constraint`
-   `continue`
-   `default`
-   `destructor`
-   `else`
-   `export`
-   `extend`
-   `final`
-   `fn`
-   `for`
-   `forall`
-   `friend`
-   `if`
-   `impl`
-   `impls`
-   `import`
-   `in`
-   `interface`
-   `let`
-   `library`
-   `like`
-   `match`
-   `namespace`
-   `not`
-   `observe`
-   `or`
-   `override`
-   `package`
-   `partial`
-   `private`
-   `protected`
-   `require`
-   `return`
-   `returned`
-   `Self`
-   `template`
-   `then`
-   `type`
-   `var`
-   `virtual`
-   `where`
-   `while`

### Raw identifiers

A raw identifier looks like `r#<identifier>`. This can be used for identifiers
which have the same spelling as keywords; for example, `r#impl`. It can help
when using C++ code with identifiers that are keywords in Carbon.

The identifier doesn't need to be a keyword, in order to support forwards
compatibility when a keyword is planned to be added. When `<identifier>` is not
a keyword, it will refer to the same entity as `r#<identifier>`.

## Alternatives considered

-   [Character encoding: We could restrict words to ASCII.](/proposals/p0142.md#character-encoding-1)
-   [Normalization form alternatives considered](/proposals/p0142.md#normalization-forms)
-   [Other raw identifier syntaxes](/proposals/p3797.md#other-raw-identifier-syntaxes)
-   [Restrict raw identifier syntax to current and future keywords](/proposals/p3797.md#restrict-raw-identifier-syntax-to-current-and-future-keywords)
-   [Don't require syntax for references to raw identifiers](/proposals/p3797.md#dont-require-syntax-for-references-to-raw-identifiers)
-   [Don't provide raw identifier syntax](/proposals/p3797.md#dont-provide-raw-identifier-syntax)

## References

-   Proposal
    [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
-   Proposal
    [#3797: Raw identifier syntax](https://github.com/carbon-language/carbon-lang/pull/3797)
