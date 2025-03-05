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
    -   [Type literals](#type-literals)
    -   [Identifiers](#identifiers)
    -   [Raw identifiers](#raw-identifiers)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _word_ is a lexical element formed from a sequence of letters or letter-like
characters, such as `fn` or `Foo` or `Int`, optionally preceded by `r#`.

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
- utils/textmate/Syntaxes/carbom.tmLanguage.json
- utils/tree_sitter/queries/highlights.scm
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
-   `Core`
-   `case`
-   `choice`
-   `class`
-   `constraint`
-   `continue`
-   `default`
-   `destroy`
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
-   `self`
-   `template`
-   `then`
-   `type`
-   `var`
-   `virtual`
-   `where`
-   `while`

### Type literals

A word starting with `i`, `u`, or `f`, followed by a decimal integer, is a
[_numeric type literal_](/docs/design/expressions/literals.md#numeric-type-literals).

### Identifiers

A word is interpreted as an _identifier_ if it is neither a keyword nor a type
literal.

### Raw identifiers

A _raw identifier_ is a word starting with `r#`. A raw identifier is equivalent
to the word following the `r#` prefix, except that it is always interpreted as
an identifier, even if it would otherwise be a keyword or type literal.

Raw identifiers can be used to specify identifiers which have the same spelling
as keywords; for example, `r#impl`. This can be useful when interoperating with
C++ code that uses identifiers that are keywords in Carbon, and when migrating
between versions of Carbon.

The word doesn't need to be a keyword, in order to support forwards
compatibility when a keyword is planned to be added. If `word` is an identifier,
then `word` and `r#word` have the same meaning.

## Alternatives considered

Overview:

-   [Character encoding: We could restrict words to ASCII.](/proposals/p0142.md#character-encoding-1)
-   [Normalization form alternatives considered](/proposals/p0142.md#normalization-forms)

Type literals:

-   [Use C++ type keywords with LP64 convention](/proposals/p2015.md#c-lp64-convention)
-   [Use full type name with length suffix](/proposals/p2015.md#type-name-with-length-suffix)
-   [Use uppercase for type names](/proposals/p2015.md#uppercase-suffixes)
-   [Support additional bit widths](/proposals/p2015.md#additional-bit-sizes)

Raw identifiers:

-   [Other raw identifier syntaxes](/proposals/p3797.md#other-raw-identifier-syntaxes)
-   [Restrict raw identifier syntax to current and future keywords](/proposals/p3797.md#restrict-raw-identifier-syntax-to-current-and-future-keywords)
-   [Don't require syntax for references to raw identifiers](/proposals/p3797.md#dont-require-syntax-for-references-to-raw-identifiers)
-   [Don't provide raw identifier syntax](/proposals/p3797.md#dont-provide-raw-identifier-syntax)

## References

-   Proposal
    [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
-   Proposal
    [#2015: Numeric type literal syntax](https://github.com/carbon-language/carbon-lang/pull/2015)
-   Proposal
    [#3797: Raw identifier syntax](https://github.com/carbon-language/carbon-lang/pull/3797)
