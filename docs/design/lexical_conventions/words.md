# Words

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Alternatives](#alternatives)

<!-- tocstop -->

## Overview

A _word_ is a lexical element formed from a sequence of letters or letter-like
characters, such as `fn` or `Foo` or `Int`.

The exact lexical form of words has not yet been settled. However, Carbon will
follow lexical conventions for identifiers based on
[Unicode Annex #31](https://unicode.org/reports/tr31/). TODO: Update this once
the precise rules are decided; see the
[Unicode source files](/proposals/p0142.md#characters-in-identifiers) proposal.

## Alternatives

**We could restrict words to ASCII.**

Advantages:

-   Reduced implementation complexity.
-   Avoids all problems relating to normalization, homoglyphs, text
    directionality, and so on.
-   We have no intention of using non-ASCII characters in the language syntax or
    in any library name.
-   Provides assurance that all names in libraries can reliably be typed by all
    developers -- we already require that keywords, and thus all ASCII letters,
    can be typed.

Disadvantages:

-   An overarching goal of the Carbon project is to provide a language that is
    inclusive and welcoming. A language that does not permit names in programs
    to be expressed in the developer's native language will not meet that goal
    for at least some of our developers.
