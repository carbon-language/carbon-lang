# Whitespace

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [References](#references)

<!-- tocstop -->

## Overview

The exact lexical form of Carbon whitespace has not yet been settled. However,
Carbon will follow lexical conventions for whitespace based on
[Unicode Annex #31](https://unicode.org/reports/tr31/). TODO: Update this once
the precise rules are decided; see the
[Unicode source files](/proposals/p0142.md#characters-in-identifiers) proposal.

Unicode Annex #31 suggests selecting whitespace characters based on the
characters with Unicode property `Pattern_White_Space`, which is currently these
11 characters:

-   Horizontal whitespace:
    -   U+0009 CHARACTER TABULATION (horizontal tab)
    -   U+0020 SPACE
    -   U+200E LEFT-TO-RIGHT MARK
    -   U+200F RIGHT-TO-LEFT MARK
-   Vertical whitespace:
    -   U+000A LINE FEED (traditional newline)
    -   U+000B LINE TABULATION (vertical tab)
    -   U+000C FORM FEED (page break)
    -   U+000D CARRIAGE RETURN
    -   U+0085 NEXT LINE (Unicode newline)
    -   U+2028 LINE SEPARATOR
    -   U+2029 PARAGRAPH SEPARATOR

The quantity and kind of whitespace separating tokens is ignored except where
otherwise specified.

## References

-   Proposal
    [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
