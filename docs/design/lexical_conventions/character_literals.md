# Character literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Escape Sequences](#escape-sequences)
-   [References](#references)

<!-- tocstop -->

## Overview

A character literal represents a single Unicode code point at compile time.
Character literals are delimited by single quotes (`'`), for example `'a'`.

```carbon
var a: char = 'a';
var newline: char = '\n';
```

## Details

A character literal consists of a sequence of characters enclosed in single
quotes (`'`).

-   The contents must represent precisely one Unicode code point.
-   Hex escape sequences (`\xHH`) are supported but limited to values up to
    `0x7F` (where the UTF-8 code unit and Unicode code point values are
    identical). Values `0x80` and above are disallowed in character literals to
    avoid ambiguity between arbitrary byte values and Unicode code points.
-   Grapheme clusters (sequences of multiple code points representing a single
    visual character) are **not** supported in character literals.

### Escape Sequences

Character literals support the same escape sequences as
[string literals](string_literals.md#escape-sequences):

| Escape        | Meaning                                                             |
| ------------- | ------------------------------------------------------------------- |
| `\t`          | U+0009 CHARACTER TABULATION                                         |
| `\n`          | U+000A LINE FEED                                                    |
| `\r`          | U+000D CARRIAGE RETURN                                              |
| `\"`          | U+0022 QUOTATION MARK (`"`)                                         |
| `\'`          | U+0027 APOSTROPHE (`'`)                                             |
| `\\`          | U+005C REVERSE SOLIDUS (`\`)                                        |
| `\0`          | Code unit with value 0                                              |
| `\xHH`        | Code unit with hexadecimal value HH<sub>16</sub>(limited to ≤ `7F`) |
| `\u{HHHH...}` | Unicode code point U+HHHH...                                        |

Hexadecimal digits (`H`) in `\x` or `\u` escape sequences must use uppercase
letters (for example, `\x0A`, not `\x0a`).

## References

-   Proposal
    [#1964: Character Literals](https://github.com/carbon-language/carbon-lang/pull/1964)
-   Proposal
    [#6710: `char` redesign](https://github.com/carbon-language/carbon-lang/pull/6710)
