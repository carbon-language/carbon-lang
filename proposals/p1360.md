# Change raw string literal syntax: `[#]\*"` represents single-line string and `[#]\*'''` represents block string

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1360)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [The current implementation](#the-current-implementation)
    -   [Using `"""` to start block string literals](#using--to-start-block-string-literals)
    -   [Non-quote marker after the open quote](#non-quote-marker-after-the-open-quote)
    -   [Use different quotes to allow `#'"'#`](#use-different-quotes-to-allow-)

<!-- tocstop -->

## Problem

Under current design of string literals, users may make assumptions that a
starting `[#]*"""` represents a block string and misunderstand the syntax.

## Background

The design of
[string literals](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/string_literals.md)
specifies a block string literal to start with `[#]*"""`. Users may take for
granted the other way around, where any string literal starting with `[#]*"""`
is a block string literal. This does not hold true, however. Two counter-cases
are `"""abc"""` represents three tokens `""`, `"abc"` and `""`, and `#"""#`
which is equivalent to `"\""`. Neither is a block string literal and may be
visually confusing.

## Proposal

Interpret `[#]*"` as the start of single-line string literals, and `[#]*'''` as
the start of block string literals. Disallow adjacent string literals like
`"""abc"""`.

## Details

Users can easily distinguish single-line string literals from block string
literals with the proposed change. Confusion on `"""abc"""` will be eliminated
because adjacent string literals are invalid in the proposal. On the other hand,
`#"""#` will be clear to the user of representing `"\""`, as `"""` does not
represent a block string literal any more. More details can be found
[here](/docs/design/lexical_conventions/string_literals.md).

## Rationale

This principle helps make Carbon code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
because it avoids confusion on the type of certain string literals.

## Alternatives considered

### The current implementation

In addition to the confusion described above, the
[current implementation](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/string_literals.md)
complicates the lexing. When the lexer sees `[#]+"""`, it temporarily accepts
syntax of both string types because the type of the string is undecided.
Specifically, it accepts vertical whitespaces even if they are not allowed in
single-line strings. The type of the string won't be decided until the lexer
sees a closing `"#` or a new line. In case of a closing `"#` where the string is
single-line, the lexer will look back on the scanned characters for vertical
whitespaces to decide if the single-line string is valid.

### Using `"""` to start block string literals

This approach loses some convenience in using raw string literals while
addressing the problem. For example, as discussed in
[issue #1359](https://github.com/carbon-language/carbon-lang/issues/1359),
`#"""#` is a natural way to write a string of `"`.

### Non-quote marker after the open quote

Although something similar to C++ style like `"(` solves the problem, the syntax
becomes complicated and hurts readability. In addition, `"(")"` is no simpler
than `"\""` or `#"""#`.

### Use different quotes to allow `#'"'#`

When disallowing adjacent string literals, we can additionally allow `[#]+'` on
single-line string literals. Another option is to use `[#]+'` for single-line
string literals and `[#]+"` for block string literals. In general, `#'"'#` is
visually confusing with character `'"'` and hurts readability.
