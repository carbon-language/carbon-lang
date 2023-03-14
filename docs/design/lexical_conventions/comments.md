# Comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A comment is a lexical element beginning with the characters `//` and running to
the end of the line. We have no mechanism for physical line continuation, so a
trailing `\` does not extend a comment to subsequent lines.

## Details

In the comments after the `//` a whitespace character is required to make the
comment valid. Newline is a whitespace character, so a line containing only `//`
is a valid comment. The end of the file also constitutes whitespace.

All comments are removed prior to formation of tokens.

Example:

```
// This is a comment and is ignored. \
This is not a comment.

var Int: x; // error, trailing comments not allowed
```

Currently no support for block comments is provided. Commenting out larger
regions of human-readable text or code is accomplished by commenting out every
line in the region.

## Alternatives considered

-   [Intra-line comments](/proposals/p0198.md#intra-line-comments)
-   [Multi-line text comments](/proposals/p0198.md#multi-line-text-comments)
-   [Block comments](/proposals/p0198.md#block-comments-2)
-   [Documentation comments](/proposals/p0198.md#documentation-comments)
-   [Code folding comments](/proposals/p0198.md#code-folding-comments)

## References

-   Proposal
    [#198: Comments](https://github.com/carbon-language/carbon-lang/pull/198)
