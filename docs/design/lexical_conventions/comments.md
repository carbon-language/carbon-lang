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
-   [Style](#style)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A comment is a lexical element beginning with the characters `//` and running to
the end of the line. We have no mechanism for physical line continuation, so a
trailing `\` does not extend a comment to subsequent lines.

A comment may be the entire content of a line, or it may follow other content on
the line as a _trailing comment_. A full-line comment typically introduces the
code below it, while a trailing comment annotates the content on its own line.

## Details

After the `//` a whitespace character is required to make the comment valid.
Newline is a whitespace character, so a line containing only `//` is a valid
comment. The end of the file also constitutes whitespace.

All comments are removed prior to formation of tokens; a comment produces no
token.

Example:

```
// This is a comment and is ignored. \
This is not a comment.

var Int: x; // This is a trailing comment annotating `x`.
```

Currently no support for block comments is provided. Commenting out larger
regions of human-readable text or code is accomplished by commenting out every
line in the region.

## Style

Full-line and trailing comments serve different purposes:

-   Prefer a full-line comment for documentation. It has less line-length
    pressure and more easily stays attached to the code it describes as that
    code changes.
-   Use a trailing comment only to annotate or mark a specific line, in a
    context where a comment on the preceding line would be awkward, verbose, or
    imprecise.

## Alternatives considered

-   [Intra-line comments](/proposals/p000198-comments.md#intra-line-comments)
-   [Multi-line text comments](/proposals/p000198-comments.md#multi-line-text-comments)
-   [Block comments](/proposals/p000198-comments.md#block-comments-2)
-   [Documentation comments](/proposals/p000198-comments.md#documentation-comments)
-   [Code folding comments](/proposals/p000198-comments.md#code-folding-comments)
-   [Keep requiring comments to be alone on their line](/proposals/pNNNNNN-trailing-comments.md#keep-requiring-comments-to-be-alone-on-their-line)

## References

-   Proposal
    [#198: Comments](https://github.com/carbon-language/carbon-lang/pull/198)
-   Proposal
    [#NNNN: Trailing comments](https://github.com/carbon-language/carbon-lang/pull/NNNN)
