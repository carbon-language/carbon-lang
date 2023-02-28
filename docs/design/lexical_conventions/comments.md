# Comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Comments](#comments)
    -   [Table of contents](#table-of-contents)
    -   [Overview](#overview)
    -   [Details](#details)
    -   [Alternatives considered](#alternatives-considered)
        -   [Intra-line comments](#intra-line-comments)
        -   [Block comments](#block-comments)
        -   [Documentation comments](#documentation-comments)
        -   [Code folding comments](#code-folding-comments)
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

### Intra-line comments

Intra-line comments are comments that appear within a line of code, rather than
on a separate line. A feature similar to C-style block comments, as a way to
provide comments that attach to some element of the program smaller than a line.

Pros:

-   provide explanation for single line of code.
-   ability to permit trailing comments on a line that contains other content.
-   possibility of future extensions to the grammar to allow for such utterances
    to be expressed as code rather than as comments.

Cons:

-   presents a challenge for code formatting tools to identify the program
    syntax the comment "attaches to".
-   restricting the text comments to appear in only certain syntactic locations
    would remove the flexibility to insert comments in arbitrary places.

### Block comments

They permit commenting out a large body of Carbon code, which may or may not be
well-formed (including code that contains a block comment, meaning that such
comments would need to nest).

Alternatives considered included:

-   Fully line-oriented block comments: It would remove lines without regard for
    whether they are nested within a string literal, also allowing some of the
    contents of a block string literal to be commented out. This alternative has
    the disadvantage that it would result in surprising behavior inside string
    literals containing Carbon code.
-   Fully lexed block comments: Here a token sequence between the opening and
    closing comment marker is produced and discarded, with the lexing rules
    relaxed somewhat to avoid rejecting ill-formed code. This would be analogous
    to C and C++'s `#if 0` ... `#endif`. This alternative has the disadvantage
    that it would be unable to cope with incomplete code fragments, such as an
    unterminated block string literal. Its also inefficient to process compared
    to non-lexing syntaxes.
-   A hybrid approach: Using `//\{` and `//\}` delimiters that are invalid in
    non-raw string literals, and with an indentation requirement for raw string
    literals only. This alternative has the disadvantage of introducing
    additional complexity into the lexical rules by treating different kinds of
    string literals differently.
-   Use of `/*` and `*/` as comment markers. This alternative has the  
    disadvantage that it risks confusion by using similar syntax to C and C++
    but with divergent semantics.

After considerations, it was decided not to provide support to block comments.

### Documentation comments

A distinct comment syntax for documentation comments, there by treating
documentation comments as producing real tokens rather than being stripped out
by the lexer. A significant support for using a syntax that does not resemble a
comment for representing documentation. For example, we could introduce an
attribute syntax, such as using `@ <expression>` as a prefix to a declaration to
attach attributes. Then a string literal attribute can be treated as
documentation:

```carbon
@"Get the size of the thing."
fn GetThingSize() -> Int;
@"""
Rate the quality of the widget.
Returns a quality factor between 0.0 and 1.0.
"""
fn RateQuality(
  @"The widget to rate."
  Widget: w,
  @"A widget quality database."
  QualityDB: db) -> Float;
```

This use case will be explored by a FUTURE PROPOSAL.

### Code folding comments

Its found that some of the code editors can "fold" regions of a source file in
order to ease navigation. In some cases, these fold regions can be customized by
the use of comment lines.

Example: In VS Code, this is accomplished with comments containing `#region` and
`#endregion`:

```
// #region Functions F and G
fn f() { ... }
fn g() { ... }
// #endregion
```

Supporting such markers as normal text within line comments requires no
additional effort, but consider introducing a specific Carbon syntax for region
comments. This would encourage a common representation across code editors. This
would be handled by a new form of comment in the future.

## References

-   Proposal
    [#198 Comments](https://github.com/carbon-language/carbon-lang/pull/198)
