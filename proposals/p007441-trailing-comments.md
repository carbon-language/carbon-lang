# Trailing comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7441)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [The experimental restriction](#the-experimental-restriction)
    -   [The lexer](#the-lexer)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Lexical rule](#lexical-rule)
    -   [Examples](#examples)
    -   [Style guidance](#style-guidance)
    -   [Tooling](#tooling)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep requiring comments to be alone on their line](#keep-requiring-comments-to-be-alone-on-their-line)
    -   [Also allow intra-line and block comments](#also-allow-intra-line-and-block-comments)
    -   [Restrict trailing comments to specific positions](#restrict-trailing-comments-to-specific-positions)
    -   [Use a directionality marker](#use-a-directionality-marker)
    -   [Express annotations only through the grammar](#express-annotations-only-through-the-grammar)

<!-- tocstop -->

## Abstract

Carbon currently requires a comment to be the only non-whitespace on its line. A
`//` comment that follows other content on a line, called a _trailing comment_,
is a lexer error. This proposal removes that restriction, allowing a comment to
follow other content on a line. Everything else about comments is unchanged: a
comment still begins with `//`, still requires whitespace after the `//`, and
still runs to the end of the line. Carbon continues to provide only line
comments; no block or intra-line comments are added.

Three observations motivate the change. First, trailing comments are well suited
to short _annotations_ attached to a specific entity or value on a line. Second,
the lexer design now makes it trivial to lex trailing comments, and in fact
requires extra logic and potentially cost to reject them. Third, C++ code
routinely uses trailing comments, so allowing them lets Carbon carry the layout
of migrated code over directly, rather than reworking each comment to read well
in a different structure.

## Problem

The [original comments proposal](/proposals/p000198-comments.md) provided a
single kind of comment, `//` to end of line, and restricted it so that "no code
is permitted prior to a comment on the same line." That proposal explicitly
labeled the restriction **experimental** and said the decision "should be
revisited if we find there is a need for such comments in the context of the
complete language design."

Several considerations now motivate revisiting it:

-   The need for short, free-form _annotations_ that attach directly to a line.
    These are especially useful in examples, presentations, and when discussing
    code, even though they may rarely come up in real-world code bases.
-   In a reversal from the original lexer, it is now requiring _extra_ lexing
    complexity and potentially performance to reject trailing comments.
-   Migrating C++ code, a core Carbon goal, routinely encounters trailing
    comments. The restriction forces each one to be reworked into a layout that
    reads well without it, rather than carrying the existing layout over
    directly.

## Background

### The experimental restriction

The [comments proposal](/proposals/p000198-comments.md) made several decisions
that this proposal leaves untouched:

-   There is one kind of comment, introduced by `//` and running to the end of
    the line; there is no physical line continuation.
-   The `//` must be followed by whitespace (or end of file). Sequences where
    `//` is _not_ followed by whitespace are
    [reserved for future extension](/proposals/p000198-comments.md#reserved-comments)
    such as documentation or fold markers.
-   There are no block comments; a region is commented out by prefixing each
    line.

The one decision this proposal changes is the
[experimental](/proposals/p000198-comments.md#intra-line-comments) requirement
that a comment be the only non-whitespace on its line. The original proposal's
rationale for that requirement was primarily about _tools_: an intra-line or
trailing comment makes it harder for a formatter to know what program element
the comment "attaches to," and reflowing aligned trailing comments across edits
is genuinely complex. It grouped trailing comments together with intra-line
(C-style `/* ... */`) comments and declined both, pending experience.

This proposal separates the two. It allows trailing line comments, which run to
the end of the line and so have an unambiguous extent, while continuing to
exclude intra-line and block comments, which are where most of the original
tooling complexity lives.

### The lexer

Carbon's lexer is a table-dispatch loop: the first byte of each lexical element
selects a handler, and `/` is wired to a handler that decides between a `/`
operator and a `//` comment with a max-munch rule. That dispatch does not depend
on _where_ in the line the `/` occurs. A `//` reaching the comment handler in
the middle of a line takes the same path as one at the start of a line.

A comment, once recognized, is consumed by advancing to the end of the line,
which the lexer already tracks. The only work the lexer does today that is
specific to trailing comments is the explicit check that the `//` begins at the
line's first non-whitespace position, which exists solely to _reject_ trailing
comments. Removing that rejection and recording the comment instead is less
code, not more, and adds no new scanning. In this sense, supporting trailing
comments is nearly free in the lexer, while continuing to reject them is what
now costs extra logic in the comment lexer's hot path.

## Proposal

Allow a `//` comment to follow other content on a line. Such a comment is called
a _trailing comment_. It is lexically identical to a full-line comment: it
begins with `//`, requires whitespace after the `//`, and runs to the end of the
line. The sole change is that a comment is no longer required to be the only
non-whitespace on its line.

Carbon still provides only line comments. This proposal does not add intra-line
or block comments, and does not change the reserved status of `//` sequences
that are not followed by whitespace.

## Details

### Lexical rule

A comment is a lexical element beginning with `//` and running to the end of the
line. A comment may appear either as the entire content of a line (after
optional leading whitespace) or following other content on the line. In both
cases the character after `//` must be whitespace (newline and end of file count
as whitespace), and the comment is removed prior to formation of tokens; it
produces no token.

The `//@...` tooling directives (such as `//@dump-sem-ir-begin`) remain
recognized only as full-line comments, since they are line-oriented markers; in
trailing position `//@` is simply a `//` not followed by whitespace, and so is a
reserved (currently invalid) comment as before.

A block string literal's introducer line, consisting of the `'''` and an optional
file type indicator, may also carry a trailing comment. Since the file type
indicator runs to the end of the line, a trailing comment there is treated like
trailing whitespace: it ends the file type indicator, is not itself recorded as a
comment, and may contain `#` and `"` even though the indicator may not.

### Examples

A trailing comment may follow any content, and the following code is now valid
rather than an error:

```carbon
var count: i32 = 0;          // a) A local variable,

fn Render(frame: Frame) {
  Draw(frame);               // b) A function call,
  Flush();
}                            // c) And a closing brace.
```

Full-line and trailing comments coexist; a full-line comment still introduces
the code below it, while a trailing comment annotates the content on its own
line:

```carbon
// Compute the smallest prime factor.
var factor: i32 = SmallestFactor(n); // TODO: i32 -> i64
```

The reserved-comment rule is unchanged, so a trailing `//` that is not followed
by whitespace is still an error:

```carbon
var x: i32 = 0; //rejected: whitespace is required after `//`
```

A trailing comment may also follow the file type indicator on a block string
literal's introducer line:

```carbon
var query: String = '''sql // TODO: switch to a prepared statement
  SELECT * FROM t
  ''';
```

### Style guidance

Trailing comments are an addition to full-line comments, not a replacement, and
the two suit different purposes:

-   Prefer a full-line comment for documentation. It has less line-length
    pressure and more easily stays attached to the code it describes as that
    code changes.
-   Use a trailing comment only to annotate or mark a specific line, in a
    context where a comment on the preceding line would be awkward, verbose,
    or imprecise.

### Tooling

The lexer records each trailing comment as a comment, marked as trailing so that
tools can distinguish it from a comment that introduces the following code. A
trailing comment is never coalesced with the full-line comments adjacent to it,
even when they line up, because it belongs to the content on its own line.

`carbon format` keeps a trailing comment on the line it annotates, separated
from the preceding content by a single space, rather than relocating it to its
own line. This is the behavior an author intends, and it lets code migrated from
C++ keep the comment layout it already had.

Reflowing trailing comments under more aggressive reformatting, for example
maintaining a column of aligned trailing comments or moving an over-long
trailing comment to its own line, is a formatting-quality concern rather than a
language one, and can be improved over time without further language changes.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Line-specific annotations are often needed when explaining or discussing
    code in order to understand it.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    This is the planned revisit of an explicitly experimental restriction, now
    that the surrounding design has matured. The change is strictly relaxing: it
    only makes previously-invalid programs valid, so it imposes no migration on
    existing code.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    and
    [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    Supporting trailing comments costs the lexer essentially nothing; continuing
    to reject them is what would cost more, so the feature does not compromise
    performance or simplicity.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    Trailing line comments are ubiquitous in C++ and are what programmers
    expect. Allowing them lets migrated C++ code keep its existing layout
    instead of reworking each trailing comment to read well in a different
    structure, which avoids gratuitous churn and is consistent with the original
    proposal's choice of `//` to match C++.

## Alternatives considered

### Keep requiring comments to be alone on their line

We could leave the experimental restriction in place and continue to require
every comment to be the only non-whitespace on its line.

-   Advantages: it is the simplest possible rule; it sidesteps the formatter
    questions about what a trailing comment attaches to and how to reflow
    aligned trailing comments; and it gently pushes authors to promote per-line
    notes into the grammar (such as named arguments) where tools can see them.
-   Disadvantages: it leaves the _annotation_ use case unserved. A short note
    that belongs beside a specific value or branch must move to its own line or
    be omitted, which is awkward when teaching, presenting, or discussing code.
    It also forces C++ code that uses trailing comments to be restructured on
    migration rather than carried over as-is, and it keeps the lexer doing extra
    work to reject a construct it can otherwise handle for free.
-   Core of the decision: the original proposal deferred this precisely so it
    could be revisited with experience. That experience is that trailing
    comments matter for explaining and discussing code and are pervasive in the
    C++ code Carbon aims to migrate, while the lexer cost that motivated the
    restriction has reversed: rejecting trailing comments now takes more work
    than accepting them. Connecting to
    [code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
    the value for annotations and migration outweighs the simplicity of the
    stricter rule.

### Also allow intra-line and block comments

We could go further and also allow comments that attach to something smaller
than a line, such as C-style intra-line comments like `f(/*size=*/5)` or block
comments that span or interrupt lines, as the
[original proposal discussed](/proposals/p000198-comments.md#block-comments-2).

-   Advantages: this would additionally cover the syntactic-disambiguation use
    case (annotating an argument inline) and would make commenting out a region
    or a fragment of a line more ergonomic.
-   Disadvantages: intra-line comments are where most of the formatting
    difficulty lives, since a tool must understand which token a `/*...*/`
    attaches to and how to wrap it, and `/*...*/` with Carbon-specific semantics
    risks confusing C++ readers, as the original proposal noted. Carbon also
    intends to address inline argument annotation through the grammar (for
    example, named arguments) so that such utterances are meaningful to tools
    rather than being text.
-   Core of the decision: trailing line comments capture most of the value, the
    short annotations on a line, while keeping the lexical model simple (still
    only `//` to end of line) and the comment's extent unambiguous. The
    additional intra-line and block forms carry the costs the original proposal
    identified without a corresponding need, so they remain out of scope and the
    syntactic-disambiguation use case continues to be a matter for the grammar.

### Restrict trailing comments to specific positions

We could allow trailing comments only after particular constructs, for example
only after a `;`, an enumerator, or a struct field, to bound the formatter's
problem to a small set of well-understood layouts.

-   Advantages: it would limit where aligned-comment reflow can arise and would
    let a formatter special-case each permitted position.
-   Disadvantages: it makes the lexical and grammatical rules markedly more
    complex and harder to remember, introduces surprising "why not here?" edges,
    and requires the grammar to enumerate the allowed positions and keep that
    list current as the language grows.
-   Core of the decision: a uniform "a comment may follow any content on its
    line" rule is simpler to specify, learn, and implement, and the formatter
    handles the general case directly. A positional restriction trades a real
    and pervasive cost in simplicity for a speculative tooling convenience,
    which is a poor fit for Carbon's preference for one obvious way to do
    things.

### Use a directionality marker

We could permit comments that attach intra-line but require a marker indicating
direction, such as the `//>` / `//<` forms
[sketched in the original proposal](/proposals/p000198-comments.md#intra-line-comments),
so a tool can tell what the comment attaches to.

-   Advantages: it gives tools an explicit attachment hint without full
    intra-line comment parsing.
-   Disadvantages: it invents novel syntax for a niche case, is unfamiliar to
    C++ developers, and still leaves the hard problem of line-wrapping such
    comments largely unsolved.
-   Core of the decision: this was already set aside in the original proposal
    and the same reasoning holds. End-of-line trailing comments need no
    attachment marker because their extent and their subject, the content on
    their line, are clear, so the marker adds complexity without solving a
    problem this proposal has.

### Express annotations only through the grammar

We could decline trailing comments and instead require every per-line annotation
to be promoted into the language, for example as a named argument or a future
attribute or documentation syntax, so the information is structured and visible
to tools.

-   Advantages: structured annotations are meaningful to tools and can be
    validated, and this keeps a single, uniform comment placement.
-   Disadvantages: many annotations are free-form prose ("RFC 2324", "linear if
    1.0", "pixels") with no structured meaning to capture, and forcing them into
    the grammar is heavyweight and often impossible. A declaration-level
    documentation facility, were one added, would target declarations rather
    than arbitrary values on a line, so it would not cover the annotation case
    either. None of these options help carry trailing comments over from
    migrated C++.
-   Core of the decision: the grammar is the right tool when an annotation has
    structure worth capturing, and such cases can use it as the language grows.
    Free-form notes beside a line of code, including those already present in
    migrated C++, are what a comment is for, and trailing comments let authors
    keep them where they belong.
