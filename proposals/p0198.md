# Comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/198)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
    -   [Use cases](#use-cases)
-   [Background](#background)
    -   [Line comments](#line-comments)
    -   [Block comments](#block-comments)
    -   [`#if 0`](#if-0)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Overview](#overview)
    -   [Block comments](#block-comments-1)
        -   [Block comments rationale](#block-comments-rationale)
    -   [Reserved comments](#reserved-comments)
        -   [Reserved comments rationale](#reserved-comments-rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Intra-line comments](#intra-line-comments)
    -   [Multi-line text comments](#multi-line-text-comments)
    -   [Block comments](#block-comments-2)
    -   [Documentation comments](#documentation-comments)
    -   [Code folding comments](#code-folding-comments)
-   [Rationale](#rationale)

<!-- tocstop -->

## Problem

This proposal provides a suggested concrete lexical syntax for comments.

### Use cases

Comments serve a variety of purposes in existing programming languages. The
primary use cases are:

-   _Documentation_: human-readable commentary explaining to users and future
    maintainers of an API what function it performs and how to use it. Such
    comments are typically attached to function declarations, class definitions,
    public member declarations, at file scope, and similar levels of granularity
    in an API.

    ```
    /// A container for a collection of connected widgets.
    class WidgetAssembly {
      /// Improve the appearance of the assembly if possible.
      void decorate(bool repaint_all = false);

      // ...
    };
    ```

-   _Implementation comments_: human-readable commentary explaining intent and
    mechanism to future readers or maintainers of code, or summarizing the
    behavior of code to avoid readers or maintainers needing to read it in
    detail. Such comments are typically used when such details may not be
    readily apparent from the code itself or may require non-trivial work to
    infer, and tend to be short.

    ```
    void WidgetAssembly::decorate(bool repaint_all) {
      // ...

      // Paint all the widgets that have been changed since last time.
      for (auto &w : widgets) {
        if (repaint_all || w.modified > last_foo)
          w.paint();
      }
      last_decorate = now();

      // ...
    }
    ```

-   _Syntactic disambiguation comments_: comments that contain code or
    pseudocode intended to allow the human reader to more easily parse the code
    in the same way that the compiler does.

    ```
    void WidgetAssembly::decorate(bool repaint_all /*= false*/) {
    // ...

    /*static*/ std::unique_ptr<WidgetAssembly> WidgetAssembly::make() {
    // ...

    assembly.decorate(/*repaint_all=*/true);
    // ...

    }  // end namespace WidgetLibrary
    ```

-   _Disabled code_: comments that contain regions of code that have been
    disabled, because the code is incomplete or incorrect, or in order to
    isolate a problem while debugging, or as reference material for a change in
    progress. It is often considered bad practice to check such comments into
    version control.

## Background

In C++, there are three different ways in which comments are expressed in
practice:

### Line comments

Single-line comments (and sometimes multiline comments) are expressed in C++
using `// ...`:

```
// The next line declares a variable.
int n; // This is a comment about 'n'.
```

(These are sometimes called "BCPL comments".)

-   Can appear anywhere (at the start of a line or after tokens).
-   Can contain any text (other than newline).
-   End at the end of the logical line.
-   Can be continued by ending the comment with `\` (or `??/` in C++14 and
    earlier).
-   Unambiguous with non-comment syntax.
-   "Nest" in that `//` within `//` has no effect.
-   Do not nest with other kinds of comment.

This comment syntax is often used to express documentation (sometimes with a
Doxygen-style `///` introducer) and implementation comments.

### Block comments

Comments within lines (or sometimes multiline comments) are expressed in C++
using `/*...*/`:

```
f(/*size*/5, /*initial value*/1);
```

-   Can appear anywhere (at the start of a line or after tokens).
-   Can contain any text (other than `*/`).
-   End at the `*/` delimiter (which might be separated by a `\` line
    continuation).
-   Ambiguous with non-comment syntax: `int a=1, *b=&a, c=a/*b;` though this is
    not a problem in practice.
-   Do not nest -- the first `*/` ends the comment.

This comment syntax is often used to express syntactic disambiguation comments,
and is sometimes used for disabled code. Some coding styles also use this
comment style for longer documentation comments (sometimes with a Doxygen-style
`/**` introducer).

### `#if 0`

Blocks of code are often commented out in C++ programs using `#if 0`:

```
#if 0
int n;
#endif
```

-   Can appear only at the start of a logical line.
-   Can only contain sequences of preprocessing tokens (including invalid tokens
    such as `'`, but not including unterminated multiline string literals).
-   End at the matching `#endif` delimiter.
-   Unambiguous with any other syntax.
-   Nest properly, and can have other kinds of comments nested within.

This syntax is generally only used for disabled code.

## Proposal

We provide only one kind of comment, which starts with `//` and runs to the end
of the line. No code is permitted prior to a comment on the same line, and the
`//` introducing the comment is required to be followed by whitespace.

This comment syntax is intended to support implementation comments and
(_experimentally_) disabled code. The documentation use case is
[not covered](#documentation-comments), with the intent that a separate
(non-comment) facility is explored for this use case. The syntactic
disambiguation use case is not covered, with the intent that the language syntax
is designed in a way that avoids this use case.

## Details

### Overview

A _comment_ is a lexical element beginning with the characters `//` and running
to the end of the line. We have no mechanism for physical line continuation, so
a trailing `\` does not extend a comment to subsequent lines.

> _Experimental:_ There can be no text other than horizontal whitespace before
> the `//` characters introducing a comment. Either all of a line is a comment,
> or none of it.

The character after the `//` is required to be a whitespace character. Newline
is a whitespace character, so a line containing only `//` is a valid comment.
The end of the file also constitutes whitespace.

All comments are removed prior to formation of tokens.

Example:

```carbon
// This is a comment and is ignored. \
This is not a comment.

var Int: x; // error, trailing comments not allowed
```

### Block comments

> _Experimental:_ No support for block comments is provided. Commenting out
> larger regions of human-readable text or code is accomplished by commenting
> out every line in the region.

#### Block comments rationale

There is little value in supporting block comments for the implementation
comments use case. We expect such comments to typically be short, and in
existing C++ codebases with long implementation comments, it is typical for line
comments rather than block comments to be used. Therefore, as we consider the
documentation use case to be out of scope, and intend for the syntactic
disambiguation use case to be solved by language syntax, the sole purpose of
block comments would be for disabled code. Block comments could provide more
ergonomic support for intra-line disabled code and multiline blocks of disabled
code.

Existing block comment syntaxes are not a great fit for the use case of
disabling code. The `/* ... */` block comment syntax does not nest in C++, and
cannot be used to reliably comment out a block of code because it can be
terminated by a `*/` appearing in a `//` comment or in a string literal. The
`#if 0 ... #endif` syntax would not be a good fit in Carbon as we do not intend
to have a preprocessor in general, and requires the text in between to consist
of a mostly-valid token sequence, disallowing certain forms of incomplete code.

We should be reluctant to invent something new: it is hard to justify the cost
of introducing a novel syntax for the transient and rare use case of disabling
code. And similarly, we should be reluctant to use existing syntax with novel
semantics, such as a `/* ... */` comment that tokenizes its contents, to avoid
surprise to C++ developers.

The disabled code use cases can be addressed with line comments, by commenting
out each line in the intended region, and reflowing or duplicating lines when
disabling code within a line. That may be cumbersome, but it's unclear whether
that burden is sufficient to warrant introducing another form of comment into
the language. By providing no such form of comments, we aim to discover if the
resulting friction warrants a language addition.

### Reserved comments

Comments in which the `//` characters are not followed by whitespace are
reserved for future extension. Anticipated possible extensions are block
comments, documentation comments, and code folding region markers.

#### Reserved comments rationale

We anticipate the possibility of adding additional kinds of comment in the
future. Reserving syntactic space in comment syntax, in a way that is easy for
programs to avoid, allows us to add such additional kinds of comment as a
non-breaking change.

## Alternatives considered

### Intra-line comments

We could include a feature similar to C-style block comments, as a way to
provide comments that attach to some element of the program smaller than a line.
In C++ code, such comments are frequently used to annotate function parameter
names and similar syntactic disambiguation use cases:

```
render(/*use_world_coords=*/true, /*draw_frame=*/false);
```

We expect these use cases to be addressed by extensions to Carbon's grammar,
such as by adding named parameters or annotation syntax, to allow such
utterances to be expressed as code rather than as comments, so they are
meaningful to both the Carbon programmer and the Carbon language tools.

We could permit trailing comments on a line that contains other content. Such
comments are most frequently used in our sample C++ corpus to describe the
meaning of an entity, label, or close brace on the same line:

```
namespace N {
int n; // number of hats
enum Mode {
  mode1, // first mode
  mode2 // second mode
};
} // end namespace N
```

In all cases but the last, we expect it to be reasonable to move the comment to
before the declaration. The case of the "end namespace" comment is another
instance of the syntactic disambiguation use case, which we expect to be
addressed by grammar changes. In general, we should avoid any syntax that would
need disambiguation comments, either by promoting those comments to the language
grammar or by altering the syntax until the comment is unnecessary, such as by
not providing a delimited scope syntax for describing the contents of large
scopes such as namespaces and packages. For example:

```carbon
// This declares the namespace N but does not open a scope.
namespace N;

// This declares a member of namespace N.
@"Number of hats."
var Int: N.n;

enum N.Mode {
  @"First mode."
  mode1;
  @"Second mode."
  mode2;
}
```

Intra-line comments present a challenge for code formatting tools, which would
need to understand what part of the program syntax the comment "attaches to" in
order properly reflow the comment with the code. This concern is mitigated, but
not fully eliminated, by requiring comments to always be on their own line. We
could restrict text comments to appear in only certain syntactic locations to
fully resolve this concern, but doing so would remove the flexibility to insert
comments in arbitrary places:

```
match (x) {
  case .Foo(1, 2,
            // This might be 3 or 4 depending on the size of the Foo.
            Int: n) => { ... }
}
```

We could allow intra-line comments and still retain some idea of what the
comment syntactically attaches to by using a directionality marker in the
comment:

```
match (x) {
  case .Foo(1, 2, //> either 3 or 4 >// Int: n) => { ... }
  case .Foo(2, Int: n //< either 3 or 4 <//, 5) => { ... }
}
```

Even with an understanding of how comments attach, line wrapping such comments
is a complex challenge. For example, formatting in a situation with aligned
trailing comments across multiple lines requires special handling:

```
var Int: quality = 3;   // The quality of the widget. It should always
                        // be between 1 and 9.
var Int: blueness = 72; // The blueness of the widget, as a percentage.
```

Here, a tool that renames `blueness` to `blue_percent` may need to reflow the
comment following `quality` as well as the comment following `blueness`.
Moreover, if the last line becomes too long, keeping the comment on the same
line as the variable may become untenable, requiring a more substantive
rewriting:

```
// The blueness of the widget, as a percentage.
var Int: blue_percent = Floor(ComputeBluenessRatio() * 100);
```

The decision to not support trailing and intra-line comments is **experimental**
and should be revisited if we find there is a need for such comments in the
context of the complete language design.

### Multi-line text comments

No support is provided for multi-line text comments. Instead, the intent is that
such comments are expressed by prepending each line with the same `// ` comment
marker.

Requiring each line to repeat the comment marker will improve readability, by
removing a source of non-local state, and removes a needless source of stylistic
variability. The resulting style of comment is common in other languages and
well-supported by editors. Even in C and C++ code that uses `/* ... */` to
comment out a block of human-readable text, it is common to include a `*` at the
start of each comment continuation line.

### Block comments

We considered various different options for block comments. Our primary goal was
to permit commenting out a large body of Carbon code, which may or may not be
well-formed (including code that contains a block comment, meaning that such
comments would need to nest). Alternatives considered included:

-   Fully line-oriented block comments, which would remove lines without regard
    for whether they are nested within a string literal, with the novel feature
    of allowing some of the contents of a block string literal to be commented
    out. This alternative has the disadvantage that it would result in
    surprising behavior inside string literals containing Carbon code.
-   Fully lexed block comments, in which a token sequence between the opening
    and closing comment marker is produced and discarded, with the lexing rules
    relaxed somewhat to avoid rejecting ill-formed code. This would be analogous
    to C and C++'s `#if 0` ... `#endif`. This alternative has the disadvantage
    that it would be unable to cope with incomplete code fragments, such as an
    unterminated block string literal. It would also be somewhat inefficient to
    process compared to non-lexing syntaxes, but that's likely to be largely
    irrelevant given that block comments are expected to be transient.
-   A hybrid approach, with `//\{` and `//\}` delimiters that are invalid in
    non-raw string literals, and with an indentation requirement for raw string
    literals only. This alternative has the disadvantage of introducing
    additional complexity into the lexical rules by treating different kinds of
    string literals differently.
-   Use of `/*` and `*/` as comment markers. This alternative has the
    disadvantage that it risks confusion by using similar syntax to C and C++
    but with divergent semantics.

However, given the limited use cases for such comments and a desire to minimize
our inventiveness, we are not pursuing any of these options in this proposal.

### Documentation comments

We could add a distinct comment syntax for documentation comments, perhaps
treating documentation comments as producing real tokens rather than being
stripped out by the lexer. However, during discussion, there was significant
support for using a syntax that does not resemble a comment for representing
documentation. For example, we could introduce an attribute syntax, such as
using `@ <expression>` as a prefix to a declaration to attach attributes. Then a
string literal attribute can be treated as documentation:

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

This use case will be explored by a future proposal.

### Code folding comments

Some code editors are able to "fold" regions of a source file in order to ease
navigation. In some cases, these fold regions can be customized by the use of
comment lines. For example, in VS Code, this is accomplished with comments
containing `#region` and `#endregion`:

```
// #region Functions F and G
fn f() { ... }
fn g() { ... }
// #endregion
```

Supporting such markers as normal text within line comments requires no
additional effort. However, we could consider introducing a specific Carbon
syntax for region comments, in order to encourage a common representation across
code editors. Such support is not covered by this proposal, but could be handled
by a new form of comment.

## Rationale

-   Some comment syntax is necessary to support software evolution, readable and
    understandable code, and many other goals of Carbon.
-   A single, simple, and consistent comment style supports Carbon's goal of
    easy to read and understand code, and fast development tools.
-   The experiment of restricting comments to be the only non-whitespace text on
    a line supports Carbon's goal of software evolution.
-   The careful open lexical space left supports Carbon's goal of language
    evolution.
-   The use of `//` as the primary syntax marking comments supports
    interoperability with C++-trained programmers and codebases by avoiding
    unnecessary and unhelpful churn of comment syntax.
