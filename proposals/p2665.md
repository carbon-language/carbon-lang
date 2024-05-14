# Semicolons terminate statements

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2665)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Discussion in Carbon](#discussion-in-carbon)
    -   [In other languages](#in-other-languages)
        -   [Requiring semicolons](#requiring-semicolons)
        -   [Optional semicolons](#optional-semicolons)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Optional semicolons](#optional-semicolons-1)

<!-- tocstop -->

## Abstract

Statements, declarations, and definitions will terminate with either a semicolon
(`;`) or a close curly brace (`}`). Semicolons are never optional.

For example, with a semicolon, `x = x + 2;` or `class C;`. With a close curly
brace, `for ( ... ) { ... }`, or `class C { ...}`.

This does not affect any approved proposal; rather, it makes an important
assumption explicit.

## Problem

Statements need some system for separation. There are two main options for this:

1. Require semicolons to terminate statements.
2. Automatically determine where statements terminate.
    - Some languages, such as Python, define a syntax where a newline terminates
      statements.
    - Other languages, such as Javascript, require semicolons but define rules
      for semicolon insertion.

Although Carbon's design currently assumes semicolons are required, it hasn't
been directly addressed by a proposal.

## Background

### Discussion in Carbon

This was discussed on leads issue
[#1924: Semicolon](https://github.com/carbon-language/carbon-lang/issues/1924).
Some rationale is provided there, stemming from discussion
[#1739: Semicolon](https://github.com/carbon-language/carbon-lang/discussions/1739).

### In other languages

[This blog](https://pling.jondgoodwin.com/post/semicolon-inference/) provides a
similar survey of multiple languages.

#### Requiring semicolons

In C++, C#, and Java, semicolons are always required.

In Rust, semicolons are generally required, but may be omitted for an
[implicit return](https://doc.rust-lang.org/std/keyword.return.html). Because
[blocks are expressions](https://doc.rust-lang.org/reference/expressions/block-expr.html),
there are
[ambiguities in expression statements](https://doc.rust-lang.org/reference/statements.html#expression-statements)
between parsing as a standalone statement and parsing as part of an expression.

#### Optional semicolons

In Python, a line is a
[simple statement](https://docs.python.org/3/reference/simple_stmts.html), and
parentheses are an idiomatic way to create multi-line statements. Semicolons may
be used to explicitly separate statements. For example:

```python
value = (
  "text"
)
a = 1; b = 2; c = 3
```

Swift allows some statements to wrap lines, although multiple statements on the
same line (`x = 1 x = 1`) require a semicolon. The detailed rules aren't
documented so it's difficult to assess other than that Swift developers are
generally happy with the results. Swift's
[statements section](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/statements)
doesn't define statement boundaries, and the
[grammar](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/summaryofthegrammar/)
documents that line-breaks are treated as whitespace. However, there are
observable ways the behavior can lead to small mistakes; these may may often be
caught by the compiler, but will sometimes be missed. For example:

```swift
// One statement in Swift, but two in Python and Kotlin.
var x = 1
      + 1
// Two statements in Swift because of whitespace sensitivity. Second statement
// is a compiler warning.
var x = 1
      +1
// Two calls, the second on the return value of the first.
Make() ()
// A single call followed by an empty tuple. Second statement is valid.
Make()
()
```

Kotlin permits a newline to be used to terminate statements instead of a
semicolon. Kotlin's grammar
[explicitly enumerates](https://kotlinlang.org/spec/syntax-and-grammar.html) all
the places where newlines can appear (see mentions of `NL` in the grammar), and
doesn't allow newlines in places where they would introduce ambiguity.

```kotlin
// This is unambiguously parsed as two statements, because
// a newline is not permitted before a `+` operator.
var x = 1
+ 1
```

In JavaScript and TypeScript, semicolons are part of the formal syntax, and
ECMAScript provides
[Automatic Semicolon Insertion (ASI)](https://tc39.es/ecma262/#sec-automatic-semicolon-insertion).
Note ECMAScript also documents
[Interesting Cases](https://tc39.es/ecma262/#sec-interesting-cases-of-automatic-semicolon-insertion)
which may lead to confusion for developers.

In Go, semicolons are similarly part of the formal syntax, and
[certain tokens cause a semicolon insertion](https://go.dev/ref/spec#Semicolons).
This is also used to enforce style, for example by requiring the opening `{` of
an `if` body to be on the same line in order to avoid semicolon insertion.

## Proposal

As described in the abstract, Carbon will require semicolons to terminate
statements and forward declarations.

Examples with a semicolon include:

-   Most statements, such as `Foo();` and `x = x + 2;`.
-   `var` statements and declarations, such as `var x: i32 = 0;`
-   Forward declarations, such as `class C;` or `fn Foo();`.

Examples with a close curly brace include:

-   Statement grammars that terminate with a curly brace, such as
    `if ( ... ) { ... }` or `match ( ... ) { ... }`.
-   Declarations that include a definition, such as `class C { ... }` or
    `fn Foo() { ... }`.
    -   This is partly in contrast with C++, which would requires a semicolon in
        `class C { ... };`.

Carbon's current design has been written assuming the above; this is making
requiring semicolons an explicit decision.

## Rationale

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   We expect it to be easier to write tools that parse and operate on
        source code if semicolons are required.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Requiring semicolons leaves open the most evolutionary paths; any
        optional semicolon approach means the design would need to be more
        thoughtful about handling ambiguities.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Semicolons are a
        [visual aid](/docs/project/principles/low_context_sensitivity.md#visual-aids)
        that reinforces statement termination, even though they might be viewed
        as a nuisance to write or visually unnecessary for some developers.
        -   Carbon weighs readability more heavily because of the expectation
            that code will be read more often.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The use of semicolons is expected to improve familiarity for C++
        developers, even for developers who might prefer optional semicolons.

## Alternatives considered

### Optional semicolons

Semicolons could be made optional. This would most likely be with an approach
similar to Python, based mainly on newlines.

Advantages:

-   Languages with optional semicolons are very popular. Python is either the
    most, or the 2nd most, widely used programming language by most measures
    ([1](https://pypl.github.io/PYPL.html)
    [2](https://octoverse.github.com/2022/top-programming-languages)
    [3](https://www.tiobe.com/tiobe-index/)).
-   Echoes the direction of evolution in other languages.
    -   For example, Swift and Kotlin are recently designed languages that make
        semicolons optional in ways that work well for developers in practice.
-   Compile-time validation and errors on no-op statements could be used to
    detect some of the issues that arise with optional semicolons in Python and
    JavaScript.
    -   For example, TypeScript may improve the handling of ASI ambiguities by
        [increasing detectability of mistakes](https://medium.com/@eugenkiss/dont-use-semicolons-in-typescript-474ccfe4bdb3).
-   While optional semicolons seem to get fewer complaints, requiring semicolons
    is likely to lead to ongoing friction due to the overall trend. This can be
    seen for languages like Rust
    ([1](https://github.com/rust-lang/rust/issues/27116)
    [2](https://internals.rust-lang.org/t/make-some-separators-optional/4846)
    [3](https://github.com/rust-lang/rfcs/issues/2583)
    [4](https://users.rust-lang.org/t/why-semicolons/25074)) or C#
    ([1](https://github.com/dotnet/roslyn/issues/5355)
    [2](https://github.com/dotnet/csharplang/discussions/496)
    [3](https://github.com/dotnet/csharplang/discussions/5655)).

Disadvantages:

-   Semicolons are a visual anchor for statement termination when scanning code.
-   Requiring semicolons leaves more evolutionary paths available for Carbon.
    This includes both syntactic changes without introducing ambiguity and
    implicit returns as in Rust.
    -   Although it's not clear Carbon will fully adopt implicit returns,
        similar syntactic choices may arise for lambdas.
-   Semicolons are a signal to the compiler about where statements were intended
    to terminate, and can be used to provide better error detection as a
    consequence.
    -   For contrast, optional semicolons may lead to unintended statements.
        While ASI's problems are
        [documented](https://tc39.es/ecma262/#sec-automatic-semicolon-insertion),
        we expect any optional semicolon approach will lead to some increase in
        bugs that the compiler cannot detect, if only because fewer mistakes are
        necessary in order to produce valid but incorrect code.
-   Making code with no semicolons idiomatic may increase the "strangeness" for
    C++ developers, who are the primary target for Carbon.

Semicolons are expected to be a net benefit, as explained by the
[rationale](#rationale).
