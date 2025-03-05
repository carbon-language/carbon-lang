# Destructor syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5017)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Not directly callable](#not-directly-callable)
-   [Future work](#future-work)
    -   [Extend syntax to allow explicit marking of _trivial_ destructors](#extend-syntax-to-allow-explicit-marking-of-trivial-destructors)
    -   [Decide whether to desugar destructors to interfaces](#decide-whether-to-desugar-destructors-to-interfaces)
    -   [Copy and move functions](#copy-and-move-functions)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Destructor syntax options](#destructor-syntax-options)
    -   [Destructor name options](#destructor-name-options)

<!-- tocstop -->

## Abstract

Fix destructor syntax ambiguity by switching to `fn destroy` mirroring standard
function syntax. This is a purely syntactic change, maintaining destructor
semantics.

## Problem

The
[accepted destructor syntax](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/classes.md#destructors)
includes out-of-line definitions such as:

```carbon
class MyClass {
  destructor [addr self: Self*];
}
destructor MyClass [addr self: Self*] { ... }
```

The implicit parameter here could be interpreted as either an implicit parameter
for `MyClass` or an implicit parameter for the destructor. How should
ambiguities like this be resolved?

For comparison, note a generic might look like:

```carbon
class GenericClass[T:! type](N:! T) { ... }
destructor GenericClass[T:! type](N:! T) [addr self: Self*] { ... }
```

The toolchain is able to parse this in constant time, but only because the lexer
will pair brackets, so we can do lookahead at the bracket in `GenericClass[` for
the closing `]`, and look past that for the `(` versus `{`. However, this is
arbitrary lookahead and may be significantly less efficient in other parsers
that people might want to use with Carbon, such as tree-sitter.

## Background

-   Proposal
    [#1154: Destructors](https://github.com/carbon-language/carbon-lang/pull/1154)
-   Leads question
    [#4999: Out-of-line destructor syntax ambiguity](https://github.com/carbon-language/carbon-lang/issues/4999)
-   [2025-02-25 Toolchain minutes](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.vootuzze8e8e)

In particular, we are discussing destruction as possibly similar to copy and
move syntax, and trying to create a consistency between the functions.

## Proposal

Destructor syntax will use standard function syntax, with `destroy` as a keyword
for the function name.

For example, in contrast with [problem examples](#problem):

```carbon
class MyClass {
  fn destroy[addr self: Self*]();
}
fn MyClass.destroy[addr self: Self*]() { ... }

class GenericClass[T:! type](N:! T) { ... }
fn GenericClass[T:! type](N:! T).destroy[addr self: Self*]() { ... }
```

It is invalid to add other implicit or explicit parameters to the `destroy`
function.

### Not directly callable

Although the syntax of `fn destroy` looks similar to a regular function, the
functions are not designed to be directly callable. This does not add support
for `my_var.destroy()`. See Proposal #1154, alternative
[Allow functions to act as destructors](/proposals/p1154.md#allow-functions-to-act-as-destructors)
for details.

## Future work

### Extend syntax to allow explicit marking of _trivial_ destructors

Discussion has indicated potential utility in syntax to make the expectation of
a trivial destructor _explicit_. This would allow a declarative way of ensuring
no member accidentally caused a type to have non-trivial destruction.

Still, this requires a further extension of syntax that isn't proposed at this
time. Both determining syntax for such a feature and motivating it fully are
left as future work.

### Decide whether to desugar destructors to interfaces

Under this proposal, `fn destroy` remains a special function. We may want to
make it desugar to an interface implementation, but even if we do so, the terse
destructor syntax seems likely to remain. There are concerns about the
ergonomics of requiring an `impl` in order to add a destructor to a type, and
decisions would need to be made for how virtual destructors should be handled.

### Copy and move functions

This proposal is set up for consistency with a possible `fn copy` and `fn move`,
but those will be evaluated as part of copy and move semantics.

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Eliminates ambiguity in `destructor` syntax, by creating consistency
        with `fn` syntax.
    -   Claiming `destroy` as a keyword is considered to be a good balance.
    -   Syntax choices, particularly with the keyword as a function name, should
        not create a barrier for desugaring to an interface approach for
        destructions.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Consistency with `fn` syntax should improve readability.
    -   Features that impact data layout are consistently written like member
        declarations.

## Alternatives considered

### Destructor syntax options

The ambiguity between `destructor MyClass [...]` out-of-line destructor syntax
and implicit parameters for generics is a sufficient barrier to change syntax.
We do not want parsing Carbon to require arbitrary lookahead.

`fn destroy` was preferred because it builds on existing `fn` syntax.

Although adding a `.`, as in `destructor MyClass.[...]`, was brought up, it
didn't present interesting advantages over `fn destroy`.

### Destructor name options

We expect more name conflicts with C++ code using the `destroy` keyword than
with the `destructor` keyword, for example with
[`std::allocator::destroy`](https://en.cppreference.com/w/cpp/memory/allocator/destroy),
or visible
[searching LLVM code](https://github.com/search?q=repository%3Allvm%2Fllvm-project+language%3Ac%2B%2B+symbol%3A%2F%28%3F-i%29%5Edestroy%24%2F&type=code).

Still, the phrasing of `destroy`, particularly if we have `copy` and `move` to
match, is preferred. Raw identifier syntax (`r#destroy`) is expected to be
sufficient for name conflicts.

`fn delete` was mentioned as an option reusing current keywords, but declined
due to the "heap allocated" implication of `delete`.

Non-keyword names were considered as part of proposal
[#1154: Destructors](https://github.com/carbon-language/carbon-lang/pull/1154),
and the trade-off considerations still apply.
