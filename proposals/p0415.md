# Syntax: `return`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/415)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Deferred questions](#deferred-questions)
-   [Alternatives considered](#alternatives-considered)
    -   [Implicit or expression returns](#implicit-or-expression-returns)

<!-- tocstop -->

## Problem

Carbon has functions. We should write down how those functions indicate their
code should stop running and, if applicable, what result should be provided back
to their caller.

## Background

The Carbon overview [contains](/docs/design/README.md#return) a "skeletal
design" for `return`.

Carbon aims for
[_familiarity for experienced C++ developers with a gentle learning curve_](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).
C++ returns from functions with a `return` statement, which is documented well
[here](https://en.cppreference.com/w/cpp/language/return).

## Proposal

Carbon will have a `return` statement that is unsurprising to C++ programmers.

Each of these is a valid Carbon function:

```
// A function that returns no value
fn NoReturn() {
  // This function intentionally left blank
}


// A function that returns no value in a different way
fn ReturnNoValue() {
  // Usually no expression in a function with no return value.
  return;
}


// A function that returns a value
fn ReturnFive() -> Int {
  // Must include an expression "convertible" to `Int`.
  // The (lack of) definition for "convertible" is addressed later.
  return 5;
}
```

## Details

We define a `return` statement in Carbon. It can occur any place a statement is
allowed. It takes one optional expression, the _return value_. Functions may
contain multiple `return` statements.

A function that returns `Void` may have `return` statements. Most will have no
return value, though some may provide a `Void`-type expression, for example the
result of another `Void`-returning function.

> This last affordance is provided to avoid making `Void` a special case in
> templating.

Functions returning `Void` are also defined to end with an implicit `return`
statement.

> This implicit `return` is intended to make it easier for later designs to
> refer to "the (now always present) `return` statement" rather than the less
> well-defined "the point the function returns".

A function that returns anything other than `Void` must have at least one
`return` statement. All `return` statements in the function must provide an
expression "convertible" to the function's return type. If a control flow path
reaches the end of the function without executing a `return` statement, it is a
compile error.

### Deferred questions

This proposal is limited to the syntax and simple static semantics of the
`return` statement.

Some questions about the full validity and meaning of a `return` statement in
context are left unresolved, in the expectation they will be addressed when
adjacent parts of the language are specified.

Those include:

-   **What does "convertible" mean?**

    When a function returns a value, we say the `return` statement takes an
    expression "convertible" to the function's return type. As a coarse
    approximation, this may mean "assignable to a variable of the function's
    return type", but there are likely to be subtleties. We assume this question
    will be addressed as the type system develops.

-   **What optimizations are encouraged or guaranteed for return values?**

    C++ has
    [copy elision](https://en.cppreference.com/w/cpp/language/copy_elision) and
    it's a good bet Carbon will too. But first we need to decide what copying
    means.

-   **What code, if any, runs between `return` and _returning_?**

    Carbon is likely to have approaches for deterministic cleanup that trigger
    when a function's code stops executing -- analogues to C++ destructors,
    Golang deferred functions, or Java's `finally` blocks. Once we know what
    they are and how we intend them to work, we can determine how to slot them
    in next to `return`.

## Alternatives considered

### Implicit or expression returns

Some C-family languages allow an unadorned expression in the right context to
serve as a return value.

In Rust, for example, function bodies are
[block expressions](https://doc.rust-lang.org/reference/expressions/block-expr.html),
which may have an expression as their final clause and which take the value of
that expression.

This Rust function returns `5`:

```rust
fn return_five() -> i32 {
    perhaps_unrelated_action();
    5
}
```

Rust still has a `return` statement, so this is entirely an ergonomic feature.

We can choose to add a similar feature to Carbon in the future as long as we can
unambiguously discern expressions and statements in parsing.
