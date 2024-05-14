# Functions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Function definitions](#function-definitions)
    -   [Return clause](#return-clause)
    -   [`return` statements](#return-statements)
-   [Function declarations](#function-declarations)
-   [Function calls](#function-calls)
-   [Functions in other features](#functions-in-other-features)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Functions are the core building block for applications. Carbon's basic function
syntax is:

-   _parameter_: _identifier_ `:` _expression_
-   _parameter-list_: _[ parameter_ `,` _parameter_ `,` _... ]_
-   _return-clause_: _[_ `->` _< expression |_ `auto` _> ]_
-   _signature_: `fn` _identifier_ `(` _parameter-list_ `)` _return-clause_
-   _function-definition_: _signature_ `{` _statements_ `}`
-   _function-declaration_: _signature_ `;`
-   _function-call_: _identifier_ `(` _[ expression_ `,` _expression_ `,` _...
    ]_ `)`

A function with only a signature and no body is a function declaration, or
forward declaration. When the body is a present, it's a function definition. The
body introduces nested scopes which may contain local variable declarations.

## Function definitions

A basic function definition may look like:

```carbon
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

This declares a function called `Add` which accepts two `i64` parameters, the
first called `a` and the second called `b`, and returns an `i64` result. It
returns the result of adding the two arguments.

C++ might declare the same thing:

```cpp
std::int64_t Add(std::int64_t a, std::int64_t b) {
  return a + b;
}

// Or with trailing return type syntax:
auto Add(std::int64_t a, std::int64_t b) -> std::int64_t {
  return a + b;
}
```

### Return clause

The return clause of a function specifies the return type using one of three
possible syntaxes:

-   `->` followed by an _expression_, such as `i64`, directly states the return
    type. This expression will be evaluated at compile-time, so must be valid in
    that context.
    -   For example, `fn ToString(val: i64) -> String;` has a return type of
        `String`.
-   `->` followed by the `auto` keyword indicates that
    [type inference](type_inference.md) should be used to determine the return
    type.
    -   For example, `fn Echo(val: i64) -> auto { return val; }` will have a
        return type of `i64` through type inference.
    -   Declarations must have a known return type, so `auto` is not valid.
    -   The function must have precisely one `return` statement. That `return`
        statement's expression will then be used for type inference.
-   Omission indicates that the return type is the empty tuple, `()`.
    -   For example, `fn Sleep(seconds: i64);` is similar to
        `fn Sleep(seconds: i64) -> ();`.
    -   `()` is similar to a `void` return type in C++.

### `return` statements

The [`return` statement](control_flow/return.md) is essential to function
control flow. It ends the flow of the function and returns execution to the
caller.

When the [return clause](#return-clause) is omitted, the `return` statement has
no expression argument, and function control flow implicitly ends after the last
statement in the function's body as if `return;` were present.

When the return clause is provided, including when it is `-> ()`, the `return`
statement must have an expression that is convertible to the return type, and a
`return` statement must be used to end control flow of the function.

## Function declarations

Functions may be declared separate from the definition by providing only a
signature, with no body. This provides an API which may be called. For example:

```carbon
// Declaration:
fn Add(a: i64, b: i64) -> i64;

// Definition:
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

The corresponding definition may be provided later in the same file or, when the
declaration is in an
[`api` file of a library](code_and_name_organization/#libraries), in the `impl`
file of the same library. The signature of a function declaration must match the
corresponding definition. This includes the [return clause](#return-clause);
even though an omitted return type has equivalent behavior to `-> ()`, the
presence or omission must match.

## Function calls

Function calls use a function's identifier to pass multiple expression arguments
corresponding to the function signature's parameters. For example:

```carbon
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}

fn Run() {
  Add(1, 2);
}
```

Here, `Add(1, 2)` is a function call expression. `Add` refers to the function
definition's identifier. The parenthesized arguments `1` and `2` are passed to
the `a` and `b` parameters of `Add`.

## Functions in other features

Other designs build upon basic function syntax to add advanced features:

-   [Generic functions](generics/overview.md#generic-functions) adds support for
    deduced parameters and compile-time parameters.
-   [Class member functions](classes.md#member-functions) adds support for
    methods and class functions.

## Alternatives considered

-   [Function keyword](/proposals/p0438.md#function-keyword)
-   [Only allow `auto` return types if parameters are compile-time](/proposals/p0826.md#only-allow-auto-return-types-if-parameters-are-generic)
-   [Provide alternate function syntax for concise return type inference](/proposals/p0826.md#provide-alternate-function-syntax-for-concise-return-type-inference)
-   [Allow separate declaration and definition](/proposals/p0826.md#allow-separate-declaration-and-definition)

## References

-   Proposal
    [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
-   Proposal
    [#826: Function return type inference](https://github.com/carbon-language/carbon-lang/pull/826)
