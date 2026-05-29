# Rename `me` -> `self`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1382)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Don't change anything](#dont-change-anything)
    -   [`this`](#this)

<!-- tocstop -->

## Problem

We've tried the `fn MethodName[me: Self]()` syntax for a while, and from our
experience the brevity of `me` is not worth doing something novel in this space.
We have found that `me` doesn't read well in practice.

## Background

The current method syntax, including these choices, was decided in
questions-for-leads issue
[#494: Method syntax](https://github.com/carbon-language/carbon-lang/issues/494)
and implemented in proposal
[#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722).

Looking at other languages that use reserved word for the receiver value:

| When       | Language   | Receiver when<br />accessing members | Receiver value | Receiver type |
| ---------- | ---------- | ------------------------------------ | -------------- | ------------- |
| 1983       | C++        | implicit                             | `this`         | ---           |
| 1991       | Python     | explicit                             | `self`         | ---           |
| 1995       | Java       | implicit                             | `this`         | ---           |
| 1995       | JavaScript | explicit                             | `this`         | ---           |
| 2000       | C#         | implicit                             | `this`         | ---           |
| 2009       | Go         | explicit                             | (see below)    | ---           |
| 2010       | Rust       | explicit                             | `self`         | `Self`        |
| 2011       | Kotlin     | implicit                             | `this`         | ---           |
| 2012       | TypeScript | explicit                             | `this`         | `this`        |
| 2014       | Swift      | implicit                             | `self`         | `Self`        |
| previously | Carbon     | explicit                             | `me`           | `Self`        |
| proposed   | Carbon     | explicit                             | `self`         | `Self`        |

In detail:

-   C++ uses
    [`this` for address of the receiver value](https://en.cppreference.com/w/cpp/language/this).
    C++23 includes
    [an explicit `this` facility](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0847r7.html).
    Examples in the proposal frequently use `self` as the name of the parameter,
    and `Self` as its type.
-   Swift uses
    [`self` for the receiver value](https://docs.swift.org/swift-book/LanguageGuide/Methods.html#ID238).
    and
    [`Self` for its type](https://docs.swift.org/swift-book/ReferenceManual/Types.html#ID610).
-   Rust uses
    [`self` for the receiver value](https://doc.rust-lang.org/std/keyword.self.html)
    and
    [`Self` for its type](https://doc.rust-lang.org/rust-by-example/fn/methods.html).
-   C# uses
    [`this` for a reference to the receiver value](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/this).
-   Python as a convention uses `self` for the receiver value, but it's almost
    universally followed.
-   Go conventionally uses an abbreviation of the type name.

Some history about the use of `self` in programming languages is documented in
[this StackOverflow answer](https://stackoverflow.com/a/1080192/624900),
including that `self` is also used in Smalltalk, Modula-3, Delphi/Object Pascal,
and Objective-C.

## Proposal

Use `self` instead of `me` to be consistent with Swift and Rust.

## Rationale

This is consistent with Carbon's goal to make
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
by choosing a keyword for this role that is less surprising to users.

## Alternatives considered

### Don't change anything

We could stay with the status quo, which has the benefit that `me` is shorter
than `self`. There are two considerations:

-   For accessing members of the current object, the chart in
    [the background section](#background) shows plenty of precedent for
    requiring a 4 character explicit keyword.
-   We would also like to reduce ceremony when declaring the signature of a
    method. For this concern, both `me: Self` and `addr me: Self*` are already
    longer than what other languages use in practice. It would probably be
    better to solve this problem with a shortcut approach like Rust
    ([1](https://doc.rust-lang.org/book/ch05-03-method-syntax.html),
    [2](https://doc.rust-lang.org/rust-by-example/fn/methods.html)), where
    `&self` is short for `self: &Self` and `&mut self` is short for
    `self: &mut Self`.

### `this`

We could also switch to `this`, primarily to benefit
[C++ users](https://en.cppreference.com/w/cpp/language/this). This had a few
disadvantages:

-   We are worried that it frequently not being a pointer would be surprising to
    those C++ users.
-   As noted in [the background section](#background), C++23 code using explicit
    this frequently uses the name `self`.
-   We view it as an advantage to use the same spelling for the variable `self`
    as for the type `Self`, and while `this` might make an acceptable name for
    the object parameter, `Self` is much more strongly established as the name
    for the current class, for example in PL research, and there is no precedent
    for a type named `This`.
