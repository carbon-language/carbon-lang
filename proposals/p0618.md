# var ordering

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/618)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Type ordering](#type-ordering)
        -   [`<type>: <name>`](#type-name)
        -   [`<type> <name>`](#type-name-1)
        -   [`<name>: <type>`](#name-type)
    -   [`:` versus `in`](#-versus-in)

<!-- tocstop -->

## Problem

As stated by on
[Re-evaluate core variable & parameter identifier/type order (including a default for parameters) #542](https://github.com/carbon-language/carbon-lang/issues/542):

> Somewhat condensed, bullet-point-y background for this question:
>
> -   We've been using first `Type: variable` and then `Type variable` syntax in
>     variables, parameters, and other declarations.
> -   This was primarily based on a _lack of compelling data_ to select a better
>     syntax, with trying to stay similar to C++ as a fallback.
> -   It was _specifically_ intended to be revisited. The expected trigger for
>     this was some form of broader user data (surveys at least of decent #s of
>     developers, or potentially real user studies).
> -   However, we have gained specific new information as we've filled out a
>     great deal of the surrounding syntax. We have also gotten some data on
>     parsing challenges (although perhaps surmountable challenges) of
>     `Type variable`.
> -   We also don't have a short/definite timeline to start getting useful data.
> -   The leads should re-evaluate the core variable syntax based on the new
>     information we have, but _without_ trying to wait for data.
>     -   We can always re-evaluate again if and when data arrives and indicates
>         any need for it.
> -   The leads should do this ASAP and make a decision so that we can focus our
>     energy, reduce frustrating discussions, and have consistent syntax in
>     examples and proposals.

## Background

Background may be found in the related
[#542](https://github.com/carbon-language/carbon-lang/issues/542) and
[Docs](https://docs.google.com/document/d/1EhZA3AlY9TaCMho9jz2ynFxK-6eS6BwMAkE5jNYQzEA/edit?usp=sharing&resourcekey=0-QXEoh-b4_sQG2u636gIa1A).

## Proposal

Two changes:

-   Switch to `<name>: <type>`, replacing `<type>: <name>`.
-   Use `in` instead of `:` in range-based for loops.

Note these changes were largely implemented by
[#563](https://github.com/carbon-language/carbon-lang/pull/563).

## Rationale based on Carbon's goals

Both of these changes are done for consistency with other modern languages,
particularly Swift and Rust. The switch from `:` to `in` is for ease of
understanding and parsing.

## Alternatives considered

### Type ordering

Alternatives are pulled from
[Docs](https://docs.google.com/document/d/1EhZA3AlY9TaCMho9jz2ynFxK-6eS6BwMAkE5jNYQzEA/edit?usp=sharing&resourcekey=0-QXEoh-b4_sQG2u636gIa1A).

#### `<type>: <name>`

`var String: message = "Hello world";`

Advantages:

-   Roughly matches the order of C, C++, C#, D and Java, except with extra `var`
    and `:`.
-   Type at the beginning puts most important information up front.
-   Name followed by default matches assignment statements.

Disadvantages:

-   Existing languages that use a `:` put the name before and the type after
    ([universally](http://rosettacode.org/wiki/Variables)).
-   Beyond simple inconsistency, the overlap of `:` in this syntax with
    different order will add confusion for people working/familiar with multiple
    languages.
-   Does not end up having a syntax that is consistent with using colons for
    marking labelled parameters and arguments, such as how Swift does.
    -   We currently do not plan to use a colon syntax for labelled parameters
        and arguments, regardless of the decision here.

Opinions vary:

-   Not friendly to optionally dropping the `type:` to represent auto type
    deduction.

#### `<type> <name>`

`var String message = "Hello world";`

Advantages

-   Matches C, C++, C#, D and Java the closest.

Disadvantages:

-   Creates parse ambiguity, particularly when we start adding syntax to the
    name to indicate that a parameter is labeled, etc.

Currently hard to see how we can make this work, since it isn't compatible with
other choices, detailed in
[Docs](https://docs.google.com/document/d/1EhZA3AlY9TaCMho9jz2ynFxK-6eS6BwMAkE5jNYQzEA/edit?usp=sharing&resourcekey=0-QXEoh-b4_sQG2u636gIa1A).

#### `<name>: <type>`

`var message: String = "Hello world";`

Advantages:

-   Matches [Swift](http://rosettacode.org/wiki/Variables#Swift),
    [Rust](https://doc.rust-lang.org/stable/rust-by-example/primitives.html),
    [Kotlin](http://rosettacode.org/wiki/Variables#Kotlin),
    [Python3](https://docs.python.org/3/library/typing.html), and many smaller
    languages ([Ada](http://rosettacode.org/wiki/Variables#Ada),
    [Pascal languages like Delphi](http://rosettacode.org/wiki/Variables#Delphi)
    and [Modula-3](http://rosettacode.org/wiki/Variables#Modula-3),
    [Eiffel](http://rosettacode.org/wiki/Variables#Eiffel),
    [Nim](https://nim-lang.org/docs/tut1.html#the-var-statement),
    [Pony](http://rosettacode.org/wiki/Variables#Pony),
    [Zig](https://ziglang.org/documentation/0.7.1/#Variables)).
-   Names will line up better with method names in a `struct` definition.

Disadvantages:

-   Name separated from initializer; default doesn't match assignment
    statements.
-   Further from the simplistic appearance of common C and C++ variable
    declarations.

Opinions vary:

-   Existing languages typically make the "`: type`" part optional when an
    "`= value`" clause is present.

### `:` versus `in`

The `:` operator for range-based for loops becomes harder to read, and more
likely to cause ambiguity, when `:` is also used for `var`. That is,
`for (var i: Int : list)` is just harder to understand than
`for (var i: Int in list)`. `in` is a favorable choice for its use in other
languages.
