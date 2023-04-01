# Principle: Information accumulation

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of this principle](#applications-of-this-principle)
-   [Exceptions](#exceptions)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Background

There are many different sources of information in a program, and a tool or a
human interpreting code will not in general have full information, but will
still want to draw conclusions about the code.

Different languages take different approaches to this problem. For example:

-   In C, information is accumulated linearly in each source file independently,
    and only information from earlier in the same file is available. A program
    can observe that information is incomplete at one point and complete at
    another.
-   In C++, the behavior is largely similar to C, except:
    -   Within certain contexts in a class, information from later in the class
        definition is available.
    -   With C++20 modules, information from other source files can be made
        available.
    -   It is easier to observe -- perhaps even accidentally -- that information
        is accumulated incrementally.
-   In Rust, all information from the entire crate is available everywhere
    within that crate, with exceptions for constructs like proc macros that can
    see the state of the program being incrementally built.
-   In Swift, all information from the entire source file is available within
    that source file.

## Principle

In Carbon, information is accumulated incrementally within each source file.
Carbon programs are invalid if they would have a different meaning if more
information were available.

Carbon source files can be interpreted top-down, without referring to
information that appears substantially later in a file. Source files are
expected to be organized into a topological order where that makes sense, with
forward declarations used to introduce names before they are first referenced
when necessary.

If a program attempts to use information that has not yet been provided, the
program is invalid. There are multiple options for how this can be reported:

-   The program can be rejected as soon as it tries to use information that
    might not be known yet.
-   For the case where the information can only be provided in the same source
    file, an assumption about the information can be made at the point where it
    is needed, and the program can be rejected only if that assumption turns out
    to be incorrect.

Disallowing programs from changing meaning in the context of more information
ensures that the program is interpreted consistently or is rejected. This is
especially important to the coherence of generics and templates.

## Applications of this principle

-   As in C++, and unlike in Rust and Swift, name lookup only finds names
    declared earlier.
-   Classes are incomplete until the end of their definition. Unlike in C++, any
    attempt to observe a property of an incomplete class that is not known until
    the class is complete renders the program invalid.
-   When an `impl` needs to be resolved, only those `impl` declarations that
    have that appear earlier are considered. However, if a later `impl`
    declaration would change the result of any earlier `impl` lookup, the
    program is invalid.

## Exceptions

Because a class is not complete until its definition has been fully parsed,
applying this rule would make it impossible to define most member functions
within the class definition. In order to still provide the convenience of
defining class member functions inline, such member function bodies are deferred
and processed as if they appeared immediately after the end of the outermost
enclosing class, like in C++.

## Alternatives considered

-   Allow information to be used before it is provided
    [globally](/proposals/p0875.md#strict-global-consistency),
    [within a file](/proposals/p0875.md#context-sensitive-local-consistency), or
    [within a top-level declaration](/proposals/p0875.md#top-down-with-minimally-deferred-type-checking).
-   [Do not allow inline method bodies to use members before they are declared](/proposals/p0875.md#strict-top-down)
-   [Do not allow separate declaration and definition](/proposals/p0875.md#disallow-separate-declaration-and-definition)
