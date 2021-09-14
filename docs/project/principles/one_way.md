# Principle: Only provide one way of doing things

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of these principles](#applications-of-these-principles)

<!-- tocstop -->

## Background

It's common in programming to provide multiple, equivalent ways of doing the
same thing. Sometimes this reflects the legacy of a language, while some
reflects a desire to provide both verbose and concise versions of the same
syntax.

## Principle

In Carbon, we will generally try to only provide one way of doing things. That
is, given a syntax scenario where multiple options are available, we will tend
to provide _one_ option rather than than providing several and letting users
choose. This serves several goals:

-   [Language tools](/docs/project/goals.md#language-tools-and-ecosystem) should
    be easier to write and maintain with the lower language complexity implied
    by less duplication of functionality.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    processes should find it easier to both consider existing syntax and avoid
    creation of new syntax conflicts.
-   [Understandability of code](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    should be promoted if developers have less syntax they need to understand.
    This can be expected to improve code quality and productivity so long as the
    resulting code structures aren't overly complicated.

A couple situations where this will not apply are:

-   For [evolution](/docs/project/goals.md#software-and-language-evolution), it
    will often be necessary to temporarily provide an "old" and "new" way of
    doing things simultaneously.
    -   For example, if renaming a method.
-   For
    [migration and interoperability](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code),
    it may be pragmatic to provide both an ideal way of doing things for new
    Carbon code, and a separate approach that is more C++-compatible for
    migration.
    -   For example, consider generics and templates: generics are considered to
        be the preferred form for new code, but templates are considered a
        necessity for migration of C++ code. This is not an evolution situation
        because we do not anticipate ever removing templates.

## Applications of these principles

We can observe the application of this principle by comparing several language
features to C++:

-   Where C++ allows logical operators to be written with either symbols (for
    example, `&&`) or text (for example, `and`), Carbon will only support one
    form (in this case, [text](/proposals/p0680.md)).
    -   This is motivated by improving understandability.
-   Where C++ allows braces to be omitted for single-statement control flow
    blocks, Carbon will [require braces](/proposals/p0623.md).
    -   This is motivated by simplifying syntax and improving evolvability.
-   Where C++ allows hexadecimal numeric literals to be either lowercase
    (`0xaa`) or uppercase (`0xAA`), and with `x` optionally uppercase as well,
    Carbon will only allow the [`0xAA` casing](/proposals/p0143.md).

This should not be taken to extremes: where there is a significant difference in
syntax, Carbon may support multiple forms. For example, `for (var x: list)`
could typically be written with as a `while` loop, but it offers sufficient
syntactic advantage that it's [accepted](/proposals/p0353.md). However,
`for (;;)` syntax is sufficiently close to `while` that we hope to collapse it
into `while` syntax.
