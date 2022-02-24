# Remove `Void`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/540)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

The `Void` type as
[currently specified](https://github.com/carbon-language/carbon-lang/blob/4bf396b8f6e7f5289c170c5ad9dda64c5c680d4a/docs/design/README.md#primitive-types)
is redundant with `()`, the type of a tuple with no elements.

## Background

[Issue 443](https://github.com/carbon-language/carbon-lang/issues/443) contains
further discussion of the problem, and possible solutions. The consensus of the
Carbon leads was that `Void` should be removed.

## Proposal

Remove `Void` from the Carbon design.

## Rationale based on Carbon's goals

Eliminating `Void` will make Carbon code
[easier to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
The main advantage of `Void` is that it is recognizable and familiar to C++
programmers. However, we haven't yet found any use cases where using `Void`
results in clearer code, even to programmers transitioning from C++. In
particular, omitting a function's return type is more concise and at least as
clear as explicitly specifying `-> Void`. In most other use cases, the
appearance of familiarity is more likely to mislead than to clarify: most other
use cases for C++ `void`, such as using `void*` to mean "pointer to anything",
will not work with Carbon's `Void`, and most other use cases for Carbon's
`Void`, such as using it as the type of a variable, would not work with C++'s
`void`,

## Alternatives considered

-   Define `Void` as an alias for `()`. This is workable, but forces users to
    understand both spellings, and make a style choice between them.
-   Define `Void` as a distinct type from `()` with the same semantics. This
    forces users to know "which kind of nothing" to use in any given context
-   Define `Void` as a distinct type from `()`, with more C++-like semantics.
    This would reproduce the problems of C++'s `void`, for no clear benefit.
-   Eliminate `()`. This would needlessly complicate programming with tuples,
    especially in variadic settings.

See [issue 443](https://github.com/carbon-language/carbon-lang/issues/443) for
details.
