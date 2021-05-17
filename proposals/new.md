# Remove `Void`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

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
[easier to read, understand, and write](https://carbon-lang.dev/docs/project/goals.html#code-that-is-easy-to-read-understand-and-write),
by sparing programmers the need to understand the relationship between them, or
choose between them.

## Alternatives considered

See [issue 443](https://github.com/carbon-language/carbon-lang/issues/443).
