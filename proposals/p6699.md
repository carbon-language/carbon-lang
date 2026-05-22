# Diagnostic sorting

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6699)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Don't sort diagnostics](#dont-sort-diagnostics)
    -   [Sort by line and column](#sort-by-line-and-column)
    -   [Sort by last processed token](#sort-by-last-processed-token)

<!-- tocstop -->

## Abstract

Change `SortingConsumer` from sorting by last processed token (per-phase) to
additionally allow diagnostics to request sorting by start position (line and
column) when the last processed token is the same.

## Problem

Diagnostics in many toolchains are emitted in the order they are discovered
during code traversal. While this naturally reflects the causal relationship
between errors (for example, an error in a macro expansion causing subsequent
parse errors), it can lead to a confusing experience for developers if the
diagnostics jump back and forth through the file. Conversely, sorting purely by
source location (line and column) can break causality, presenting a consequence
before its cause. We need a sorting strategy that feels natural to humans but
respects the underlying toolchain logic.

## Background

Carbon's processing of code in stages (lex, parse, check) causes diagnostics to
be produced in that order. In contrast, Clang interleaves parse and check, and
as a consequence the diagnostics produced are similarly interleaved.

A more detailed overview of Carbon's diagnostic infrastructure can be found in
[diagnostics.md](/toolchain/docs/diagnostics.md).

## Proposal

In addition to sorting by the last processed token (which `SortingConsumer`
already does), add a way to sort based on the start position (line and column)
by request. This is being called "on-scope" because current cases we've
discussed are scope-related.

See [SortingConsumer](/toolchain/docs/diagnostics.md#sortingconsumer) for more
documentation.

## Details

This was already implemented in
[PR #6687](https://github.com/carbon-language/carbon-lang/pull/6687).

## Rationale

This proposal advances Carbon's goal of providing
[Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
and creating
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
by making the developer experience with error messages more predictable and
logical. It respects the inherent causal order of the toolchain while tailoring
the output to human expectations.

## Alternatives considered

### Don't sort diagnostics

We could just print diagnostics in the order they are produced. This would print
all lex errors, then all parse errors, then all check errors. This would be
simple, but might be confusing when a parse error at the end of a file comes
before check errors, and fixing the check errors would fix the parse error.

### Sort by line and column

We could sort diagnostics purely by their line and column. This runs into issues
with cases such as:

```carbon
fn F(x: i32, y: i32);

fn G() {
  F(1 2);
}
```

Here, the diagnostic for an invalid parse of `1 2` would appear after the
diagnostic that `F` expects two arguments, not one. This is confusing because
the missing comma is the root cause of the incorrect argument count.

### Sort by last processed token

We could sort diagnostics purely by the last token that was processed when the
diagnostic was emitted. This runs into issues with cases such as:

```carbon
fn F(x: i32, y: i32) {}
```

Here, both `x` and `y` would be diagnosed as unused at the `}`. The order would
be non-deterministic, hindering golden tests.

This could be partially addressed by sorting the diagnostics locally (for
example, sorting each `unused` diagnostic together), but this is an incomplete
solution because we may introduce further scope-related checks, particularly
flow checking (for example, checking if there are provable out-of-bounds
accesses). These would all have the same last processed tokens. It also would
likely lead to sorting regardless of whether sorting was requested by the tool
user, a performance overhead we want to avoid.

We believe sorting by the last processed token is a partial solution, which
we're building on.
