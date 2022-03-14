# Principle: One static open extension mechanism

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/998)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

There are a few ways of approaching open extension, such as defining how
operators are overload for every type. For Carbon, with its
[focus on performance](/docs/project/goals.md#performance-critical-software), we
are particularly interested in those that support static dispatch, to avoid the
runtime overhead of dynamic dispatch. The three main options are:

-   Open function overloading, where new overloads for a given name can be
    defined broadly, as is done in C++.
-   Interfaces, as is done in
    [Rust](https://doc.rust-lang.org/rust-by-example/trait/ops.html).
-   Special method names, as are used in
    [C++](https://en.cppreference.com/w/cpp/language/operators) and
    [Python](https://docs.python.org/3/reference/datamodel.html#special-method-names).

We would prefer to use a single mechanism, if possible, for simplicity.

## Proposal

Proposal is to use interfaces as the single open extension mechanism.

## Details

Details are in the added principle doc:
[docs/project/principles/static_open_extension.md](/docs/project/principles/static_open_extension.md).

## Rationale based on Carbon's goals

This proposal is pursuing Carbon's goal of having
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).

## Alternatives considered

Early arguments for this approach were put forth in
[this document](https://docs.google.com/document/d/1uvX_hmw5DVs1SFjnehUUizGnI6C099vqIGfUhfCwBIo/edit#).
