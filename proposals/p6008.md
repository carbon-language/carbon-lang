# Replace `impl fn` with `override fn`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6008)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)

<!-- tocstop -->

## Abstract

This proposal renames the syntax used to mark an overriding definition of a
virtual method from `impl fn` to `override fn` to avoid ambiguity.

## Problem

The phrase `impl fn` was introduced to mark overriding virtual functions,
however, it is now ambiguous: besides indicating an overriding virtual function,
it can be parsed as an "impl" declaration when the construct following "impl"
begins with a lambda introduced by "fn".

## Background

The original syntax was adopted in
[proposal #777](https://github.com/carbon-language/carbon-lang/pull/777) to
emphasize that implementing interfaces and overriding virtual functions are
similar operations.

## Proposal

This proposal is to replace the `impl fn` syntax with `override fn` for method
overriding in class inheritance.

Note that `impl` is still a modifier keyword for library and package
declarations (`impl library ...` and `impl package ...` respectively). The
former is unambiguous and the latter is easy to disambiguate, so no change is
needed for these two cases.

## Rationale

This proposal is focused on the
[That code is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
Carbon goal. Changing the keyword from `impl` to `override` makes the intent
clearer, resolves syntax ambiguity, and adheres to existing C++ conventions.
