# Carbon: Add initial version of "Carbon tuples"

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/todo)

## Table of contents

<!-- toc -->

- [Problem](#problem)
- [Proposal](#proposal)

<!-- tocstop -->

## Problem

Tuples in Carbon play a role in several parts of the language:

- They are a light-weight product type.
- They support multiple return values from functions.
- They provide a way to specify a literal that will convert into a struct value.
- They are involved in pattern matching use cases (such as function signatures)
  particularly for supporting varying numbers of arguments, called "variadics."
- A tuple may be unpacked into multiple arguments of a function call.

## Proposal

We propose adding (`docs/design/tuples.md`)[/docs/design/tuples.md], which has
the details.
