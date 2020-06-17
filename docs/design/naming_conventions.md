# Naming conventions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

We would like to have widespread and consistent naming conventions across Carbon
code to the extent possible. This is for the same core reason as naming
conventions are provided in most major style guides. Even migrating existing C++
code at-scale presents a significant opportunity to converge even more broadly
and we're interested in pursuing this if viable.

Our current proposed naming convention, which we at least are attempting to
follow within Carbon documentation in order to keep code samples as consistent
as possible:

- `UpperCamelCase` for names of compile-time resolved constants, such that they
  can participate in the type system and type checking of the program.
- `lower_snake_case` for names of run-time resolved values.

As an example, an integer that is a compile-time constant sufficient to use in
the construction a compile-time array size might be named `N`, where an integer
that is not available as part of the type system would be named `n`, even if it
happened to be immutable or only take on a single value. Functions and most
types will be in `UpperCamelCase`, but a type where only run-time type
information queries are available would end up as `lower_snake_case`.

We only use `UpperCamelCase` and `lower_snake_case` (skipping other variations
on both snake-case and camel-case naming conventions) because these two have the
most significant visual separation. For example, the value of adding
`lowerCamelCase` for another set seems low given the small visual difference
provided.

The rationale for the specific division between the two isn't a huge or
fundamental concept, but it stems from a convention in Ruby where constants are
named with a leading capital letter. The idea is that it mirrors the English
language capitalization of proper nouns: the name of a constant refers to a
_specific_ value that is precisely resolved at compile time, not just to _some_
value. For example, there are many different _shires_ in Britain, but Frodo
comes from the _Shire_ -- a specific fictional region.
