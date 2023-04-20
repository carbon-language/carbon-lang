# Pointer operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Precedence](#precedence)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides three unary operators related to pointers:

-   `&` as a prefix unary operator takes the address of an object, forming a
    pointer to it.
-   `*` as a prefix unary operator dereferences a pointer.
-   `*` as a postfix unary operator produces a pointer _type_ to some other
    type.

## Details

The semantic details of pointer operators are collected in the main
[pointers](/docs/design/values.md#pointers) design. The syntax and precedence
details are covered here.

The syntax tries to remain as similar as possible to C++ pointer types as they
are commonly written in code and are expected to be extremely common and a key
anchor of syntactic similarity between the languages. The different alternatives
and tradeoffs for this syntax issue were discussed extensively in
[#523](https://github.com/carbon-language/carbon-lang/issues/523).

### Precedence

All of these operators have high precedence. Only
[member access](member_access.md) expressions can be used as an unparenthesized
operand to them. The postfix operator, because it is exclusively a _type_
operator, doesn't have a precedence relationship with most other operators. The
prefix operators, however, are generally above the other unary and binary
operators and can appear inside them as unparenthesized operands. For the full
details, see the [precedence graph](README.md#precedence).

## Alternatives considered

TODO

## References

-   Proposal
    [#2006: Values, variables, and pointers](https://github.com/carbon-language/carbon-lang/pull/2006).
