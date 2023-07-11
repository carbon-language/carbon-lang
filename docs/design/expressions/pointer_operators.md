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

Carbon provides the following operators related to pointers:

-   `&` as a prefix unary operator takes the address of an object, forming a
    pointer to it.
-   `*` as a prefix unary operator dereferences a pointer.

Note that [member access expressions](member_access.md) include an `->` form
that implicitly performs a dereference in the same way as the `*` operator.

## Details

The semantic details of pointer operators are collected in the main
[pointers](/docs/design/values.md#pointers) design. The syntax and precedence
details are covered here.

The syntax tries to remain as similar as possible to C++ pointer types as they
are commonly written in code and are expected to be extremely common and a key
anchor of syntactic similarity between the languages.

### Precedence

These operators have high precedence. Only [member access](member_access.md)
expressions can be used as an unparenthesized operand to them.

The two prefix operators `&` and `*` are generally above the other unary and
binary operators and can appear inside them as unparenthesized operands. For the
full details, see the [precedence graph](README.md#precedence).

## Alternatives considered

-   [Alternative pointer syntaxes](/proposals/p2006.md#alternative-pointer-syntaxes)

## References

-   [Proposal #2006: Values, variables, and pointers](/proposals/p2006.md)
