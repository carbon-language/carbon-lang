# Update sum types design

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2187)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep the continuation interface as a parameter](#keep-the-continuation-interface-as-a-parameter)
    -   [Require the continuation to execute the whole `case` body](#require-the-continuation-to-execute-the-whole-case-body)

<!-- tocstop -->

## Abstract

This proposal updates the design of [#157](p0157.md) to reflect subsequent
evolution of the language.

## Problem

Some aspects of the design of #157 have become inconsistent with the rest of the
language, or can be made more precise in light of subsequent language
development.

## Proposal

-   Make the continuation interface an associated type rather than an interface
    parameter.
-   For patterns that can be matched using either `Match` or `==`, require both
    implementations to be well-formed.
-   Use `name: Type` order instead of `Type: name` order in alternative
    declarations.
-   Use current syntax and semantics for generics and class types.
-   Rename `Matchable.Match` to `Match.Op`, following the resolution of
    [#1058](https://github.com/carbon-language/carbon-lang/issues/1058).

## Alternatives considered

### Keep the continuation interface as a parameter

Making the continuation interface a parameter of `Match` could in principle
allow a single type to support pattern matching in multiple ways, by
implementing `Match` for multiple continuation interfaces. However, that would
require something like overload resolution on interfaces, to choose the
implementation of `Match(C)` on the sum type for which `C` best matches the
continuation constructed by the compiler. No such overloading mechanism is
planned, and we don't have sufficiently compelling use cases to motivate it.

### Require the continuation to execute the whole `case` body

We will probably want to support in-place mutation of alternative parameters
(for example so you can call a mutable method on the value stored in an
`Optional(Foo)`), and we might even want to extend that to cases where the
underlying parameter isn't represented as an lvalue of that type, but has to be
unpacked by `Match.Op`. The only way I see to make that work is to have
`Match.Op` unpack the parameter to a local lvalue, pass it to the continuation,
and then pack the possibly-mutated value back into the sum object after the
continuation returns. That would mean the compiler has to execute the case body
inside the continuation, not after `Match.Op` returns.

However, that's fairly speculative, and wouldn't apply to the read-only cases
that we currently support, so we need not preemptively constrain the compiler
here.
