# Extended types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
-   [`Core.ExtType` values](#coreexttype-values)
    -   [Primitive extended type literals](#primitive-extended-type-literals)
    -   [`exttypeof`](#exttypeof)
    -   [Compound extended type values](#compound-extended-type-values)
    -   [The `.type` property](#the-type-property)
-   [Extended type facet types](#extended-type-facet-types)
-   [`fwd` bindings, arguments, and returns](#fwd-bindings-arguments-and-returns)
    -   [`fwd` bindings](#fwd-bindings)
    -   [`fwd` call arguments](#fwd-call-arguments)
    -   [`fwd` return tags](#fwd-return-tags)
    -   [`auto` extended types](#auto-extended-types)
-   [Deducing extended type values](#deducing-extended-type-values)
-   [Forwarding](#forwarding)
-   [References](#references)

<!-- tocstop -->

## TODO

> **TODO:** Fill in this design document to reflect
> [proposal #8183: First-class extended types and `fwd`](/proposals/p008183-first-class-extended-types-and-fwd.md).

This is a stub document for the design of first-class extended types
(`Core.ExtType`), `fwd`, and forwarding. For the underlying model of expression
categories, phases, and extended types of expressions, see
[Values, variables, and pointers](values.md#extended-types).

## Overview

> **TODO:** Summarize how `Core.ExtType` values reify the
> [extended type](values.md#extended-types) (type, category, phase, and
> compile-time value) of bindings, returns, and expressions as compile-time
> values, and explain how this relates to
> [expression extended types](values.md#extended-types) (including why
> `Core.ExtType` uses a single `"compile time"` phase rather than separate
> `"template"` and `"symbolic"` phases). See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#extended-types-and-values).

## `Core.ExtType` values

### Primitive extended type literals

> **TODO:** Document `exttype(val T)`, `exttype(ref T)`, and `exttype(var T)`
> literals for runtime primitive `Core.ExtType` values. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#primitive-extended-type-literals).

### `exttypeof`

> **TODO:** Document `exttypeof(expression)` for computing the `Core.ExtType` of
> an expression without runtime evaluation, including capturing compile-time
> phase and value. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#exttypeof).

### Compound extended type values

> **TODO:** Document constructing compound tuple and struct `Core.ExtType`
> values by way of `as Core.ExtType` (and implicit conversion in `fwd` binding
> and return positions), including nesting. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#compound-extended-type-values).

### The `.type` property

> **TODO:** Document `<exttype_value>.type` producing the unqualified type
> (stripping `const` and `partial` qualifiers, recursively for tuple and struct
> extended types), and its use in constraints. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#type-property-of-extended-type-values).

## Extended type facet types

> **TODO:** Document extended type facet types (`Core.NoValExtType`,
> `Core.NoRefExtType`, `Core.NoVarExtType`, `Core.NoCompileTimeExtType`,
> `Core.NoRunTimeExtType`), combining them with `&`, conversion rules between
> extended type facet types, and constraining with `where .type = T` and
> `where .type impls C`. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#extended-type-facet-types)
> and
> [any extended type with this type](/proposals/p008183-first-class-extended-types-and-fwd.md#any-extended-type-with-this-type).

## `fwd` bindings, arguments, and returns

### `fwd` bindings

> **TODO:** Document `fwd n: B` and anonymous `fwd _: B` binding patterns,
> implicit conversion of `B` to `Core.ExtType`, allowed nesting (tuple, struct,
> pack expansion `fwd each`, `unused`), disallowed combinations with other
> keyword binding modifiers (`ref`, `generic`, `template`, `runtime`) and
> pattern operators (such as `var`), and overriding contextual default phases.
> See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#specifying-the-extended-type-of-a-binding).

### `fwd` call arguments

> **TODO:** Document marking call arguments that have a symbolic extended type
> with `fwd` (for example `F(fwd ~y)`), ordering with `~`, and omission on
> method receivers and function call callees. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#specifying-the-extended-type-of-a-binding).

### `fwd` return tags

> **TODO:** Document `-> fwd R`, `-> fwd auto`, and nested `fwd` inside tuple
> and struct extended return types (anywhere `val`, `ref`, or `var` is allowed).
> See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#specifying-the-extended-type-of-a-return).

### `auto` extended types

> **TODO:** Document `auto` in `Core.ExtType` positions: referentially
> transparent `let fwd x: auto = y;` bindings, `-> fwd auto` returns (and why
> `=>` is unchanged), and positional parameters (`fn F(... fwd _: auto)`). See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#auto-extended-type).

## Deducing extended type values

> **TODO:** Document deducing `Core.ExtType` parameters from argument
> expressions (compile-time, `var`, `ref`, `val`), category conversion when
> deducing restricted extended type facet types (`Core.No*ExtType`), the
> requirement that `ref` or `fwd` be written at the call site to deduce a `ref`
> extended type, deducing types through `where .type = ...`, exact-value
> compile-time parameters (`fwd _: exttypeof(V)`), and exact type matching
> (`[fwd T: exttypeof(bool)](x: T)`). See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#deducing-an-extended-type-value),
> [compile-time parameter equal to a specified value](/proposals/p008183-first-class-extended-types-and-fwd.md#compile-time-parameter-equal-to-a-specified-value),
> and
> [parameter requires exact type match](/proposals/p008183-first-class-extended-types-and-fwd.md#parameter-requires-exact-type-match).

## Forwarding

> **TODO:** Document input-to-output forwarding (such as `Id`) and forwarding
> with `fwd`. See
> [proposal #8183](/proposals/p008183-first-class-extended-types-and-fwd.md#input-to-output-forwarding).

## References

-   Proposal
    [#8183: First-class extended types and `fwd`](/proposals/p008183-first-class-extended-types-and-fwd.md)
-   Proposal
    [#5434: `ref` parameters, arguments, returns and `val` returns](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md)
-   Proposal
    [#5545: Expression form basics](/proposals/p005545-expression-form-basics.md)
-   Proposal
    [#7254: Replace `:!` and `:?` with keywords and contextual defaults](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md)
-   Leads issue
    [#6932: Replace `:!` and `:?` with keywords and contextual defaults](https://github.com/carbon-language/carbon-lang/issues/6932)
