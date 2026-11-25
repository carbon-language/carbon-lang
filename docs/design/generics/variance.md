# Generics: Interface variance

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
-   [`implicit_into` variance (contravariance)](#implicit_into-variance-contravariance)
-   [`implicit_from` variance (covariance)](#implicit_from-variance-covariance)
-   [Coherence and `impl` lookup](#coherence-and-impl-lookup)
    -   [Type structures and prioritization](#type-structures-and-prioritization)
    -   [`anchor` and `exact` modifiers](#anchor-and-exact-modifiers)
    -   [`impl` lookup algorithm for variant parameters](#impl-lookup-algorithm-for-variant-parameters)
-   [Facets create adapters](#facets-create-adapters)
-   [Implied constraints](#implied-constraints)
-   [References](#references)

<!-- tocstop -->

## TODO

> **TODO:** Fill in this design document to reflect
> [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md).

This is a stub document for interface parameter and associated constant variance
(`implicit_into` and `implicit_from`), which replaces the `like` mechanism in
[Generics: Details](details.md#like-operator-for-implicit-conversions).

## Overview

> **TODO:** Summarize how interfaces can declare variant `Core.ExtType`
> parameters and associated constants to allow implicit conversions during
> `impl` lookup, and how this replaces `like` on `impl` declarations. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#implicit_into-interface-parameter-variance).

## `implicit_into` variance (contravariance)

> **TODO:** Document marking `Core.ExtType` interface parameters (including
> `Self`) with `implicit_into`, notional `impl`s parameterized by
> [`Core.ConvertsInto`](/docs/design/extended_types.md#extended-type-conversion-constraints),
> allowed contravariant positions within an interface definition, and invariance
> of class parameters. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#implicit_into-interface-parameter-variance).

## `implicit_from` variance (covariance)

> **TODO:** Document marking `Core.ExtType` interface parameters and associated
> constants with `implicit_from`, notional `impl`s parameterized by
> [`Core.ConvertsFrom`](/docs/design/extended_types.md#extended-type-conversion-constraints),
> and allowed covariant positions within an interface definition. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#implicit_from-variance).

## Coherence and `impl` lookup

### Type structures and prioritization

> **TODO:** Document how a single `impl` of an interface with variant parameters
> corresponds to multiple type structures using `(type, extended type)` pairs,
> how matching type versus matching full extended type affects prioritization
> (such as preferring a `ref` `impl` over a `val` `impl` for reference queries),
> and when `match_first` blocks are required. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#implicit_into-interface-parameter-variance).

### `anchor` and `exact` modifiers

> **TODO:** Document `anchor` and `exact` modifiers on variant parameters in
> `impl` declarations, `anchor` on `implicit_into` interface parameters, how
> they interact with the [orphan rule](details.md#orphan-rule), and how they
> interact with [named constraints](details.md#named-constraints). See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#modifiers-for-coherence),
> [interaction with named constraints](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#interaction-between-anchor-exact-and-named-constraints),
> and
> [coherence modifier examples](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#coherence-modifier-examples).

### `impl` lookup algorithm for variant parameters

> **TODO:** Document the updates to `impl` lookup for interfaces with variant
> parameters (initial type-structure filtering, `anchor` matching requirement,
> and per-query type-structure prioritization). See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#impl-lookup-for-an-interface-with-variant-parameters).

## Facets create adapters

> **TODO:** Document why forming a facet of an interface with variant parameters
> creates an [adapter](details.md#adapting-types) type rather than preserving
> the original type, how this preserves [coherence](terminology.md#coherence),
> what observable differences arise (conversions, accepted input types, ignoring
> `anchor`/`exact` and other `impl`s), and consistency with `template`
> parameters. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#facets-create-adapters).

## Implied constraints

> **TODO:** Document why [implied constraints](details.md#implied-constraints)
> do not apply to interfaces with variant parameters. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#implied-constraints).

## References

-   Proposal
    [#8186: Interface variance (`implicit_into` and `implicit_from`)](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md)
-   Proposal
    [#5337: Interface extension and `final impl` update](/proposals/p005337-interface-extension-and-final-impl-update.md)
-   Proposal
    [#1144: Generic details 11: operator overloading](/proposals/p001144-generic-details-11-operator-overloading.md)
