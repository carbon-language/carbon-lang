# Extended types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [`Core.ExtType` values](#coreexttype-values)
    -   [The `.type` property](#the-type-property)
-   [Extended type facet types](#extended-type-facet-types)
-   [References](#references)

<!-- tocstop -->

## TODO

> **TODO:** Fill in this design document (created in
> [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)) to
> reflect
> [proposal #8185: Extended type interfaces and named constraint extensions](/proposals/p008185-extended-type-interfaces-and-named-constraint-extensions.md).

For extended type interfaces and named extended type constraints, see
[Generics: Details](generics/details.md).

## `Core.ExtType` values

### The `.type` property

> **TODO:** Document the use of `<exttype_value>.type` in the orphan rule for
> `impl`s of extended type interfaces and interfaces with `Core.ExtType`
> parameters. See
> [proposal #8185](/proposals/p008185-extended-type-interfaces-and-named-constraint-extensions.md#extended-types-in-impls).

## Extended type facet types

> **TODO:** Document converting extended type facet types to ordinary type facet
> types by way of `Core.ValSelf`, `Core.RefSelf`, `Core.VarSelf`, named
> constraints, or `where ... impls`. See
> [proposal #8185](/proposals/p008185-extended-type-interfaces-and-named-constraint-extensions.md#converting-from-an-extended-type-facet-type-to-a-type-facet-type).

## References

-   Proposal
    [#8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
-   Proposal
    [#8185: Extended type interfaces and named constraint extensions](/proposals/p008185-extended-type-interfaces-and-named-constraint-extensions.md)
