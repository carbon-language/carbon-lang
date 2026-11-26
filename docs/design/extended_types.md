# Extended types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Extended type conversion constraints](#extended-type-conversion-constraints)
-   [References](#references)

<!-- tocstop -->

## TODO

> **TODO:** Fill in this design document (created in
> [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)) to
> reflect
> [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md).

For extended type interfaces, named extended type constraints, and interface
variance, see [Generics: Details](generics/details.md) and
[Generics: Interface variance](generics/variance.md).

## Extended type conversion constraints

> **TODO:** Document `Core.ConvertsInto(PET: ExtType)` and
> `Core.ConvertsFrom(RET: ExtType)`, their definitions in terms of forwarding
> parameters (`F(fwd ~y)`) and returns (`return F();`), their non-implementable
> constraint nature, and how they differ from implicit type conversions and
> extended type facet type conversions. See
> [proposal #8186](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#coreconvertsinto)
> and
> [`Core.ConvertsFrom`](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md#coreconvertsfrom).

## References

-   Proposal
    [#8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
-   Proposal
    [#8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185)
-   Proposal
    [#8186: Interface variance (`implicit_into` and `implicit_from`)](/proposals/p008186-interface-variance-implicit-into-and-implicit-from.md)
