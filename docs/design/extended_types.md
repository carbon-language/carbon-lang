# Extended types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [`fwd` bindings, arguments, and returns](#fwd-bindings-arguments-and-returns)
    -   [`fwd` call arguments and operator operands](#fwd-call-arguments-and-operator-operands)
-   [Extended type conversion constraints](#extended-type-conversion-constraints)
-   [References](#references)

<!-- tocstop -->

## TODO

> **TODO:** Fill in this design document (created in
> [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)) to
> reflect
> [proposal #8188: Extended-type operator, conversion, and member binding interfaces](/proposals/p008188-extended-type-operator-conversion-and-member-binding-interfaces.md).

## `fwd` bindings, arguments, and returns

### `fwd` call arguments and operator operands

> **TODO:** Document marking operator operands that have a symbolic extended
> type with `fwd` (for example `(fwd self) == (fwd u)`). See
> [proposal #8188](/proposals/p008188-extended-type-operator-conversion-and-member-binding-interfaces.md#marking-ref-and-fwd-on-operator-operands).

## Extended type conversion constraints

> **TODO:** Document how `Core.ConvertsInto` and `Core.ConvertsFrom` (from
> [proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186))
> differ from `Core.ImplicitAsPrimitive`. See
> [proposal #8188](/proposals/p008188-extended-type-operator-conversion-and-member-binding-interfaces.md#primitive-conversion-interfaces).

## References

-   Proposal
    [#8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
-   Proposal
    [#8186: Interface variance (`implicit_into` and `implicit_from`)](https://github.com/carbon-language/carbon-lang/pull/8186)
-   Proposal
    [#8188: Extended-type operator, conversion, and member binding interfaces](/proposals/p008188-extended-type-operator-conversion-and-member-binding-interfaces.md)
