# Primitive types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Integers](#integers)
    -   [Floats](#floats)
    -   [BFloat16](#bfloat16)
-   [Open questions](#open-questions)
    -   [Primitive types as code vs built-in](#primitive-types-as-code-vs-built-in)
    -   [String view vs owning string](#string-view-vs-owning-string)
    -   [Syntax for wrapping operations](#syntax-for-wrapping-operations)
    -   [Non-power-of-two sizes](#non-power-of-two-sizes)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

These types are fundamental to the language as they aren't either formed from or
modifying other types. They also have semantics that are defined from first
principles rather than in terms of other operations. These will be made
available through the [prelude package](README.md#name-lookup-for-common-types).

-   `Bool` - a boolean type with two possible values: `True` and `False`.
-   `Int` and `UInt` - signed and unsigned 64-bit integer types.
    -   Standard sizes are available, both signed and unsigned, including
        `Int8`, `Int16`, `Int32`, `Int128`, and `Int256`.
    -   Overflow in either direction is an error.
-   `Float64` - a floating point type with semantics based on IEEE-754.
    -   Standard sizes are available, including `Float16`, `Float32`, and
        `Float128`.
    -   [`BFloat16`](primitive_types.md#bfloat16) is also provided.
-   `String` - a byte sequence treated as containing UTF-8 encoded text.
    -   `StringView` - a read-only reference to a byte sequence treated as
        containing UTF-8 encoded text.

### Integers

Integer types can be either signed or unsigned, much like in C++. Signed
integers are represented using 2's complement and notionally modeled as
unbounded natural numbers. Overflow in either direction is an error. That
includes unsigned integers, differing from C++. The default size for both is
64-bits: `Int` and `UInt`. Specific sizes are also available, for example:
`Int8`, `Int16`, `Int32`, `Int128`, `UInt256`. Arbitrary powers of two above `8`
are supported for both (although perhaps we'll want to avoid _huge_ values for
implementation simplicity).

### Floats

Floating point types are based on the binary floating point formats provided by
IEEE-754. `Float16`, `Float32`, `Float64` and `Float128` correspond exactly to
those sized IEEE-754 formats, and have the semantics defined by IEEE-754.

### BFloat16

Carbon also supports the
[`BFloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
format, a 16-bit truncation of a "binary32" IEEE-754 format floating point
number.

## Open questions

### Primitive types as code vs built-in

There are open questions about the extent to which these types should be defined
in Carbon code rather than special. Clearly they can't be directly implemented
w/o help, but it might still be useful to force the programmer-observed
interface to reside in code. However, this can cause difficulty with avoiding
the need to import things gratuitously.

### String view vs owning string

The right model of a string view versus an owning string is still very much
unsettled.

### Syntax for wrapping operations

Open question around allowing special syntax for wrapping operations (even on
signed types) and/or requiring such syntax for wrapping operations on unsigned
types.

### Non-power-of-two sizes

Supporting non-power-of-two sizes is likely needed to have a clean model for
bitfields, but requires more details to be worked out around memory access.
