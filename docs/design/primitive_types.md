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

-   `bool` - a boolean type with two possible values: `true` and `false`.
-   Signed and unsigned 64-bit integer types:
    -   Standard sizes are available, both signed and unsigned, including `i8`,
        `i16`, `i32`, `i64`, and `i128`, and `u8`, `u16`, `u32`, `u64`, and
        `u128`.
    -   Signed overflow in either direction is an error.
-   Floating points type with semantics based on IEEE-754.
    -   Standard sizes are available, including `f16`, `f32`, and `f64`.
    -   [`BFloat16`](primitive_types.md#bfloat16) is also provided.
-   `String` - a byte sequence treated as containing UTF-8 encoded text.
    -   `StringView` - a read-only reference to a byte sequence treated as
        containing UTF-8 encoded text.

The names `bool`, `true`, and `false` are keywords, and identifiers of the form
`i[0-9]*`, `u[0-9]*`, and `f[0-9*]` are _type literals_, resulting in the
corresponding type.

### Integers

Integer types can be either signed or unsigned, much like in C++. Signed
integers are represented using 2's complement and notionally modeled as
unbounded natural numbers. Signed overflow in either direction is an error.
Specific sizes are available, for example: `i8`, `u16`, `i32`, and `u128`.

There is an upper bound on the size of an integer, most likely initially set to
128 bits due to LLVM limitations.

### Floats

Floating point types are based on the binary floating point formats provided by
IEEE-754. `f16`, `f32`, `f64` and, if available, `f128` correspond exactly to
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
