# Numeric type literal syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2015)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Non-goals](#non-goals)
-   [Details](#details)
    -   [Syntax](#syntax)
    -   [Usage](#usage)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [C++ LP64 convention](#c-lp64-convention)
    -   [Type name with length suffix](#type-name-with-length-suffix)
    -   [Uppercase suffixes](#uppercase-suffixes)
    -   [Additional bit sizes](#additional-bit-sizes)

<!-- tocstop -->

## Problem

We want to establish a syntax for fixed-size scalar number types. These types
include the two's complement signed integer, the unsigned integer, and the
floating-point number.

As these types are pervasive throughout the language, our goal here is to align
on a terse, convenient, yet understandable, and ergonomic syntax to the author.

## Background

For developer convenience, names are given to number types that map to native
machine register widths. These sizes typically include 8-bit, 16-bit, 32-bit,
64-bit, and, more recently, 128-bit widths.

For example, in [C++11+](https://en.cppreference.com/w/cpp/types/integer),
integer types such as `int8_t` (8-bit two's complement signed integer type) and
`uint16_t` (16-bit unsigned integer type) exist, among similar types for 32- and
64-bit values. Correspondingly, you have the `i8` and `u16`
([among others](https://doc.rust-lang.org/book/ch03-02-data-types.html#scalar-types))
scalar integer types in Rust. And in Swift, the `Int8` and `UInt16`
([among others](https://developer.apple.com/documentation/swift/uint8)) integer
value types.

In each case, the intent is to provide a clear and pragmatic syntax.

Additional discussion around this proposal's background can be found in
[#543](https://github.com/carbon-language/carbon-lang/issues/543).

## Proposal

We introduce a simple keyword-like syntax of `iN`, `uN`, and `fN` for two's
complement integers, unsigned integers, and floating-point numbers,
respectively. Where `N` can be a positive multiple of 8, including the common
power-of-two sizes (for example, `N = 8, 16, 32`). We think of these as "type
literals" just like `7` is a "numeric literal." This structure follows the
successful precedent set by Rust and LLVM development communities and
potentially saves 40% or more on characters required compared to other options
such as `IntN` (for example, `i16` versus `Int16`). While bit sizes greater than
128-bits will be well-supported, some operations like division will not be
available on these large sizes.

### Non-goals

-   This does not address any considerations around the `bool` type
-   This does not provide a formal plan for the shape or mapping of the
    underlying types
    ([#767 comments](https://github.com/carbon-language/carbon-lang/issues/767#issuecomment-1214153375))
-   This does not prescribe an official grammar for parsing these types
-   This proposal does not address other, non-multiple of 8 bit sizes, such as
    those used in a bit field

## Details

### Syntax

The syntax for a two's complement signed integer, the unsigned integer, and the
floating-point number corresponds to a lowercase 'i', 'u', or 'f' character,
respectively, indicating the type followed by a numeric value specifying the
width.

As a regular expression, this can be illustrated as:

```re
([iuf])([1-9][0-9]*)
```

Capture group 1 indicates either an 'i' for a two's complement signed integer
type, a 'u' for an unsigned integer type, or an 'f' for an
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) binary floating-point number
type. Capture group 2 specifies the width in bits. Note that this bit width is
restricted to a multiple of 8.

Examples of this syntax include:

-   `i16` - A 16-bit two's complement signed integer type
-   `u32` - A 32-bit unsigned integer type
-   `f64` - A 64-bit IEEE-754 binary floating-point number type

### Usage

```carbon
package sample api;

fn Sum(x: i32, y: i32) -> i32 {
  return x + y;
}

fn Main() -> i32 {
  return Sum(4, 2);
}
```

In the above example, `Sum` has parameters `x` and `y`, each of which is typed
as a 32-bit two's complement signed integer. `Main` then returns the output of
`Sum` as a 32-bit two's complement signed integer.

## Rationale

Following Carbon's goal to facilitate
["Code that is easy to read, understand, and write"](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
an explicit goal is to provide excellent ergonomics.

Highlighting relevant aspects of this from the project goals:

-   _Carbon should not use symbols that are difficult to type, see, or
    differentiate from similar symbols in commonly used contexts._
-   _Syntax should be easily parsed and scanned by any human in any development
    environment, not just a machine or a human aided by semantic hints from an
    IDE._
-   _Explicitness must be balanced against conciseness, as verbosity and
    ceremony add cognitive overhead for the reader, while explicitness reduces
    the amount of outside context the reader must have or assume._

The type system syntax must also complement Carbon's target for
["Performance-critical software"](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#performance-critical-software)

Specifically, there should be "No need for a lower level language."

-   _Developers should not need to leave the rules and structure of Carbon,
    whether to gain control over performance problems or to gain access to
    hardware facilities._

## Alternatives considered

As discussed in
[#543](https://github.com/carbon-language/carbon-lang/issues/543), four other
options were considered:

### C++ LP64 convention

Where `char` is the 8-bit type, `short` is the 16-bit type, `int` is the 32-bit
type, `long` is the 64-bit type.

Advantages:

-   The type name indicates its use to the reader
-   There is an existing precedent of this pattern in many programming
    languages, including C++
-   In the case of a typo, potentially better compiler checks versus an
    abbreviated form (for example, `i332`)

Disadvantages:

-   The type names themselves, as compared to the actual width and potentially
    use often can be arbitrary and confusing
-   The names themselves can be longer than the other syntax options
-   Some common C++ implementations use other models, which may create confusion
    when interoperating with C++ code. For example, Windows uses the LLP64
    model, where `long` is a 32-bit type, so Carbon code and C++ on Windows
    would have different and incompatible definitions for `long`.

### Type name with length suffix

Complete type name with a length-specifying suffix - `int8`, `int16`, `int32`,
`int64`, `uint32`, `float64`.

Advantages:

-   Are more explicit than an abbreviated version
-   Stand out against similar variable names, for example, `i8` versus `i = 8`)

Disadvantages:

-   Contain additional verbosity for potentially a non-significant amount of
    clarity
-   There are precedents from other communities (for example, Rust) that
    indicate authors enjoy a more compact syntax

### Uppercase suffixes

The suffix can be upper - `Int8`, `UInt8`, `Float16`; `I8`, `U8`, `F16`.

Advantages:

-   May help screen readers distinguish the type

Disadvantages:

-   Can be visually similar to other values, for example, `I8` versus `l8`
    (second is a lowercase L)

### Additional bit sizes

Support for additional bit sizes such as all bit sizes or common powers of two.

Advantages:

-   Adds flexibility and convenience for further use cases such as bit fields

Disadvantages:

-   May increase chances of typos without strong compiler guards, for example,
    `i32` versus `i22` versus `i23`
-   Variables such as `i1` and `i2` already exist in C++ code in practice
    ([example1](https://github.com/google/googletest/blob/main/googlemock/include/gmock/gmock-matchers.h#L878),
    [example2](https://chromium.googlesource.com/external/github.com/abseil/abseil-cpp/+/HEAD/absl/container/btree_test.cc#2772),
    [example3](https://sourcegraph.com/search?q=context:global+lang:c%2B%2B+%5Ei1%24+type:symbol&patternType=regexp&case=yes))
-   Adds complexity through additional size rules, for example, we can't support
    pointers to arbitrary bits
-   Adds confusion in syntactical overlap, for example, `i1`, `il`, `i18`, and
    `i18n`
