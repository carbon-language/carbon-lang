# Numeric literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Integer literals](#integer-literals)
    -   [Real number literals](#real-number-literals)
        -   [Ties](#ties)
    -   [Digit separators](#digit-separators)
-   [Caveats](#caveats)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

The following syntaxes are supported:

-   Integer literals
    -   `12345` (decimal)
    -   `0x1FE` (hexadecimal)
    -   `0b1010` (binary)
-   Real number literals
    -   `123.456` (digits on both sides of the `.`)
    -   `123.456e789` (optional `+` or `-` after the `e`)
    -   `0x1.2p123` (optional `+` or `-` after the `p`)
-   Digit separators (`_`) may be used, but only in conventional locations

Note that real number literals always contain a `.` with digits on both sides,
and integer literals never contain a `.`.

Literals are case-sensitive. Unlike in C++, literals do not have a suffix to
indicate their type.

## Details

### Integer literals

Decimal integers are written as a non-zero decimal digit followed by zero or
more additional decimal digits, or as a single `0`.

Integers in other bases are written as a `0` followed by a base specifier
character, followed by a sequence of digits in the corresponding base. The
available base specifiers and corresponding bases are:

| Base specifier | Base | Digits                   |
| -------------- | ---- | ------------------------ |
| `b`            | 2    | `0` and `1`              |
| `x`            | 16   | `0` ... `9`, `A` ... `F` |

The above table is case-sensitive. For example, `0b1` and `0x1A` are valid, and
`0B1`, `0X1A`, and `0x1a` are invalid.

A zero at the start of a literal can never be followed by another digit: either
the literal is `0`, the `0` begins a base specifier, or the next character is a
decimal point (see below). No support is provided for octal literals, and any C
or C++ octal literal (other than `0`) is invalid in Carbon.

### Real number literals

Real numbers are written as a decimal or hexadecimal integer followed by a
period (`.`) followed by a sequence of one or more decimal or hexadecimal
digits, respectively. A digit is required on each side of the period. `0.` and
`.3` are both invalid.

A real number can be followed by an exponent character, an optional `+` or `-`
(defaulting to `+` if absent), and a character sequence matching the grammar of
a decimal integer with some value _N_. For a decimal real number, the exponent
character is `e`, and the effect is to multiply the given value by
10<sup>&plusmn;_N_</sup>. For a hexadecimal real number, the exponent character
is `p`, and the effect is to multiply the given value by
2<sup>&plusmn;_N_</sup>. The exponent suffix is optional for both decimal and
hexadecimal real numbers.

Note that a decimal integer followed by `e` is not a real number literal. For
example, `3e10` is not a valid literal.

When a real number literal is interpreted as a value of a real number type, its
value is the representable real number closest to the value of the literal. In
the case of a [tie](#ties), the conversion to the real number type is invalid.

The decimal real number syntax allows for any decimal fraction to be expressed
-- that is, any number of the form _a_ x 10<sup>-_b_</sup>, where _a_ is an
integer and _b_ is a non-negative integer. Because the decimal fractions are
dense in the reals and the set of values of the real number type is assumed to
be discrete, every value of the real number type can be expressed as a real
number literal. However, for certain applications, directly expressing the
intended real number representation may be more convenient than producing a
decimal equivalent that is known to convert to the intended value. Hexadecimal
real number literals are provided in order to permit values of binary floating
or fixed point real number types to be expressed directly.

#### Ties

As described above, a real number literal that lies exactly between two
representable values for its target type is invalid. Such ties are extremely
unlikely to occur by accident: for example, when interpreting a literal as
`Float64`, `1.` would need to be followed by exactly 53 decimal digits (followed
by zero or more `0`s) to land exactly half-way between two representable values,
and the probability of `1.` followed by a random 53-digit sequence resulting in
such a tie is one in 5<sup>53</sup>, or about
0.000000000000000000000000000000000009%. For `Float32`, it's about
0.000000000000001%, and even for a typical `Float16` implementation with 10
fractional bits, it's around 0.00001%.

Ties are much easier to express as hexadecimal floating-point literals: for
example, `0x1.0000_0000_0000_08p+0` is exactly half way between `1.0` and the
smallest `Float64` value greater than `1.0`, which is `0x1.0000_0000_0000_1p+0`.

Whether written in decimal or hexadecimal, a tie provides very strong evidence
that the developer intended to express a precise floating-point value, and
provided one bit too much precision (or one bit too little, depending on whether
they expected some rounding to occur), so rejecting the literal is preferred
over making an arbitrary choice between the two possible values.

### Digit separators

If digit separators (`_`) are included in literals, they must meet the
respective condition:

-   For decimal integers, the digit separators shall occur every three digits
    starting from the right. For example, `2_147_483_648`.
-   For hexadecimal integers, the digit separators shall occur every four digits
    starting from the right. For example, `0x7FFF_FFFF`.
-   For real number literals, digit separators can appear in the decimal and
    hexadecimal integer portions (prior to the period and after the optional `e`
    or mandatory `p`) as described in the previous bullets. For example,
    `2_147.483648e12_345` or `0x1_00CA.FEF00Dp+24`
-   For binary literals, digit separators can appear between any two digits. For
    example, `0b1_000_101_11`.

## Caveats

The proposal provides a syntax that is deliberately close to that used both by
C++ and many other languages, so it should feel familiar to developers. However,
it selects a reasonably minimal subset of the syntaxes. This minimal approach
provides benefits directly in line with the goal that Carbon code should be
[easy to read, understand, and write](../docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):

-   Reduces unnecessary choices for programmers.
-   Simplifies the syntax rules of the language.
-   Improves consistency of written Carbon code.

That said, it still provides sufficient variations to address important use
cases for the goal of not leaving room for a lower level language:

-   Hexadecimal and binary integer literals.
-   Scientific notation floating point literals.
-   Hexadecimal (scientific) floating point literals.

## Alternatives considered

-   [Integer bases](/proposals/p0143.md#integer-bases)
    -   [Octal literals](/proposals/p0143.md#octal-literals)
    -   [Decimal literals](/proposals/p0143.md#decimal-literals)
    -   [Case sensitivity](/proposals/p0143.md#case-sensitivity)
-   [Real number syntax](/proposals/p0143.md#real-number-syntax)
-   [Digit separator syntax](/proposals/p0143.md#digit-separator-syntax)

## References

-   Proposal
    [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
