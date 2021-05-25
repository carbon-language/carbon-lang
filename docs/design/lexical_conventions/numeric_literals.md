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
-   [Alternatives](#alternatives)
    -   [Integer bases](#integer-bases)
        -   [Octal literals](#octal-literals)
        -   [Decimal literals](#decimal-literals)
        -   [Case sensitivity](#case-sensitivity)
    -   [Real number syntax](#real-number-syntax)
    -   [Digit separator syntax](#digit-separator-syntax)
    -   [Digit separator positioning](#digit-separator-positioning)

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

## Alternatives

### Integer bases

#### Octal literals

No support is proposed for octal literals. In practice, their appearance in C
and C++ code in a sample corpus consisted of (in decreasing order of commonality
and excluding `0` literals):

-   file permissions,
-   cases where decimal was clearly intended (`CivilDay(2020, 04, 01)`), and
-   (in _distant_ third place) anything else.

The number of intentional uses of octal literals, other than in file
permissions, was negligible. We considered the following alternatives:

**Alternative 1:** Follow C and C++, and use `0` as the base prefix for octal.

Advantages:

-   More similar to C++ and other languages.

Disadvantages:

-   Subtle and error-prone rule: for example, left-padding with zeroes for
    alignment changes the meaning of literals.

**Alternative 2:** Use `0o` as the base prefix for octal.

Advantages:

-   Unlikely to be misinterpreted as decimal.
-   Follows several other languages (for example, Python).

Disadvantages:

-   Additional language complexity.

If we decide we want to introduce octal literals at a later date, use of
alternative 2 is suggested.

#### Decimal literals

**We could permit leading `0`s in decimal integers (and in floating-point
numbers).**

Advantages:

-   We would allow leading `0`s to be used to align columns of numbers.

Disadvantages:

-   The same literal could be valid but have a different value in C++ and
    Carbon.

**We could add an (optional) base specifier `0d` for decimal integers.**

Advantages:

-   Uniform treatment of all bases. Left-padding with `0` could be achieved by
    using `0d000123`.

Disadvantages:

-   No evidence of need for this functionality.

**We could permit an `e` in decimal literals to express large powers of 10.**

Advantages:

-   Many uses of (eg) `1e6` in our sample C++ corpus intend to form an integer
    literal instead of a floating-point literal.

Disadvantages:

-   Would violate the expectations of many C++ programmers used to `e`
    indicating a floating-point constant.

#### Case sensitivity

**We could make base specifiers case-insensitive.**

Advantages:

-   More similar to C++.

Disadvantages:

-   `0B1` is easily mistaken for `081`
-   `0B1` can be confused with `0xB1`
-   `0O17` is easily mistaken for `0017`
-   Allowing more than one way to write literals will lead to style divergence.

**We could make the digit sequence in hexadecimal integers case-insensitive.**

Advantages:

-   More similar to C++.
-   Some developers will be more comfortable writing hexadecimal digits in
    lowercase. Some tools, such as `md5`, will print lowercase.

Disadvantages:

-   Allowing more than one way to write literals will lead to style divergence.
-   Lowercase hexadecimal digits are less visually distinct from the `x` base
    specifier (for example, the digit sequence is more visually distinct in
    `0xAC` than in `0xac`).

**We could require the digit sequence in hexadecimal integers to be written
using lowercase letters `a`..`f`.**

Advantages:

-   Some developers will be more comfortable writing hexadecimal digits in
    lowercase. Some tools, such as `md5`, will print lowercase.
-   `B` and `D` are more likely to be confused with `8` and `0` than `b` and `d`
    are.

Disadvantages:

-   Some developers will be more comfortable writing hexadecimal digits in
    uppercase. Some tools will print uppercase.
-   Lowercase hexadecimal digits are less visually distinct from the `x` base
    specifier (for example, the digit sequence is more visually distinct in
    `0xAC` than in `0xac`).

### Real number syntax

**We could allow real numbers with no digits on one side of the period (`3.` or
`.5`).**

Advantages:

-   More similar to C++.
-   Allows numbers to be expressed more tersely.

Disadvantages:

-   Gives meaning to `tup.0` syntax that may be useful for indexing tuples.
-   Gives meaning to `0.ToString()` syntax that may be useful for performing
    member access on literals.
-   May harm readability by making the difference between an integer literal and
    a real number literal less significant.
-   Allowing more than one way to write literals will lead to style divergence.

See also the section on
[floating-point literals](https://google.github.io/styleguide/cppguide.html#Floating_Literals)
in the Google style guide, which argues for the same rule.

**We could allow a real number with no `e` or `p` to omit a period (`1e100`).**

Advantages:

-   More similar to C++.
-   Allows numbers to be expressed more tersely.

Disadvantages:

-   Assuming that such numbers are integers rather than real numbers is a common
    error in C++.

**We could allow the `e` or `p` to be written in uppercase.**

Advantages:

-   More similar to C++.
-   Most calculators use `E`, to avoid confusion with the constant `e`.

Disadvantages:

-   Allowing more than one way to write literals will lead to style divergence.
-   `E` may be confused with a hexadecimal digit.

**We could require a `p` in a hexadecimal real number literal.**

Advantages:

-   More similar to C++.
-   When explicitly writing a bit-pattern for a floating-point type, it's
    reasonable to always include the exponent value.

Disadvantages:

-   Less consistent.
-   Makes hexadecimal floating-point values even more expert-only.

**We could arbitrarily pick one of the two values when a real number is exactly
half-way between two representable values.**

Advantages:

-   More similar to C++.
-   Would accept more cases, and it's likely that either of the two possible
    values would be acceptable in practice.

Disadvantages:

-   Would either need to specify which option is chosen or, following C++,
    accept that programs using such literals have non-portable semantics.
-   Numbers specified to the exact level of precision required to form a tie are
    a strong signal that the programmer intended to specify a particular value.

### Digit separator syntax

We considered the following characters as digit separators:

**Status quo:** `_` as a digit separator.

Advantages:

-   Follows convention of C#, Java, JavaScript, Python, D, Ruby, Rust, Swift,
    ...
-   Culturally agnostic, because it doesn't match any common human writing
    convention.

Disadvantages:

-   Underscore is not used as a digit grouping separator in any common human
    writing convention.

**Alternative 1:** `'` as a digit separator.

Advantages:

-   Follows C++ syntax.
-   Used in several (mostly European) writing conventions.

Disadvantages:

-   `'` is also likely to be used to introduce character literals.

**Alternative 2:** `,` as a digit separator.

Advantages:

-   More similar to how numbers are written in English text and many other
    cultures.

Disadvantages:

-   Commas are expected to widely be used in Carbon programs for other purposes,
    where there may be digits on both sides of the comma. For example, there
    could be readability problems if `f(1, 234)` called `f` with two arguments
    but `f(1,234)` called `f` with a single argument.
-   Comma is interpreted as a decimal point in the conventions of many cultures.
-   Unprecedented in common programming languages.

**Alternative 3:** whitespace as a digit separator.

Advantages:

-   Used and understood by many cultures.
-   Never interpreted as a decimal point instead of a grouping separator.
-   Also usable to the right of a decimal point.

Disadvantages:

-   Omitted separators in lists of numbers may result in distinct numbers being
    spliced together. For example, `f(1, 23, 4 567)` may be interpreted as three
    separate numerical arguments instead of four arguments with a missing comma.
-   Unprecedented in other programming languages.

**Alternative 4:** `.` as digit separator, `,` as decimal point.

Advantages:

-   More familiar to cultures that write numbers this way.

Disadvantages:

-   As with `,` as a digit separator, `,` as a decimal point is problematic.
-   This usage is unfamiliar and would be surprising to programmers; programmers
    from cultures where `,` is the decimal point in regular writing are likely
    already accustomed to using `.` as the decimal point in programming
    environments, and the converse is not true.

**Alternative 5:** No digit separator syntax.

Advantages:

-   Simpler language rules.
-   More consistent source syntax, as there is no choice as to whether to use
    digit separators or not.

Disadvantages:

-   Harms the readability of long literals.

### Digit separator positioning

**Alternative 1:** allow any digit groupings (for example, `123_4567_89`).

Advantages:

-   Simpler, more flexible rule, that may allow some groupings that are
    conventional in a specific domain. For example, `var Date d = 01_12_1983;`,
    or `var Int64 time_in_microseconds = 123456_000000;`.
-   Culturally agnostic. For example, the Indian convention for digit separators
    would group the last three digits, and then every two digits before that
    (1,23,45,678 could be written `1_23_45_678`).

Disadvantages:

-   Less self-checking that numeric literals are interpreted the way that the
    author intends.

**Alternative 2:** as above, but additionally require binary digits to be
grouped in 4s.

Advantages:

-   More enforcement that digit grouping is conventional.

Disadvantages:

-   No clear, established rule for how to group binary digits. In some cases, 8
    digit groups may be more conventional.
-   When used to express literals involving bit-fields, arbitrary grouping may
    be desirable. For example:

    ```carbon
    var Float32 flt_max =
      BitCast(Float32, 0b0_11111110_11111111111111111111111);
    ```

**Alternative 3:** allow any regular grouping.

Advantages:

-   Can be applied uniformly to all bases.

Disadvantages:

-   Provides no assistance for decimal numbers with a single digit separator.
-   Does not allow binary literals to express an intent to initialize irregular
    bit-fields.
