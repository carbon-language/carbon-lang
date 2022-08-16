# Unicode escape code length

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2040)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow zero digits](#allow-zero-digits)
    -   [Allow any number of hexadecimal characters](#allow-any-number-of-hexadecimal-characters)
    -   [Limiting to 6 digits versus 8](#limiting-to-6-digits-versus-8)

<!-- tocstop -->

## Abstract

The `\u{HHHH...}` can be an arbitrary length, potentially including `\u{}`.
Restrict to 1 to 8 characters.

## Problem

[Proposal #199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)
says "any number of hexadecimal characters" is valid for `\u{HHHH}`. This is
undesirable, because it means `\u{000 ... 000E9}` is a valid escape sequence,
for any number of `0` characters. Additionally, it's not clear if `\u{}` is
meant to be valid.

## Background

[Proposal #199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)
says:

> As in JavaScript, Rust, and Swift, Unicode code points can be expressed by
> number using `\u{10FFFF}` notation, which accepts any number of hexadecimal
> characters. Any numeric code point in the ranges
> 0<sub>16</sub>-D7FF<sub>16</sub> or E000<sub>16</sub>-10FFFF<sub>16</sub> can
> be expressed this way.

When it comes to the number of digits, the languages differ:

-   In [JavaScript](https://262.ecma-international.org/13.0/#prod-CodePoint),
    between 1 and 6 digits are supported, and it must be less than or equal to
    `10FFFF`.
-   In [Rust](https://doc.rust-lang.org/reference/tokens.html), between 1 and 6
    digits are supported.
-   In
    [Swift](https://docs.swift.org/swift-book/LanguageGuide/StringsAndCharacters.html),
    between 1 and 8 digits are supported.

Unicode's codespace is 0 to [`10FFFF`](https://unicode.org/glossary/#codespace).

## Proposal

The `\u{H...}` syntax is only valid for 1 to 8 unicode characters.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   This restriction does not affect the ability to write valid Unicode.
        Instead, it restricts the ability to write confusing or invalid unicode,
        which should make it easier to detect errors.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   Simplifies tooling by reducing the number of syntaxes that need to be
        supported, and allowing early failure on obviously invalid inputs.

## Alternatives considered

### Allow zero digits

We could allow `\u{}` as a version of `\u{0}`. However, as shorthand, it doesn't
save much and `\x00` is both equal length and clearer.

Rather than allowing this syntax, we prefer to disallow it for consistency with
other languages.

### Allow any number of hexadecimal characters

We could allow any number of digits in the `\u` escape. However, this has the
consequence of requiring parsing of escapes of completely arbitrary length.

This creates unnecessary complexity in the parser because we need to consider
what happens if the result is greater than 32 bits, significantly larger than
unicode's current `10FFFF` limit. One way to do this would be to store the
result in a 32-bit integer and keep parsing until the value goes above `10FFFF`,
then error as invalid if that's exceeded. This would allow an arbitrary number
of leading `0`'s to correctly parse.

It should make it easier to write a simple parser if we instead limit the number
of digits to a reasonable amount.

### Limiting to 6 digits versus 8

A limit of 6 digits offers a reasonable limit as the minimum needed to represent
Unicode's codespace. A limit of 8 digits offers a reasonable limit as a standard
4-byte value, and roughly matches UTF-32.

While it seems a weak advantage, this proposal leans towards 8.
