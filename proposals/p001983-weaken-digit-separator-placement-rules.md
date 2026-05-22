# Weaken digit separator placement rules

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1983)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [3-digit decimal groupings](#3-digit-decimal-groupings)
    -   [2-digit or 4-digit hexadecimal digit groupings](#2-digit-or-4-digit-hexadecimal-digit-groupings)
    -   [Disallow digit separators in fractions](#disallow-digit-separators-in-fractions)

<!-- tocstop -->

## Abstract

[Proposal #143: Numeric literals](/proposals/p0143.md) added digit separators
with strict rules for placement. It missed some use-cases. In order to address
this, remove placement rules for numeric literals.

## Problem

Digit separator placement rules are too strict:

-   For integers, while the original proposal mentioned Indian digit groupings,
    Chinese, Japanese, and Korean cultures use 4-digit groupings. This oversight
    is likely due to the description on
    [wikipedia](https://en.wikipedia.org/wiki/Decimal_separator#Digit_grouping):
    "but the delimiter commonly separates every three digits".
-   There are microformats where different placement rules may be desirable. See
    [alternatives](#alternatives-considered) for more specific examples.

## Background

[Proposal #143: Numeric literals](/proposals/p0143.md) added digit separators
with strict rules for placement:

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

This was asked on
[Issue #1485: Reconsider digit separators](https://github.com/carbon-language/carbon-lang/issues/1485),
where a proposal was requested.

## Proposal

Switch to simple rules: a single digit separator can appear between any two
digits. For example:

-   Decimal literals: `1_2_3`
-   Hexadecimal literals: `0xA_B_C`
-   Real literals: `1_2.3_4e5_6`
-   Binary literals are unaffected.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Removing restrictions on decimal literals provides flexibility for
        developers who don't use 3-digit groupings.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   This allows for mistakes in groupings, which is undesirable. However,
        it's useful for microformats to be able to provide their own particular
        groupings.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Reduces migration hurdles for C++ code by providing a rule that's more
        consistent with the C++ rule.

## Alternatives considered

### 3-digit decimal groupings

For decimal separators, we could enforce 3-digit groupings if digit separators
are used at all.

Advantages:

-   More enforcement of digit groupings can prevent bugs and misreadings
    resulting from accidentally incorrect groupings.

Disadvantages:

-   Indian, Chinese, Japanese, and Korean digit grouping conventions don't fit
    under this rule.
    -   These are worth considering similar to how Carbon allows
        [Unicode identifiers](/docs/design/lexical_conventions/words.md):
        although keywords are in English, we can be more flexible for both
        identifiers and literals.
-   Microformats could also pose issues. For example, microseconds (`1_00000`),
    dates (`01_12_1983`), credit cards (`1234_5678_9012_3456`), or identity
    numbers such as US social security numbers (`123_45_6789`).
    -   Since the original proposal, more cases have been raised.

Note that any regular grouping rule can present similar issues for Indian digit
grouping conventions.

[Proposal #143: Numeric literals](/proposals/p0143.md) chose 3-digit decimal
groupings.

Given there are overall advantages to not enforcing regular digit conventions,
including for hex digits, it seems unnecessary to conform to the currently
established decimal conventions. While this removes the enforcement, the
resulting accidents are considered a reasonable risk.

In theory a code linter could be told to prefer certain formats with options to
change behavior, although that may remain too low benefit to implement.

### 2-digit or 4-digit hexadecimal digit groupings

Hexadecimal digit groupings could be enforced along two axes:

-   Every 2 (`F_FF`) or 4 (`F_FFFF`) characters, corresponding to one or two
    bytes respectively.
-   Regular or irregular placement. For example, if a digit separator is placed
    every byte (every other digit), we could require regular placement every
    byte as in `FF_FF_FF_FF`, or irregular as in `FF_FFFFF_FF` which skips one
    placement.

[Proposal #143: Numeric literals](/proposals/p0143.md) chose 4-digit with
regular placement.

Advantages:

-   More enforcement of digit groupings can prevent bugs and misreadings
    resulting from accidentally incorrect groupings.

Disadvantages:

-   Again, microformats become an issue.
    -   As noted in background, MAC addresses are typically in byte pairs
        (`00_B0_D0_63_C2_26`) and UUIDs have particular groupings
        (`123E4567_E89B_12D3_A456_426614174000`).
-   If all of the other literal formats (decimal, real, and binary) allow
    arbitrary digit separator placement, enforcing placement in hexadecimal
    numbers would be an outlier.

While it may be that enforcing 2 character (one byte) groupings with irregular
placement would prevent sufficient errors versus the developer inconvenience
risks, it risks becoming an outlier and for only very loose enforcement.

Similar to the preceding conclusion, leaving these issues to code linters may be
best.

### Disallow digit separators in fractions

[Proposal #143: Numeric literals](/proposals/p0143.md) appears to disallow digit
separators in fractions. That is, in `1.2345`, `1.23_45` is disallowed. This
proposal changes that.

Advantages:

-   Reduces the chance of confusion resulting from mistaking a `.` for another
    `_` separator, as in `1_234_567.890_123_456`.

Disadvantages:

-   While sometimes commas aren't written for digit separators, it's notable
    that SI advice is to use digit spacing instead
    ([Digit spacing, rule #16](https://physics.nist.gov/cuu/Units/checklist.html)).
    As a consequence, it's likely common to want some kind of digit separator
    after the decimal point.
-   Would create an inconsistency in placement rules.

We'll allow digit separators in fractions.
