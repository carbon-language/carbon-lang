# Allow ties in floating literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/866)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

Proposal [#143](https://github.com/carbon-language/carbon-lang/pull/143)
suggested that we do not allow ties in floating-point literals. That is, given a
literal whose value lies exactly half way between two representable values, we
should reject rather than arbitrarily picking one of the two possibilities.

However, the [statistical argument](p0143.md#ties) presented in that proposal
misses an important fact: the distribution of the values that are exactly half
way between representable values includes several values of the form A x
10<sup>B</sup>, where A and B are small integers.

For example, the current rule rejects this very reasonable looking code:

```
var v: f32 = 9.0e9;
```

... because 9 x 10<sup>9</sup> lies exactly half way between the nearest two
representable values of type `f32`, namely 8999999488 and 9000000512. Similar
examples exist for larger floating point types:

```
// Error, half way between two exactly representable values.
var w: f64 = 5.0e22;
```

We would also reject an attempted workaround such as:

```
var v: f32 = 5 * 1.0e22;
```

... because the literal arithmetic would be performed exactly, resulting in the
same tie. A workaround such as

```
var v1: f32 = 5.0e22 + 1.0;
var v2: f32 = 5.0e22 - 1.0;
```

... to request rounding upwards and downwards, respectively, would work.
However, these seem cumbersome and burden the Carbon developer with
floating-point minutiae about which they very likely do not care.

## Background

For background on the ties-to-even rounding rule, see
[this Wikipedia article](https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).
The ties-to-even rule is the default rounding mode specified by ISO 60559 /
IEEE 754.

## Proposal

Instead of rejecting exact ties, we use the default IEEE floating point rounding
mode: we round to even.

## Details

See design changes.

## Rationale based on Carbon's goals

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   This improves the ease of both reading and writing floating-point
        literals that would result in ties.
    -   This improves the language consistency, by performing the same rounding
        when converting literals as is performed by default when converting
        runtime values.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   It is unlikely that making an arbitrary but consistent rounding choice
        will harm safety or program correctness.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This rule, likely because it is the IEEE default rounding mode, already
        appears to be used by major C++ compilers such as Clang, GCC, MSVC, and
        ICC.

## Alternatives considered

We could round to even only for decimal floating-point literals, and still use
the rule that ties are rejected for hexadecimal floating point. In the latter
case, a tie means that too many digits were specified, and the trailing digits
were exactly `80000...`.

However, because we support arithmetic on literals, forming other literals, this
would mean that whether a literal was originally written in hexadecimal would
form part of its value and thereby part of its type. There would also be
problems with literals produced by arithmetic. The complexity involved here far
outweighs any perceived benefit of diagnosing mistyped literals.
