# Literal expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Numeric type literals](#numeric-type-literals)
    -   [Usage](#usage)
    -   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

This document is intended to cover all literal expressions, excluding numeric,
floats and strings, which are covered in the
[Lexical Conventions](../lexical_conventions/README.md) section. For now, the
document explains the numeric type literals.

## Numeric type literals

Carbon has a simple keyword-like syntax of `iN`, `uN`, and `fN` for two's
complement integers, unsigned integers, and
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) floating-point numbers,
respectively. Here, `N` can be a positive multiple of 8, including the common
power-of-two sizes (for example, `N = 8, 16, 32`).

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

### Alternatives considered

-   [C++ LP64 convention](/proposals/p2015.md#c-lp64-convention)
-   [Type name with length suffix](/proposals/p2015.md#type-name-with-length-suffix)
-   [Uppercase suffixes](/proposals/p2015.md#uppercase-suffixes)
-   [Additional bit sizes](/proposals/p2015.md#additional-bit-sizes)

## References

-   Issue
    [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
-   Proposal
    [#2015: Numeric type literal syntax](https://github.com/carbon-language/carbon-lang/pull/2015)
