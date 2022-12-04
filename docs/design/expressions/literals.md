# Numeric Type Literal Semantics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Usage](#usage)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon's introduce a simple keyword-like syntax of `iN`, `uN`, and `fN` for
two's complement integers, unsigned integers, and
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) floating-point numbers,
respectively. Where `N` can be a positive multiple of 8, including the common
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

## Alternatives considered

-   [Alternatives considered](/proposals/p2015.md#alternatives-considered)

## References

-   [Rationale](/proposals/p2015.md#rationale)
-   Proposal
    [#2015: Numeric type literal syntax](https://github.com/carbon-language/carbon-lang/pull/2015)
-   Issue
    [#1998: Make proposal for numeric type literal syntax](https://github.com/carbon-language/carbon-lang/issues/1998#issuecomment-1212644291)
