# Conditionals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

`if` and `else` provide conditional execution of statements. Syntax is:

> `if (`_boolean expression_ `) {` _statements_ `}`
>
> [ `else if (` _boolean expression_ `) {` _statements_ `}` ] ...
>
> [ `else {` _statements_ `}` ]

Only one group of statements will execute:

-   When the first `if`'s boolean expression evaluates to true, its associated
    statements will execute.
-   When earlier boolean expressions evaluate to false and an `else if`'s
    boolean expression evaluates to true, its associated statements will
    execute.
    -   `... else if ...` is equivalent to `... else { if ... }`, but without
        visible nesting of braces.
-   When all boolean expressions evaluate to false, the `else`'s associated
    statements will execute.

When a boolean expression evaluates to true, no later boolean expressions will
evaluate.

Note that `else if` may be repeated.

For example:

```carbon
if (fruit.IsYellow()) {
  Print("Banana!");
} else if (fruit.IsOrange()) {
  Print("Orange!");
} else if (fruit.IsGreen()) {
  Print("Apple!");
} else {
  Print("Vegetable!");
}
fruit.Eat();
```

This code will:

-   Evaluate `fruit.IsYellow()`:
    -   When `True`, print `Banana!` and resume execution at `fruit.Eat()`.
    -   When `False`, evaluate `fruit.IsOrange()`:
        -   When `True`, print `Orange!` and resume execution at `fruit.Eat()`.
        -   When `False`, evaluate `fruit.IsGreen()`:
            -   When `True`, print `Orange!` and resume execution at
                `fruit.Eat()`.
            -   When `False`, print `Vegetable!` and resume execution at
                `fruit.Eat()`.

## Alternatives considered

-   [Optional braces](/proposals/p0623.md#optional-braces)
-   [Optional parentheses](/proposals/p0623.md#optional-parentheses)
-   [`elif`](/proposals/p0623.md#elif)

## References

-   Proposal
    [#285: `if` and `else`](https://github.com/carbon-language/carbon-lang/pull/285)
-   Proposal
    [#623: Require braces](https://github.com/carbon-language/carbon-lang/pull/623)
