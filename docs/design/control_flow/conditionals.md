# Control flow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Relevant proposals](#relevant-proposals)

<!-- tocstop -->

## Overview

`if` and `else` provide conditional execution of statements. For example:

```carbon
if (fruit.IsYellow()) {
  Print("Banana!");
} else if (fruit.IsOrange()) {
  Print("Orange!");
} else {
  Print("Vegetable!");
}
```

This code will:

-   Print `Banana!` if `fruit.IsYellow()` is `True`.
-   Print `Orange!` if `fruit.IsYellow()` is `False` and `fruit.IsOrange()` is
    `True`.
-   Print `Vegetable!` if both of the above return `False`.

> TODO: Flesh out text (currently just overview)

## Relevant proposals

Most discussion of design choices and alternatives may be found in relevant
proposals.

-   [`if` and `else`](/proposals/p0285.md)
