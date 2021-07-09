# `return`

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

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. For example:

```carbon
fn Sum(a: Int, b: Int) -> Int {
  return a + b;
}
```

> TODO: Flesh out text (currently just overview)

## Relevant proposals

-   [Initial syntax](/proposals/p0415.md)
-   [`return` with no argument](/proposals/p0538.md)
