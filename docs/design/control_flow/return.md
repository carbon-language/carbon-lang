# `return`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Returning empty tuples](#returning-empty-tuples)
    -   [`returned var`](#returned-var)
-   [Relevant proposals](#relevant-proposals)

<!-- tocstop -->

## Overview

The `return` statement ends the flow of execution within a
[function](../functions.md), returning execution to the caller. Its syntax is:

`return [<expression>];`

If the function returns a value to the caller, that value is provided by an
expression in the return statement. For example:

```carbon
fn Sum(a: Int, b: Int) -> Int {
  return a + b;
}
```

When a return type is specified, a function must _always_ `return` before normal
function completion. In other words, `fn DoNothing() -> Int {}` would be invalid
because the function will complete without returning a value.

### Returning empty tuples

An empty tuple `()` is special, and similar to C++'s `void` returns. When a
function has no specified return type, its return type is implicitly `()`.
`return` must not have an expression argument in this case. For example:

```carbon
// No return type is specified, so this returns `()` implicitly.
fn MaybeDraw(should_draw: bool) {
  if (!should_draw) {
    // No expression is passed to `return`.
    return;
  }
  ActuallyDraw();
}
```

When `-> ()` is specified, the return expression is required. Omitting `-> ()`
is encouraged, but is supported for generalized code structures, including
[templates](../templates.md). In order to require consistency, `return;` is
invalid in this case. For example:

```carbon
// `-> ()` defines an explicit return value.
fn MaybeDraw(should_draw: bool) -> () {
  if (!should_draw) {
    // As a consequence, a return value must be passed.
    return ();
  }
  ActuallyDraw();
}
```

### `returned var`

Variables declared with the `returned` statement must be returned using
`return var`, rather than specifying the identifier. When a `returned var` is in
scope, other expressions must not be passed to `return`; only `return var` is
allowed. For example:

> TODO: Document `returned` in variables.md, link there -- waiting on #618

```carbon
fn MakeCircle(radius: Int) -> Circle {
  returned var c: Circle;
  c.radius = radius;
  // `return c` would be invalid because `returned` is in use.
  return var;
}
```

## Relevant proposals

-   [Initialization of memory and variables](/proposals/p0257.md)
-   [Syntax: `return`](/proposals/p0415.md)
-   [`return` with no argument](/proposals/p0538.md)
