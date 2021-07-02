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

> `return` _[ expression ];_

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

[Variables](../variables.md) may be declared with a `returned` statement. Its
syntax is:

> `returned var` _var syntax_

When a variable is marked as `returned`, it must be the only `returned` value
in-scope.

If a `returned var` is returned, the specific syntax `return var` must be used.
Returning other expressions is not allowed while a `returned var` is in-scope.
For example:

```carbon
fn MakeCircle(radius: Int) -> Circle {
  returned var c: Circle;
  c.radius = radius;
  // `return c` would be invalid because `returned` is in use.
  return var;
}
```

If control flow exits the scope of a `returned` variable in any way other than
`return var`, the `returned var`'s lifetime ends as normal. When this occurs,
`return` may again be used with expressions. For example:

```carbon
fn MakePointInArea(Area area, Int preferred_x, Int preferred_y) -> Point {
  {
    returned var p: Point = { .x = preferred_x, .y = preferred_y };
    if (area.Contains(p)) {
      return var;
    }
    // p's lifetime ends here when `return var` is not reached.
  }

  return area.RandomPoint();
}
```

## Relevant proposals

-   [Initialization of memory and variables](/proposals/p0257.md)
-   [Syntax: `return`](/proposals/p0415.md)
-   [`return` with no argument](/proposals/p0538.md)
