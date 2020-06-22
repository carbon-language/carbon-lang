# Control flow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)
- [Open questions](#open-questions)
  - [`if` blocks](#if-blocks)
  - [`break` and `continue`](#break-and-continue)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

At least summarize `if` and `else` to cover basics. Especially important to
surface the idea of using basic conditionals as both expressions and statements
to avoid needing conditional operators.

Looping is an especially interesting topic to explore as there are lots of
challenges posed by the C++ loop structure. Even C++ itself has been seeing
significant interest and pressure to improve its looping facilities.

## Overview

Blocks of statements are generally executed linearly. However, statements are
the primary place where this flow of execution can be controlled. Carbon's
control flow constructs are mostly similar to those in C, C++, and other
languages.

```
fn Foo(Int: x) {
  if (x < 42) {
    Bar();
  } else if (x > 77) {
    Baz();
  }
}
```

Loops will at least be supported with a low-level primitive `loop` statement,
with `break` and `continue` statements which work the same as in C++.

Last but not least, for the basics we need to include the `return` statement.
This statement ends the flow of execution within a function, returning it to the
caller. If the function returns a value to the caller, that value is provided by
an expression in the return statement. This allows us to complete the definition
of our `Sum` function from earlier as:

```
fn Sum(Int: a, Int: b) -> Int {
  return a + b;
}
```

## Open questions

### `if` blocks

It is an open question whether a block is required or a single statement may be
nested in an `if` statement. Similarly, it is an open question whether `else if`
is a single keyword versus a nested `if` statement, and if it is a single
construct whether it should be spelled `elif` or something else.

### `break` and `continue`

If and how to support a "labeled break" or "labeled continue" is still a point
of open discussion.
