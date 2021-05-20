# Control flow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [`if` and `else`](#if-and-else)
    -   [Loops](#loops)
        -   [`while`](#while)
        -   [`for`](#for)
        -   [`break`](#break)
        -   [`continue`](#continue)
    -   [`return`](#return)
-   [Relevant proposals](#relevant-proposals)

<!-- tocstop -->

## Overview

Blocks of statements are generally executed linearly. However, statements are
the primary place where this flow of execution can be controlled.

Carbon's flow control statements are:

-   [`if` and `else`](#if-and-else) provides conditional execution of
    statements.
-   Loops:
    -   [`while`](#while) executes the loop body for as long as the loop
        expression returns `True`.
    -   [`for`](#for) iterates over an object, such as elements in an array.
    -   [`break`](#break) exits loops.
    -   [`continue`](#continue) goes to the next iteration of a loop.
-   [`return`](#return) ends the flow of execution within a function, returning
    it to the caller.

### `if` and `else`

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

### Loops

#### `while`

`while` statements loop for as long as the passed expression returns `True`. For
example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var Int x = 0;
while (x < 3) {
  Print(x);
  ++x;
}
Print("Done!");
```

> TODO: Flesh out text (currently just overview)

#### `for`

`for` statements support range-based looping, typically over containers. For
example, this prints all names in `names`:

```carbon
for (var String name : names) {
  Print(name);
}
```

`PrintNames()` prints each `String` in the `names` `List` in iteration order.

> TODO: Flesh out text (currently just overview)

#### `break`

The `break` statement immediately ends a `while` or `for` loop. Execution will
resume at the end of the loop's scope. For example, this processes steps until a
manual step is hit (if no manual step is hit, all steps are processed):

```carbon
for (var Step step : steps) {
  if (step.IsManual()) {
    Print("Reached manual step!");
    break;
  }
  step.Process();
}
```

> TODO: Flesh out text (currently just overview)

#### `continue`

The `continue` statement immediately goes to the next loop of a `while` or
`for`. In a `while`, execution continues with the `while` expression. For
example, this prints all non-empty lines of a file, using `continue` to skip
empty lines:

```carbon
File f = OpenFile(path);
while (!f.EOF()) {
  String line = f.ReadLine();
  if (line.IsEmpty()) {
    continue;
  }
  Print(line);
}
```

> TODO: Flesh out text (currently just overview)

### `return`

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. For example:

```carbon
fn Sum(Int a, Int b) -> Int {
  return a + b;
}
```

> TODO: Flesh out text (currently just overview)

## Relevant proposals

Most discussion of design choices and alternatives may be found in relevant
proposals.

-   [`if` and `else`](/proposals/p0285.md)
-   Loops:
    -   [`while`](/proposals/p0340.md)
    -   [`for`](/proposals/p0353.md)
-   `return`:
    -   [Initial syntax](/proposals/p0415.md)
    -   [`return` with no argument](/proposals/p0538.md)
