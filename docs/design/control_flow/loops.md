# Loops

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [`while`](#while)
    -   [`for`](#for)
    -   [`break`](#break)
    -   [`continue`](#continue)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides loops using the `while` and `for` statements. Within a loop, the
`break` and `continue` statements can be used for flow control.

## Details

### `while`

`while` statements loop for as long as the passed expression returns `True`.
Syntax is:

> `while (` _boolean expression_ `) {` _statements_ `}`

For example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var x: Int = 0;
while (x < 3) {
  Print(x);
  ++x;
}
Print("Done!");
```

### `for`

`for` statements support range-based looping, typically over containers. Syntax
is:

> `for (` _var declaration_ `in` _expression_ `) {` _statements_ `}`

For example, this prints all names in `names`:

```carbon
for (var name: String in names) {
  Print(name);
}
```

`PrintNames()` prints each `String` in the `names` `List` in iteration order.

### `break`

The `break` statement immediately ends a `while` or `for` loop. Execution will
resume at the end of the loop's scope. Syntax is:

> `break;`

For example, this processes steps until a manual step is hit (if no manual step
is hit, all steps are processed):

```carbon
for (var step: Step in steps) {
  if (step.IsManual()) {
    Print("Reached manual step!");
    break;
  }
  step.Process();
}
```

### `continue`

The `continue` statement immediately goes to the next loop of a `while` or
`for`. In a `while`, execution continues with the `while` expression. Syntax is:

> `continue;`

For example, this prints all non-empty lines of a file, using `continue` to skip
empty lines:

```carbon
var f: File = OpenFile(path);
while (!f.EOF()) {
  var line: String = f.ReadLine();
  if (line.IsEmpty()) {
    continue;
  }
  Print(line);
}
```

## Alternatives considered

-   [Non-C++ syntax](/proposals/p0340.md#non-c-syntax)
-   [Initializing variables in the `while`](/proposals/p0340.md#initializing-variables-in-the-while)
-   `for`:
    -   [Include semisemi `for` loops](/proposals/p0353.md#include-semisemi-for-loops)
    -   [Multi-variable bindings](/proposals/p0353.md#multi-variable-bindings)
    -   [`:` versus `in`](/proposals/p0618.md#-versus-in)
-   [Optional braces](/proposals/p0623.md#optional-braces)
-   [Optional parentheses](/proposals/p0623.md#optional-parentheses)

## References

-   Proposal
    [#340: `while`](https://github.com/carbon-language/carbon-lang/pull/340)
-   Proposal
    [#353: `for`](https://github.com/carbon-language/carbon-lang/pull/353)
-   Proposal
    [#618: `var` ordering](https://github.com/carbon-language/carbon-lang/pull/618)
-   Proposal
    [#623: Require braces](https://github.com/carbon-language/carbon-lang/pull/623)
