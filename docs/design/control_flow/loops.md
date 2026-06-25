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
        -   [Ranged-for for user-defined types](#ranged-for-for-user-defined-types)
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

> `for (` _pattern_ `in` _expression_ `) {` _statements_ `}`

For consistency with function parameters and other pattern matching contexts,
the pattern defaults to value (immutable) bindings, like `let`.

For example, this prints all names in `names`:

```carbon
for (name: strbuf in names) {
  Print(name);
}
```

This default can be overridden by adding the `var` keyword:

```carbon
for (var name: strbuf in names) {
  // `name` can be modified, but this will not modify the underlying `names` container.
}
```

Temporary entities on the right-hand side of `in` remain alive during the
execution of the `for` loop to prevent invalid memory access.

#### Ranged-for for user-defined types

User types can enable support for ranged-for loops by implementing the `Iterate`
interface:

```carbon
interface Iterate {
  let ElementType:! type;
  let CursorType:! type;
  fn NewCursor(self) -> CursorType;
  fn Next(self, ref cursor: CursorType) -> Optional(ElementType);
}
```

The cursor tracks progression, and the `Next` method advances the cursor and
returns an `Optional` value. An empty `Optional` indicates that we have reached
the end.

A `for` loop on a container of a type that implements `Iterate` behaves
conceptually as (though an API for `Optional` has not been approved):

```carbon
var cursor: range.(Iterate.CursorType) = range.(Iterate.NewCursor)();
var iter: Optional(range.(Iterate.ElementType)) = range.(Iterate.Next)(&cursor);
while (iter.HasValue()) {
  ExecuteForBlock(iter.Get());
  iter = container.(Iterate.Next)(ref cursor);
}
```

### `break`

The `break` statement immediately ends a `while` or `for` loop. Execution will
resume at the end of the loop's scope. Syntax is:

> `break;`

For example, this processes steps until a manual step is hit (if no manual step
is hit, all steps are processed):

```carbon
for (step: Step in steps) {
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
  var line: strbuf = f.ReadLine();
  if (line.IsEmpty()) {
    continue;
  }
  Print(line);
}
```

## Alternatives considered

-   [Non-C++ syntax](/proposals/p000340-while-loops.md#non-c-syntax)
-   [Initializing variables in the `while`](/proposals/p000340-while-loops.md#initializing-variables-in-the-while)
-   `for`:
    -   [Include semisemi `for` loops](/proposals/p000353-for-loops.md#include-semisemi-for-loops)
    -   [Multi-variable bindings](/proposals/p000353-for-loops.md#multi-variable-bindings)
    -   [`:` versus `in`](/proposals/p000618-var-ordering.md#-versus-in)
    -   [Atomic methods for `Iterate`](/proposals/p001885-for-statement-and-user-types.md#atomic-methods-for-iterate)
    -   [Using an iterator instead of a cursor](/proposals/p001885-for-statement-and-user-types.md#using-an-iterator-instead-of-a-cursor)
    -   [Support getter for both `T` and `T*` with `Iterate`](/proposals/p001885-for-statement-and-user-types.md#support-getter-for-both-t-and-t-with-iterate)
-   [Optional braces](/proposals/p000623-require-braces.md#optional-braces)
-   [Optional parentheses](/proposals/p000623-require-braces.md#optional-parentheses)

## References

-   Proposal
    [#340: `while`](https://github.com/carbon-language/carbon-lang/pull/340)
-   Proposal
    [#353: `for`](https://github.com/carbon-language/carbon-lang/pull/353)
-   Proposal
    [#618: `var` ordering](https://github.com/carbon-language/carbon-lang/pull/618)
-   Proposal
    [#623: Require braces](https://github.com/carbon-language/carbon-lang/pull/623)
-   Proposal
    [#1885: `for` statement and user types](https://github.com/carbon-language/carbon-lang/pull/1885)
