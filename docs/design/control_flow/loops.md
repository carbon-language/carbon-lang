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
        -   [`Iterate` interface](#iterate-interface)
        -   [Loop desugaring](#loop-desugaring)
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

For example, this prints all names in `names`:

```carbon
for (name: String in names) {
  Print(name);
}
```

`PrintNames()` prints each `String` in the `names` `List` in iteration order.

> **Note:** A name binding in the pattern of a `for` loop is a value binding by
> default. If a mutable loop variable is desired, the pattern can use `var`. For
> example: `for (var x: T in range)`.

To support iterating over user-defined types, the `for` loop is defined in terms
of the `Core.Iterate` interface. A container type can implement the
`Core.Iterate` interface to make it iterable with `for`.

#### `Iterate` interface

The `Core.Iterate` interface is defined as:

```carbon
interface Iterate {
  let ElementType:! Copy & Destroy;
  let CursorType:! Destroy;
  fn NewCursor(self) -> CursorType;
  fn Next(self, ref cursor: CursorType) -> Optional(ElementType);
}
```

-   **`CursorType`**: A type that represents the current position in the
    iteration.
-   **`ElementType`**: The type of the elements returned by the iteration.
-   **`NewCursor`**: Returns a new cursor initialized to the start of iteration.
-   **`Next`**: Advances the cursor and returns the next element, or `None` if
    the end of iteration has been reached. It takes the cursor by reference
    (`ref cursor: CursorType`), allowing the method to modify the cursor
    in-place.

#### Loop desugaring

A `for` loop of the form:

```carbon
for (<pattern> in <range>) {
  <statements>
}
```

is desugared to a loop that manages the cursor and checks the optional return
values:

```carbon
{
  let range:? auto = <range>;
  var cursor: auto = range.(Iterate.NewCursor)();
  while (true) {
    match (range.(Iterate.Next)(cursor)) {
      case .Some(<pattern>) => { <statements> }
      default => { break; }
    }
  }
}
```

> **Note:** Any temporaries in `<range>` will remain live until the end of the
> loop.

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

-   [Non-C++ syntax](/proposals/p000340-while-loops.md#non-c-syntax)
-   [Initializing variables in the `while`](/proposals/p000340-while-loops.md#initializing-variables-in-the-while)
-   `for`:
    -   [Include semisemi `for` loops](/proposals/p000353-for-loops.md#include-semisemi-for-loops)
    -   [Multi-variable bindings](/proposals/p000353-for-loops.md#multi-variable-bindings)
    -   [`:` versus `in`](/proposals/p000618-var-ordering.md#-versus-in)
    -   [Atomic methods for `Iterate` (instead of a single `Next` method)](/proposals/p001885-for-statement-and-user-types.md#atomic-methods-for-iterate)
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
-   Proposal
    [#7381: Adopt `ref` in `Core.Iterate.Next`](https://github.com/carbon-language/carbon-lang/pull/7381)
