# Loops

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [`while`](#while)
-   [`for`](#for)
-   [`break`](#break)
-   [`continue`](#continue)
-   [Relevant proposals](#relevant-proposals)

<!-- tocstop -->

## `while`

`while` statements loop for as long as the passed expression returns `True`. For
example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var x: Int = 0;
while (x < 3) {
  Print(x);
  ++x;
}
Print("Done!");
```

> TODO: Flesh out text (currently just overview)

## `for`

`for` statements support range-based looping, typically over containers. For
example, this prints all names in `names`:

```carbon
for (var name: String in names) {
  Print(name);
}
```

`PrintNames()` prints each `String` in the `names` `List` in iteration order.

> TODO: Flesh out text (currently just overview)

## `break`

The `break` statement immediately ends a `while` or `for` loop. Execution will
resume at the end of the loop's scope. For example, this processes steps until a
manual step is hit (if no manual step is hit, all steps are processed):

```carbon
for (var step: Step in steps) {
  if (step.IsManual()) {
    Print("Reached manual step!");
    break;
  }
  step.Process();
}
```

> TODO: Flesh out text (currently just overview)

## `continue`

The `continue` statement immediately goes to the next loop of a `while` or
`for`. In a `while`, execution continues with the `while` expression. For
example, this prints all non-empty lines of a file, using `continue` to skip
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

> TODO: Flesh out text (currently just overview)

## Relevant proposals

Most discussion of design choices and alternatives may be found in relevant
proposals.

-   [`while`](/proposals/p0340.md)
-   [`for`](/proposals/p0353.md)
