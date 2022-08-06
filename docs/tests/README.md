# Documentation tests

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document describes a minimal DSL that can be used to transform markdown fenced code blocks into test cases for the Carbon Explorer interpreter.

In order to test a fenced code block like

````
```
for (var name: String in names) {
  Console.Print(name);
}
```
````

The block must be preceded by a comment starting with `test`

````
<!-- test -->
```
for (var name: String in names) {
  Console.Print(name);
}
```
````

Everything between `test` and the end of the comment is treated as a list of space-separated generator commands (described below). If this list is empty, like above, then no test case is actually generated. This (easily searchable) form can be used to express the intent that the snippet ought to be tested, but Explorer does not yet support the required functionality.

Fenced blocks without a language tag are assumed to use Carbon syntax. Fenced blocks in other languages and blocks without `<!-- test ... --->` comments are ignored.

## Generator commands

Commands are executed in reading order (left to right). The following commands are available:

### Paste current snippet

`_`

Writes the contents of the current snippet (and the result of any transformation command preceding the `_`) to the output buffer.

#### Example

````
<!-- test _ -->
```
fn Main() -> i32 {
  return 0;
}
```
````

### Paste literal or named code snippet

`` `literalblock` `` or `namedblock`

#### Example

````
<!-- test `fn Main() -> i32 {` _ `}` -->
```
return 0;
```
````
or equivalently
````
<!-- test mo _ c -->
```
return 0;
```
````

Where `mo` (main open) and `c` (close brace) are predefined named snippets.

### Delete lines from current snippet

`-N` or `-N+M`

Delete `M` lines starting with line `N` (`M` is 1 if omitted). Can be used to delete lines that explain wrong syntax and should not be part of the current test case.

#### Example

````
<!-- test -11 -8 _ -->
```
fn Main() -> i32 {
  // ✅ Same as (1 | 2) | 4, evaluates to 7.
  var a: i32 = 1 | 2 | 4;

  // ❌ Error, parentheses are required to distinguish between
  //    (3 | 5) & 6, which evaluates to 6, and
  //    3 | (5 & 6), which evaluates to 7.
  var b: i32 = 3 | 5 & 6;

  // ❌ Error, can't repeat the `^` operator. Use `^(^4)` or simply `4`.
  var d: i32 = ^^4;

  return 0;
}
```
````
Note that lines are deleted bottom up to preserve numbering.

### Insert lines in current snippet

`` +L`code` `` or `+Lblockname`

Insert the literal code `` `code` `` or the named code block `blockname` at line `L`.

#### Example

````
<!-- test +5rc +4mo _ -->
```
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
Add(20, 22);
```
````

Note that lines are inserted bottom up to preserve numbering, `rc` and `mo` are predefined named snippets.

### Replace ellipsis in current snippet

`` .`code` `` or `.blockname`

Replaces successive occurrences of `...` with the contents of the specified block literal or named block. A special form `.` can be used to replace the ellipsis with an empty string.

#### Example

````
<!-- test . .`return false;` .r . _ m -->
```
class Song {
  fn Play[me: Self]() { ... }
  fn Playing[me: Self]() -> bool { ... }
  fn Duration[me: Self] -> i32 { ... }
  fn Run[me: Self]() { ... }
}
```
````

### Save current snippet with name

` =blockname `

#### Example

````
<!-- test =point _ m -->
```
class Point {
  var x: i32;
  var y: i32;
  fn Origin() -> Self {
    return {.x = 0, .y = 0};
  }
}
```

then later in the same .md file

<!-- test point mo _ rc -->
```
var p1: Point = Point.Origin();
```

````

#### Predefined named snippets

`m` means
```
fn Main() -> i32 { return 0; }
```

`mo` means
```
fn Main() -> i32 {
```

`rc` means
```
return 0; }
```

`r` means
```
return 0;
```

`c` means
```
}
```
