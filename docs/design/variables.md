# Variables

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Declaring constants](#declaring-constants)
-   [Alternatives](#alternatives)
    -   [Declaring constants](#declaring-constants-1)
    -   [Global variables](#global-variables)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters.

For example:

```
fn Foo() {
  var Int x = 42;
}
```

This introduces a local variable named `x` into the block's scope. It has the
type `Int` and is initialized with the value `42`. These variable declarations
(and function declarations) have a lot more power than what we're covering just
yet, but this gives you the basic idea.

While there can be global constants, there are no global variables.

### Declaring constants

Constants will use template-like syntax for declarations. For example, a simple
integer constant looks like:

```carbon
var Int$$ MyVal = 42;
```

## Alternatives

### Declaring constants

There is other syntax that could be used for declaring constants. There are
serious problems with the use of `const` in C++ as part of the type system.
Another alternative is `let` from Swift, although there are some questions
around how intuitive it is for this to introduce a constant. Another candidate
is `val` from Kotlin. Another thing we need to contend with is the surprise of
const and reference (semantic) types. At present we are leaning towards the
tempalte-like syntax for consistency within Carbon.

### Global variables

We are exploring several different ideas for how to design less bug-prone
patterns to replace the important use cases programmers still have for global
variables. We may be unable to fully address them, at least for migrated code,
and be forced to add some limited form of global variables back. We may also
discover that their convenience outweighs any improvements afforded.
