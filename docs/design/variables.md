# Variables

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Notes](#notes)
-   [Alternatives](#alternatives)
    -   [Global variables](#global-variables)
-   [Relevant proposals](#relevant-proposals)

<!-- tocstop -->

## Overview

Carbon's local variable syntax is:

> `var` _identifier_`:` _type_ _[_ `=` _value_ _]_`;`

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters.

For example:

```
fn Foo() {
  var x: Int = 42;
}
```

This introduces a local variable named `x` into the block's scope. It has the
type `Int` and is initialized with the value `42`. These variable declarations
(and function declarations) have a lot more power than what we're covering just
yet, but this gives you the basic idea.

While there can be global constants, there are no global variables.

## Notes

> TODO: Constant syntax is an ongoing discussion.

## Alternatives

### Global variables

We are exploring several different ideas for how to design less bug-prone
patterns to replace the important use cases programmers still have for global
variables. We may be unable to fully address them, at least for migrated code,
and be forced to add some limited form of global variables back. We may also
discover that their convenience outweighs any improvements afforded.

## Relevant proposals

Most discussion of design choices and alternatives may be found in relevant
proposals.

-   [`var` statement](/proposals/p0339.md)
-   [`var` ordering](/proposals/p0618.md)
