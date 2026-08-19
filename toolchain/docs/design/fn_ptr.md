# Function pointers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Syntax](#syntax)
-   [Implementation](#implementation)

<!-- tocstop -->

## Overview

Function pointers should eventually be implemented as library types, but that
depends on some fairly complex unimplemented features (such as variadics,
and possibly `dyn`). As a placeholder, the toolchain supports function pointer
types with a built-in syntax:

```carbon
fn F(_: Param1T, ref _: Param2T) -> ReturnT;

fn G(a: Param1T, var b: Param2T) {
  var f: __fn_ptr(form(val Param1T), form(ref Param2T)) ->? form(var ReturnT) =
    F;
  var x: ReturnT = f(a, ref b);  // Equivalent to `F(a, b)`
}
```

## Syntax

The syntax for a function pointer type is

-   _fn_ptr_type_ ::= `__fn_ptr` `(` _param_forms_ `)` `->?` _return_form_

where _param_forms_ is a comma-separated list of the forms of the parameters
and _return_form_ is the function's return form. The `->?` is mandatory; `->`
cannot be used in its place.

## Implementation

We model a call to a function pointer as a call to a special thunk function that
has the parameters and return form specified by the function pointer type,
but also takes the function pointer as a `self` parameter, so the call
`f(a, ref b)` above is roughly equivalent to `f.__fn_ptr_thunk(a, ref b)`.
This is another placeholder: eventually this will be modeled by having function
pointer types implement the `Call` interface, but we can't even define the
`Call` interface until we support variadics.

These thunks don't have bodies or linkage, because they exist only for the
purpose of checking the callsite. In effect, they are always inlined during
lowering: a call to a function pointer thunk is lowered to a `call` instruction
with the function pointer as the callee.

In the absence of variadics and the `Call` interface, we can't make function
pointer thunks generic, and so we can't use the generics machinery to
de-duplicate them. Instead, we generate them on demand, using a
per-source-file cache to minimize duplicate work.
