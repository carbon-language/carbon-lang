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
