# Control flow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Overview

Blocks of statements are generally executed linearly. However, statements are
the primary place where this flow of execution can be controlled.

Carbon's flow control statements are:

-   [`if` and `else`](conditionals.md) provides conditional execution of
    statements.
-   Loops:
    -   [`while`](loops.md#while) executes the loop body for as long as the loop
        expression returns `True`.
    -   [`for`](loops.md#for) iterates over an object, such as elements in an
        array.
    -   [`break`](loops.md#break) exits loops.
    -   [`continue`](loops.md#continue) goes to the next iteration of a loop.
-   [`return`](return.md) ends the flow of execution within a function,
    returning it to the caller.
