# Blocks and statements

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

The body or definition of a function is provided by a block of code containing
statements, much like in C or C++. The body of a function is also a new, nested
scope inside the function's scope (meaning that parameter names are available).
Statements within a block are terminated by a semicolon. Each statement can,
among other things, be an expression. Here is a trivial example of a function
definition using a block of statements:

```
fn Foo() {
  Bar();
  Baz();
}
```

Statements can also themselves be a block of statements, which provide scopes
and nesting:

```
fn Foo() {
  Bar();
  {
    Baz();
  }
}
```
