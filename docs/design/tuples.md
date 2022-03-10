# Tuples

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Indices as compile-time constants](#indices-as-compile-time-constants)
    -   [Operations performed field-wise](#operations-performed-field-wise)
-   [Open questions](#open-questions)
    -   [Slicing ranges](#slicing-ranges)
    -   [Single-value tuples](#single-value-tuples)
    -   [Function pattern match](#function-pattern-match)
    -   [Type vs tuple of types](#type-vs-tuple-of-types)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

The primary composite type involves simple aggregation of other types as a tuple
(called a "product type" in formal type theory):

```
fn DoubleBoth(x: i32, y: i32) -> (i32, i32) {
  return (2 * x, 2 * y);
}
```

This function returns a tuple of two integers represented by the type
`(i32, i32)`. The expression to return it uses a special tuple syntax to build a
tuple within an expression: `(<expression>, <expression>)`. This is actually the
same syntax in both cases. The return type is a tuple expression, and the first
and second elements are expressions referring to the `i32` type. The only
difference is the type of these expressions. Both are tuples, but one is a tuple
of types.

Element access uses subscript syntax:

```
fn Bar(x: i32, y: i32) -> i32 {
  var t: (i32, i32) = (x, y);
  return t[0] + t[1];
}
```

Tuples also support multiple indices and slicing to restructure tuple elements:

```
fn Baz(x: i32, y: i32, z: i32) -> (i32, i32) {
  var t1: (i32, i32, i32) = (x, y, z);
  var t2: (i32, i32, i32) = t1[(2, 1, 0)];
  return t2[0 .. 2];
}
```

This code first reverses the tuple, and then extracts a slice using a half-open
range of indices.

### Indices as compile-time constants

In the example `t1[(2, 1, 0)]`, we will likely want to restrict these indices to
compile-time constants. Without that, run-time indexing would need to suddenly
switch to a variant-style return type to handle heterogeneous tuples. This would
both be surprising and complex for little or no value.

### Operations performed field-wise

Like some other aggregate data types like
[struct types](classes.md#struct-types), there are some operations are defined
for tuples field-wise:

-   initialization
-   assignment
-   equality and inequality comparison
-   ordered comparison
-   implicit conversion for argument passing
-   destruction

For binary operations, the two tuples must have the same number of components
and the operation must be defined for the corresponding component types of the
two tuples.

**References:** The rules for assignment, comparison, and implicit conversion
for argument passing were decided in
[question-for-leads issue #710](https://github.com/carbon-language/carbon-lang/issues/710).

## Open questions

### Slicing ranges

The intent of `0 .. 2` is to be syntax for forming a sequence of indices based
on the half-open range [0, 2). There are a bunch of questions we'll need to
answer here:

-   Is this valid anywhere? Only some places?
-   What _is_ the sequence?
    -   If it is a tuple of indices, maybe that solves the above issue, and
        unlike function call indexing with multiple indices is different from
        indexing with a tuple of indexes.
-   Do we need syntax for a closed range (`...` perhaps, unclear if that ends up
    _aligned_ or in _conflict_ with other likely uses of `...` in pattern
    matching)?
-   All of these syntaxes are also very close to `0.2`, is that similarity of
    syntax OK?
    -   Do we want to require the `..` to be surrounded by whitespace to
        minimize that collision?

### Single-value tuples

This remains an area of active investigation. There are serious problems with
all approaches here. Without the collapse of one-tuples to scalars we need to
distinguish between a parenthesized expression (`(42)`) and a one tuple (in
Python or Rust, `(42,)`), and if we distinguish them then we cannot model a
function call as simply a function name followed by a tuple of arguments; one of
`f(0)` and `f(0,)` becomes a special case. With the collapse, we either break
genericity by forbidding `(42)[0]` from working, or it isn't clear what it means
to access a nested tuple's first element from a parenthesized expression:
`((1, 2))[0]`.

### Function pattern match

There are some interesting corner cases we need to expand on to fully and more
precisely talk about the exact semantic model of function calls and their
pattern match here, especially to handle variadic patterns and forwarding of
tuples as arguments. We are hoping for a purely type system answer here without
needing templates to be directly involved outside the type system as happens in
C++ variadics.

### Type vs tuple of types

Is `(i32, i32)` a type, a tuple of types, or is there even a difference between
the two? Is different syntax needed for these cases?
