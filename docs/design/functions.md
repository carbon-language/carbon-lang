# Functions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Basic functions](#basic-functions)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Basic functions

Programs written in Carbon, much like those written in other languages, are
primarily divided up into "functions" (or "procedures", "subroutines", or
"subprograms"). These are the core unit of behavior for the programming
language. Let's look at a simple example to understand how these work:

```
fn Sum(Int: a, Int: b) -> Int;
```

This declares a function called `Sum` which accepts two `Int` parameters, the
first called `a` and the second called `b`, and returns an `Int` result. C++
might declare the same thing:

```
std::int64_t Sum(std::int64_t a, std::int64_t b);

// Or with trailing return type syntax:
auto Sum(std::int64_t a, std::int64_t b) -> std::int64_t;
```

Let's look at how some specific parts of this work. The function declaration is
introduced with a keyword `fn` followed by the name of the function `Sum`. This
declares that name in the surrounding scope and opens up a new scope for this
function. We declare the first parameter as `Int: a`. The `Int` part is an
expression (here referring to a constant) that computes the type of the
parameter. The `:` marks the end of the type expression and introduces the
identifier for the parameter, `a`. The parameter names are introduced into the
function's scope and can be referenced immediately after they are introduced.
The return type is indicated with `-> Int`, where again `Int` is just an
expression computing the desired type. The return type can be completely omitted
in the case of functions which do not return a value.

Calling functions involves a new form of expression: `Sum(1, 2)` for example.
The first part, `Sum`, is an expression referring to the name of the function.
The second part, `(1, 2)` is a parenthesized list of arguments to the function.
The juxtaposition of one expression with parentheses forms the core of a call
expression, similar to a postfix operator.
