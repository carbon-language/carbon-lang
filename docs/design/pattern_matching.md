# Pattern matching

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Pattern match control flow](#pattern-match-control-flow)
    -   [Pattern matching in local variables](#pattern-matching-in-local-variables)
-   [Open questions](#open-questions)
    -   [Slice or array nested value pattern matching](#slice-or-array-nested-value-pattern-matching)
    -   [Generic/template pattern matching](#generictemplate-pattern-matching)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

The most prominent mechanism to manipulate and work with types in Carbon is
pattern matching. This may seem like a deviation from C++, but in fact this is
largely about building a clear, coherent model for a fundamental part of C++:
overload resolution.

### Pattern match control flow

The most powerful form and easiest to explain form of pattern matching is a
dedicated control flow construct that subsumes the `switch` of C and C++ into
something much more powerful, `match`. This is not a novel construct, and is
widely used in existing languages (Swift and Rust among others) and is currently
under active investigation for C++. Carbon's `match` can be used as follows:

```
fn Bar() -> (i32, (f32, f32));
fn Foo() -> f32 {
  match (Bar()) {
    case (42, (x: f32, y: f32)) => {
      return x - y;
    }
    case (p: i32, (x: f32, _: f32)) if (p < 13) => {
      return p * x;
    }
    case (p: i32, _: auto) if (p > 3) => {
      return p * Pi;
    }
    default => {
      return Pi;
    }
  }
}
```

There is a lot going on here. First, let's break down the core structure of a
`match` statement. It accepts a value that will be inspected, here the result of
the call to `Bar()`. It then will find the _first_ `case` that matches this
value, and execute that block. If none match, then it executes the default
block.

Each `case` contains a pattern. The first part is a value pattern
(`(p: i32, _: auto)` for example) optionally followed by an `if` and boolean
predicate. The value pattern has to match, and then the predicate has to
evaluate to `true` for the overall pattern to match. Value patterns can be
composed of the following:

-   An expression (`42` for example), whose value must be equal to match.
-   An identifier to bind the value to, followed by a colon (`:`) and a type
    (`i32` for example). An underscore (`_`) may be used instead of the
    identifier to discard the value once matched.
-   A tuple destructuring pattern containing a tuple of value patterns
    (`(x: f32, y: f32)`) which match against tuples and tuple-like values by
    recursively matching on their elements.
-   An unwrapping pattern containing a nested value pattern which matches
    against a variant or variant-like value by unwrapping it.

In order to match a value, whatever is specified in the pattern must match.
Using `auto` for a type will always match, making `_: auto` the wildcard
pattern.

### Pattern matching in local variables

Value patterns may be used when declaring local variables to conveniently
destructure them and do other type manipulations. However, the patterns must
match at compile time, so they can't use an `if` clause.

```
fn Bar() -> (i32, (f32, f32));
fn Foo() -> i32 {
  var (p: i32, _: auto) = Bar();
  return p;
}
```

This extracts the first value from the result of calling `Bar()` and binds it to
a local variable named `p` which is then returned.

## Open questions

### Slice or array nested value pattern matching

An open question is how to effectively fit a "slice" or "array" pattern into
nested value pattern matching, or whether we shouldn't do so.

### Generic/template pattern matching

An open question is going beyond a simple "type" to things that support generics
and/or templates.

### Pattern matching as function overload resolution

Need to flesh out specific details of how overload selection leverages the
pattern matching machinery, what (if any) restrictions are imposed, etc.
