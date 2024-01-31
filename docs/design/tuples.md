# Tuples

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Element access](#element-access)
    -   [Empty tuples](#empty-tuples)
    -   [Trailing commas and single-element tuples](#trailing-commas-and-single-element-tuples)
    -   [Tuple of types and tuple types](#tuple-of-types-and-tuple-types)
    -   [Operations performed field-wise](#operations-performed-field-wise)
    -   [Pattern matching](#pattern-matching)
-   [Open questions](#open-questions)
    -   [Tuple slicing](#tuple-slicing)
    -   [Slicing ranges](#slicing-ranges)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

The primary composite type involves simple aggregation of other types as a
tuple, called a "product type" in formal type theory:

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

## Element access

Element access uses a syntax similar to field access, with an element index
instead of a field name:

```
fn Sum(x: i32, y: i32) -> i32 {
  var t: (i32, i32) = (x, y);
  return t.0 + t.1;
}
```

A parenthesized template constant expression can also be used to index a tuple:

```
fn Choose(template N:! i32) -> i32 {
  return (1, 2, 3).(N % 3);
}
```

### Empty tuples

`()` is the empty tuple. This is used in other parts of the design, such as
[functions](functions.md), where a type with a single value is needed.

### Trailing commas and single-element tuples

The final element in a tuple literal may be followed by a trailing comma, such
as `(1, 2,)`. This trailing comma is optional in tuples with two or more
elements, and mandatory in a tuple with a single element: `(x,)` is a one-tuple,
whereas `(x)` is a parenthesized single expression.

### Tuple of types and tuple types

A tuple of types can be used in contexts where a type is needed. This is made
possible by a built-in implicit conversion: a tuple can be implicitly converted
to type `type` if all of its elements can be converted to type `type`, and the
result of the conversion is the corresponding tuple type.

For example, `(i32, i32)` is a value of type `(type, type)`, which is not a type
but can be implicitly converted to a type. `(i32, i32) as type` can be used to
explicitly refer to the corresponding tuple type, which is the type of
expressions such as `(1 as i32, 2 as i32)`. However, this is rarely necessary,
as contexts requiring a type will implicitly convert their operand to a type:

```carbon
// OK, both (i32, i32) values are implicitly converted to `type`.
fn F(x: (i32, i32)) -> (i32, i32);
```

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

### Pattern matching

Tuple values can be matched using a
[tuple pattern](/docs/design/pattern_matching.md#tuple-patterns), which is
written as a tuple of element patterns:

```carbon
let tup: (i32, i32, i32) = (1, 2, 3);
match (tup) {
  case (a: i32, 2, var c: i32) => {
    c = a;
    return c + 1;
  }
}
```

## Open questions

### Tuple slicing

Tuples could support multiple indices and slicing to restructure tuple elements:

```
fn Baz(x: i32, y: i32, z: i32) -> (i32, i32) {
  var t1: (i32, i32, i32) = (x, y, z);
  var t2: (i32, i32, i32) = t1.((2, 1, 0));
  return t2.(0 .. 2);
}
```

This code would first reverse the tuple, and then extract a slice using a
half-open range of indices.

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

## Alternatives considered

-   [Indexing with square brackets](/proposals/p3646.md#square-bracket-notation)
-   [Indexing from the end of a tuple](/proposals/p3646.md#negative-indexing-from-the-end-of-the-tuple)
-   [Restrict indexes to decimal integers](/proposals/p3646.md#decimal-indexing-restriction)
-   [Alternatives to trailing commas](/proposals/p3646.md#trailing-commas)

## References

-   Proposal
    [#2188: Pattern matching syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2188)
-   Proposal
    [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
-   Proposal
    [#3646: Tuples and tuple indexing](https://github.com/carbon-language/carbon-lang/pull/3646)
-   Leads issue
    [#710](https://github.com/carbon-language/carbon-lang/issues/710)
    established rules for assignment, comparison, and implicit conversion
-   Leads issue
    [#2191: one-tuples and one-tuple syntax](https://github.com/carbon-language/carbon-lang/issues/2191)
