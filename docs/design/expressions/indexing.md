# Indexing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Examples](#examples)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon supports indexing using the conventional `a[i]` subscript syntax. When
`a` is a [value expression](/docs/design/values.md#value-expressions), the
result can be a durable reference expression or a value expression, depending on
which constraint the type implements:

-   If subscripting a value expression produces a value expression, as with an
    array, the type should implement `IndexWith`.
-   If subscripting a value expression produces a durable reference expression,
    as with C++'s `std::span`, the type should implement `IndirectIndexWith`.

When `a` is a
[durable reference expression](/docs/design/values.md#durable-reference-expressions),
the result of subscripting is also a durable reference expression if either of
those constraints are implemented.

Any type that implements `IndirectIndexWith` automatically also implements
`IndexWith`.

Other behaviors can be accomplished by implementing the underlying interface
`IndexWithPrimitive`.

`IndirectIndexWith` and `IndexWith` overlap, so a type can implement at most one
of those two constraints.

The `Ref` methods of these interfaces, which are used to form durable reference
expressions on indexing, return by `ref`. The `IndexWith` interface also has an
`At` method that returns by value.

## Details

A subscript expression has the form "_lhs_ `[` _index_ `]`". As in C++, this
syntax has the same precedence as `.`, `->`, and function calls, and associates
left-to-right with all of them.

Its semantics are defined in terms of the following interface:

```carbon
package Core;

interface IndexWithPrimitive
    [anchor Self:! Form]
    (in Subscript:! Form) {
  let out ResultForm:! Form;
  fn Op[bound self:? Self](subscript:? Subscript)
    ->? ResultForm;
}
```

The expression "_lhs_ `[` _index_ `]`" is rewritten to "_lhs_
`.(IndexWithPrimitive(formof(` _index_ `)).Op)(` _index_ `)`".

The named constraint `IndexWith` covers the case of indexing into an object that
owns the storage for its elements, like an array or C++'s `std::vector`:

```carbon
// `lhs[index]` is a reference expression or value
// depending on the category of `lhs`.
constraint IndexWith(SubscriptType:! type) {
  let ElementType:! type;

  require form(let Self) impls
     IndexWithPrimitive(form(let SubscriptType))
     where .ResultForm = form(let ElementType);
  alias At = LetSelf(IndexWithPrimitive(
      form(let SubscriptType))).Op;

  require form(ref Self) impls
     IndexWithPrimitive(form(let SubscriptType))
     where .ResultForm = form(ref ElementType);
  alias Ref = RefSelf(IndexWithPrimitive(
      form(let SubscriptType))).Op;
}
```

FIXME: describe

```carbon
// `lhs[index]` is always reference expression, even
// when `lhs` is a value.
constraint IndirectIndexWith(SubscriptType:! type) {
  let ElementType:! type;

  require form(let Self) impls
     IndexWithPrimitive(form(let SubscriptType))
     where .ResultForm = form(ref ElementType);
  alias Ref = LetSelf(IndexWithPrimitive(
      form(let SubscriptType))).Op;
}
```

FIXME: describe will be used no matter the category of the _lhs_ argument.

FIXME: both of these require

```
interface IndexWith(SubscriptType:! type) {
  let ElementType:! type;
  fn At[self: Self](subscript: SubscriptType) -> ElementType;
  fn Addr[addr self: Self*](subscript: SubscriptType) -> ElementType*;
}

interface IndirectIndexWith(SubscriptType:! type) {
  require Self impls IndexWith(SubscriptType);
  fn Addr[self: Self](subscript: SubscriptType) -> ElementType*;
}
```

A subscript expression where _lhs_ has type `T` and _index_ has type `I` is
rewritten based on the expression category of _lhs_ and whether `T` is known to
implement `IndirectIndexWith(I)`:

-   If `T` implements `IndirectIndexWith(I)`, the expression is rewritten to
    "`*((` _lhs_ `).(IndirectIndexWith(I).Addr)(` _index_ `))`".
-   Otherwise, if _lhs_ is a
    [_durable reference expression_](/docs/design/values.md#durable-reference-expressions),
    the expression is rewritten to "`*((` _lhs_ `).(IndexWith(I).Addr)(` _index_
    `))`".
-   Otherwise, the expression is rewritten to "`(` _lhs_ `).(IndexWith(I).At)(`
    _index_ `)`".

`IndirectIndexWith` provides a blanket `final impl` for `IndexWith`:

```
final impl forall
    [SubscriptType:! type, T:! IndirectIndexWith(SubscriptType)]
    T as IndexWith(SubscriptType) {
  let ElementType:! type = T.(IndirectIndexWith(SubscriptType)).ElementType;
  fn At[self: Self](subscript: SubscriptType) -> ElementType {
    return *(self.(IndirectIndexWith(SubscriptType).Addr)(index));
  }
  fn Addr[addr self: Self*](subscript: SubscriptType) -> ElementType* {
    return self->(IndirectIndexWith(SubscriptType).Addr)(index);
  }
}
```

Thus, a type that implements `IndirectIndexWith` need not, and cannot, provide
its own definitions of `IndexWith.At` and `IndexWith.Addr`.

### Examples

An array type could implement subscripting like so:

```
class Array(template T:! type) {
  impl as IndexWith(like i64) {
    let ElementType:! type = T;
    fn At[self: Self](subscript: i64) -> T;
    fn Addr[addr self: Self*](subscript: i64) -> T*;
  }
}
```

And a type such as `std::span` could look like this:

```
class Span(T:! type) {
  impl as IndirectIndexWith(like i64) {
    let ElementType:! type = T;
    fn Addr[self: Self](subscript: i64) -> T*;
  }
}
```

## Alternatives considered

-   [Different subscripting syntaxes](/proposals/p2274.md#different-subscripting-syntaxes)
-   [Multiple indices](/proposals/p2274.md#multiple-indices)
-   [Read-only subscripting](/proposals/p2274.md#read-only-subscripting)
-   [Rvalue-only subscripting](/proposals/p2274.md#rvalue-only-subscripting)
-   [Map-like subscripting](/proposals/p2274.md#map-like-subscripting)

## References

-   Proposal
    [#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
-   Proposal
    [#2006: Values, variables, and pointers](https://github.com/carbon-language/carbon-lang/pull/2006)
