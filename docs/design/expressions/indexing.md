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
-   [Open questions](#open-questions)
    -   [Tuple indexing](#tuple-indexing)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon supports indexing using the conventional `a[i]` subscript syntax. When
`a` is an l-value, the result of subscripting is always an l-value, but when `a`
is an r-value, the result can be an l-value or an r-value, depending on which
interface the type implements:

-   If subscripting an r-value produces an r-value result, as with an array, the
    type should implement `IndexWith`.
-   If subscripting an r-value produces an l-value result, as with C++'s
    `std::span`, the type should implement `IndirectIndexWith`.

`IndirectIndexWith` is a subtype of `IndexWith`, and subscript expressions are
rewritten to method calls on `IndirectIndexWith` if the type is known to
implement that interface, or to method calls on `IndexWith` otherwise.

`IndirectIndexWith` provides a final blanket `impl` of `IndexWith`, so a type
can implement at most one of those two interfaces.

## Details

A subscript expression has the form "_lhs_ `[` _index_ `]`". As in C++, this
syntax has the same precedence as `.`, `->`, and function calls, and associates
left-to-right with all of them.

Its semantics are defined in terms of the following interfaces:

```
interface IndexWith(SubscriptType:! type) {
  let ElementType:! type;
  fn At[me: Self](subscript: SubscriptType) -> ElementType;
  fn Addr[addr me: Self*](subscript: SubscriptType) -> ElementType*;
}

interface IndirectIndexWith(SubscriptType:! type) {
  impl as IndexWith(SubscriptType);
  fn Addr[me: Self](subscript: SubscriptType) -> ElementType*;
}
```

A subscript expression where _lhs_ has type `T` and _index_ has type `I` is
rewritten based on the value category of _lhs_ and whether `T` is known to
implement `IndirectIndexWith(I)`:

-   If `T` implements `IndirectIndexWith(I)`, the expression is rewritten to
    "`*((` _lhs_ `).(IndirectIndexWith(I).Addr)(` _index_ `))`".
-   Otherwise, if _lhs_ is an l-value, the expression is rewritten to "`*((`
    _lhs_ `).(IndexWith(I).Addr)(` _index_ `))`".
-   Otherwise, the expression is rewritten to "`(` _lhs_ `).(IndexWith(I).At)(`
    _index_ `)`".

`IndirectIndexWith` provides a blanket `final impl` for `IndexWith`:

```
final external impl forall
    [SubscriptType:! type, T:! IndirectIndexWith(SubscriptType)]
    T as IndexWith(SubscriptType) {
  let ElementType:! type = T.(IndirectIndexWith(SubscriptType)).ElementType;
  fn At[me: Self](subscript: SubscriptType) -> ElementType {
    return *(me.(IndirectIndexWith(SubscriptType).Addr)(index));
  }
  fn Addr[addr me: Self*](subscript: SubscriptType) -> ElementType* {
    return me->(IndirectIndexWith(SubscriptType).Addr)(index);
  }
}
```

Thus, a type that implements `IndirectIndexWith` need not, and cannot, provide
its own definitions of `IndexWith.At` and `IndexWith.Addr`.

### Examples

An array type could implement subscripting like so:

```
class Array(template T:! type) {
  external impl as IndexWith(like i64) {
    let ElementType:! type = T;
    fn At[me: Self](subscript: i64) -> T;
    fn Addr[addr me: Self*](subscript: i64) -> T*;
  }
}
```

And a type such as `std::span` could look like this:

```
class Span(T:! type) {
  external impl as IndirectIndexWith(like i64) {
    let ElementType:! type = T;
    fn Addr[me: Self](subscript: i64) -> T*;
  }
}
```

## Open questions

### Tuple indexing

It is not clear how tuple indexing will be modeled. When indexing a tuple, the
index value must be a constant, and the type of the expression can depend on
that value, but we don't yet have the tools to express those properties in a
Carbon API.

## Alternatives considered

-   [Different subscripting syntaxes](/proposals/p2274.md#different-subscripting-syntaxes)
-   [Multiple indices](/proposals/p2274.md#multiple-indices)
-   [Read-only subscripting](/proposals/p2274.md#read-only-subscripting)
-   [Rvalue-only subscripting](/proposals/p2274.md#rvalue-only-subscripting)
-   [Map-like subscripting](/proposals/p2274.md#map-like-subscripting)

## References

-   Proposal
    [#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
