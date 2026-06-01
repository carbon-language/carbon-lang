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
which constraints the type implements:

-   If subscripting a value expression produces a value expression, as with an
    array, the type should implement `IndexValWith` and `IndexRefWith`. Types
    that implement both satisfy the `IndexWith` constraint.
-   If subscripting a value expression produces a durable reference expression,
    as with C++'s `std::span`, the type should implement `IndirectIndexWith`.

When `a` is a
[durable reference expression](/docs/design/values.md#durable-reference-expressions),
the result of subscripting is also a durable reference expression in either
case.

Any type that implements `IndirectIndexWith` automatically also implements
`IndexWith`.

Other behaviors can be accomplished by implementing the underlying interface
`IndexWithPrimitive`.

`IndirectIndexWith` overlaps and conflicts with `IndexValWith`, so a type can
implement at most one of those two constraints.

The `Ref` methods of `IndexRefWith` and `IndirectIndexWith`, which are used to
form durable reference expressions on indexing, return by `ref`. The
`IndexValWith` interface has an `At` method that returns by value.

## Details

A subscript expression has the form "_lhs_ `[` _index_ `]`". As in C++, this
syntax has the same precedence as `.`, `->`, and function calls, and associates
left-to-right with all of them.

Its semantics are defined in terms of the following form interface:

```carbon
package Core;

interface IndexWithPrimitive
    [implicit_into anchor Self:! Form]
    (implicit_into Subscript:! Form) {
  let implicit_from ResultForm:! Form;
  fn Op(bound self:? Self, subscript:? Subscript)
      ->? ResultForm;
}
```

The expression "_lhs_ `[` _index_ `]`" is rewritten to "_lhs_
`.(IndexWithPrimitive(formof(` _index_ `)).Op)(` _index_ `)`".

The named constraint `IndexWith` covers the case of indexing into an object that
owns the storage for its elements, like an array or C++'s `std::vector`:

```carbon
// `lhs[index]` is a value expression when `lhs`
// is a value expression.
constraint IndexValWith(SubscriptType:! type) {
  let ElementType:! type;

  require form(val Self) impls
     IndexWithPrimitive(form(val SubscriptType))
     where .ResultForm = form(val ElementType);
  alias At = LetSelf(IndexWithPrimitive(
      form(val SubscriptType))).Op;
}

// `lhs[index]` is a ref expression when `lhs`
// is a ref expression.
constraint IndexRefWith(SubscriptType:! type) {
  let ElementType:! type;

  require form(ref Self) impls
     IndexWithPrimitive(form(val SubscriptType))
     where .ResultForm = form(ref ElementType);
  alias Ref = RefSelf(IndexWithPrimitive(
      form(val SubscriptType))).Op;
}

// `lhs[index]` is a reference expression or value
// depending on the category of `lhs`.
constraint IndexWith
    [Self:! NoVarForm](SubscriptType:! type) {
  let ElementType:! type;
  match_first {
    extend require impls IndexRefWith(SubscriptType)
        where .ElementType = ElementType;
    extend require impls IndexValWith(SubscriptType)
        where .ElementType = ElementType;
  }
}
```

Note that `IndexWith` may be used as parameter's constraint, but can't be
directly implemented, since Carbon doesn't support implementing multiple
interfaces together (see
[leads issue #4566: Implementing multiple interfaces with a single `impl` definition](https://github.com/carbon-language/carbon-lang/issues/4566)
and
[proposal #5168: Forward `impl` declaration of an incomplete interface](https://github.com/carbon-language/carbon-lang/pull/5168)).

```carbon
// `lhs[index]` is always reference expression, even
// when `lhs` is a value.
constraint IndirectIndexWith(SubscriptType:! type) {
  let ElementType:! type;

  require form(val Self) impls
     IndexWithPrimitive(form(val SubscriptType))
     where .ResultForm = form(ref ElementType);
  alias Ref = LetSelf(IndexWithPrimitive(
      form(val SubscriptType))).Op;
}
```

Note that both `IndexValWith` and `IndexIndirectWith` require
`form(val Self) as IndexWithPrimitive(form(val SubscriptType))`, with different
result forms, and so conflict. However, a type that defines an `impl` of
`IndirectIndexWith` will also satisfy the `IndexValWith`, `IndexRefWith`, and
`IndexWith` constraints due to implicit conversions.

### Examples

An array type could implement subscripting like so:

```
class Array(template T:! type) {
  impl as IndexValWith(i64) {
    where ElementType = T;
    fn At(bound self, subscript: i64) -> val T;
  }
  impl as IndexRefWith(i64) {
    where ElementType = T;
    fn Ref(bound ref self, subscript: i64) -> ref T;
  }
}
```

And a type such as `std::span` could look like this:

```
class Span(T:! type) {
  impl as IndirectIndexWith(i64) {
    where ElementType = T;
    fn Ref(bound self, subscript: i64) -> ref T;
  }
}
```

## Alternatives considered

-   [Different subscripting syntaxes](/proposals/p002274-subscript-syntax-and-semantics.md#different-subscripting-syntaxes)
-   [Multiple indices](/proposals/p002274-subscript-syntax-and-semantics.md#multiple-indices)
-   [Read-only subscripting](/proposals/p002274-subscript-syntax-and-semantics.md#read-only-subscripting)
-   [Rvalue-only subscripting](/proposals/p002274-subscript-syntax-and-semantics.md#rvalue-only-subscripting)
-   [Map-like subscripting](/proposals/p002274-subscript-syntax-and-semantics.md#map-like-subscripting)

## References

-   Proposal
    [#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
-   Proposal
    [#2006: Values, variables, and pointers](https://github.com/carbon-language/carbon-lang/pull/2006)
-   Proposal
    [#5389: Generic across forms](https://github.com/carbon-language/carbon-lang/pull/5389)
