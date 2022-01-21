# Subscripting

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [TODO: Initial proposal setup](#todo-initial-proposal-setup)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Multidimensional indexing](#multidimensional-indexing)
    -   [Read-only subscripting](#read-only-subscripting)
    -   [Rvalue-only subscripting](#rvalue-only-subscripting)
    -   [Map-like subscripting](#map-like-subscripting)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./new_proposal.py "TITLE"` to do new proposal setup.

1. Copy this template to `new.md`, and create a commit.
2. Create a GitHub pull request, to get a pull request number.
    - Add the `proposal` and `WIP` labels to the pull request.
3. Rename `new.md` to `/proposals/p####.md`, where `####` should be the pull
   request number.
4. Update the title of the proposal (the `TODO` on line 1).
5. Update the link to the pull request (the `####` on line 11).
6. Delete this section.

TODOs indicate where content should be updated for a proposal. See
[Carbon Governance and Evolution](/docs/project/evolution.md) for more details.

## Problem

Carbon needs a convenient way of indexing into objects like arrays and slices.

## Background

TODO: Is there any background that readers should consider to fully understand
this problem and your approach to solving it?

FIXME: Define "slice", discuss difference from arrays

## Proposal

FIXME: discuss grammar of subscript operator

FIXME: re-examine names, especially "rvalue" and "subscript" vs "index".

Carbon will support subscripting using the conventional `a[i]` syntax, and
user-defined types can support that syntax by implementing an appropriate
interface. When `a` is an lvalue, the result of subscripting will always be an
lvalue, but when `a` is an rvalue, the result can be an lvalue or an rvalue,
depending on which interface the type implements:

-   If subscripting an rvalue produces an rvalue result, as with an array, the
    type should implement `Subscriptable`.
-   If subscripting an rvalue produces an lvalue result, as with a slice, the
    type should implement `SubscriptableAsRValue`.

`SubscriptableAsRValue` is a subtype of `Subscriptable`, and subscript
expressions are rewritten to method calls on `SubscriptableAsRValue` if the type
is known to implement that interface, or to method calls on `Subscriptable`
otherwise. `SubscriptableAsRValue` provides a final blanket `impl` of
`Subscriptable`, which ensures that valid subscript operations cannot change
behavior depending on how much type information is available.

## Details

FIXME: Add this to docs/ as well

The subscripting interfaces are defined as follows:

```
interface Subscriptable(Subscript:! Type) {
  let ElementType:! Type;
  fn Index[me: Self](subscript: Subscript) -> ElementType;
  fn LValueIndex[addr me: Self*](subscript: Subscript) -> ElementType*;
}

interface SubscriptableAsRValue(Subscript:! Type) {
  extends Subscriptable(Subscript);
  let ElementType:! Type;
  fn RValueIndex[me: Self](subscript: Subscript) -> ElementType*;
}
```

An expression of the form "_lhs_ `[` _index_ `]`", where _lhs_ has type `T` and
_index_ has type `I`, is rewritten based on the value category of _lhs_ and
whether `T` implements `SubscriptableAsRValue(I)`: FIXME figure out implicit
conversion situation for `I`

-   If `T` implements `SubscriptableAsRValue(I)`, the expression is rewritten to
    "`*((` _lhs_ `).(SubscriptableAsRValue(I).RValueIndex)(` _index_ `))`".
-   Otherwise, if _lhs_ is an lvalue, the expression is rewritten to "`*((`
    _lhs_ `).(Subscriptable(I).LValueIndex)(` _index_ `))`".
-   Otherwise, the expression is rewritten to "`(` _lhs_
    `).(Subscriptable(I).Index)(` _index_ `)`".

On their own, these rules would oblige `SubscriptableAsRValue` types to define
three methods to support a single syntax. Worse, they would permit those types
to define those methods in inconsistent ways, which would mean that identical
code operating on identical objects could behave differently depending on how
much type information is available. To avoid those problems,
`SubscriptableAsRValue` provides a blanket `final impl` for `Subscriptable`:

```
final external impl [Subscript:! Type, T:! SubscriptableAsRValue(Subscript)]
    T as Subscriptable(Subscript) {
  let ElementType:! Type = T.(SubscriptableAsRValue(Subscript)).ElementType;
  fn Index[me: Self](subscript: Subscript) -> ElementType {
    return *(me.RValueIndex(subscript));
  }
  fn LValueIndex[addr me: Self*](subscript: Subscript) -> ElementType* {
    return me->RValueIndex(subscript);
  }
}
```

Thus, a type that implements `SubscriptableAsRValue` need not, and cannot,
provide its own definitions of `Index` and `LValueIndex`.

## Rationale based on Carbon's goals

TODO: How does this proposal effectively advance Carbon's goals? Rather than
re-stating the full motivation, this should connect that motivation back to
Carbon's stated goals for the project or language. This may evolve during
review. Use links to appropriate goals, for example:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

## Alternatives considered

### Multidimensional indexing

This proposal does not support multiple comma-separated indices (such as
`a[i, j]`), which is desirable for types like multidimensional arrays. We do
support syntax like `a[(i, j)]`, which is a single index whose type is a tuple,
but the extra parens are syntactically noisy. We could add those parens
implicitly, but that would effectively move the syntactic noise to the
implementation, even for the single-index case (for example
`impl Foo Subscriptable((i64))`). It should be much cleaner to make the
interfaces and their methods variadic, once we have a design for variadics.

### Read-only subscripting

This proposal does not provide an obvious path for supporting types like C++'s
`std::string_view` or `std::span<const T>`, whose subscripting operations expose
read-only access to their contents. It is tempting to try to extend this
proposal to support those use cases, both because of their inherent importance
and because it already has to deal with read-only versus read-write access in
order to support array rvalues.

However, there's a fundamental difference between the true immutability of
something like an array rvalue, and the contextual lack of mutable _access_
provided by something like `string_view`. While value categories can express the
former, they are not well-suited to expressing the latter. To address these use
cases, Carbon will probably need something like C++'s `const` type system, but
that should be largely orthogonal to this proposal.

### Rvalue-only subscripting

This proposal does not support subscripting operations that can't produce
lvalues. In particular, this means it does not support using subscript syntax to
form slices, as in Python's `a[i:j]` or Swift's `a[i...j]`. To support this, we
would need a separate pair of interfaces that return by value:

```
interface RValueSubscriptable(Subscript:! Type) {
  let ElementType:! Type;
  fn LValueIndex[addr me: Self*](subscript: Subscript) -> ElementType;
}

interface RValueSubscriptableAsRValue(Subscript:! Type) {
  extends RValueSubscriptable(Subscript);
  let ElementType:! Type;
  fn RValueIndex[me: Self](subscript: Subscript) -> ElementType;
}

final external impl [Subscript:! Type,
                     T:! RValueSubscriptableAsRValue(Subscript)]
    T as Subscriptable(Subscript) {
  let ElementType:! Type =
      T.(RValueSubscriptableAsRValue(Subscript)).ElementType;
  fn LValueIndex[addr me: Self*](subscript: Subscript) -> ElementType {
    return me->RValueIndex(subscript);
  }
}
```

Note that we still need a pair of interfaces, and a blanket final `impl` to
enforce consistency, because arrays and slices have different semantics in this
context as well: taking a slice of an rvalue array is invalid, because taking a
slice is equivalent to (and presumably implemented in terms of) taking an
address. On the other hand, taking a slice of an rvalue slice is valid and
should be supported.

We would likewise need to extend the rewrite rules for subscript syntax to
detect and use implementations of these interfaces. This should not lead to a
combinatorial explosion of cases, though; if `T` implements both
`Subscriptable(I)` and `RValueSubscriptable(I)`, the program should be
ill-formed due to ambiguity.

However, it's not clear if this would provide enough benefit to justify the
added complexity.

### Map-like subscripting

This proposal does not support subscripting operations that insert new elements,
in to a collection, as in C++'s `std::map` and `std::unordered_map`, because it
requires subscriptable types to support subscripting of rvalues, which are
immutable. We could support this with an additional interface for types that are
subscriptable only as lvalues, and a corresponding extension to the rewrite
rules.

However, it's debatable whether such insertion behavior is desirable; it has not
been a clear-cut success in C++. Code like `x = m[i];` looks like it reads from
`m`, and the fact that it can write to `m` is surprising, and easy for even
experienced programmers to overlook. Even for wary readers, it doesn't convey
the author's intent, because it's not clear whether the author assumed `i` is
present, or is relying on the implicit insertion. Furthermore, the implicit
insertion means that `x = m[i];` won't compile when `m` is const. On the other
hand, while it's relatively unsurprising that `m[i] = x;` might insert, that
insertion is also potentially inefficient, since it must default-construct a new
value before assigning `x` to it.

We could fix most of these problems by giving special treatment to syntax of the
form `m[i] = x;`, and defining separate methods for it:

```
interface Subscriptable(Subscript:! Type) {
  let ElementType:! Type;
  fn Index[me: Self](subscript: Subscript) -> ElementType;
  fn LValueIndex[addr me: Self*](subscript: Subscript) -> ElementType*;
  fn LValueIndexAssign[addr me: Self*](subscript: Subscript,
                                       element: ElementType) {
    (*me->LValueIndex(subscript)) = element;
  }
}

interface SubscriptableAsRValue(Subscript:! Type) {
  extends Subscriptable(Subscript, ElementType);
  let ElementType:! Type;
  fn RValueIndex[me: Self](subscript: Subscript) -> ElementType*;
  fn RValueIndexAssign[me: Self](subscript: Subscript, element: ElementType) {
    me->RValueIndex(subscript) = element;
  }
}

final external impl [Subscript:! Type, T:! SubscriptableAsRValue(Subscript)]
    T as Subscriptable(Subscript, ElementType) {
  ...
  fn LValueIndexAssign[addr me: Self*](subscript: Subscript,
                                       element: ElementType) {
    me->RValueIndexAssign(subscript, element);
  }
}
```

With this approach, expressions of the form `m[i] = x` would be rewritten to
call the appropriate `IndexAssign` method, while all other subscript expressions
are rewritten as in the primary proposal. However, this does add some complexity
to both the interfaces and the rewrite rules (for example, we presumably want
`(m[i]) = x` to be treated the same as `m[i] = x`), so I'm leaving this as a
potential future extension.

Alternatively, we could take a Rust-like approach, and rewrite subscript
expressions to use different interfaces depending on whether the usage expects
an lvalue. This would enable type authors to make `m[i] = x;` potentially
inserting while ensuring that `x = m[i];` is not. However, it would also mean
that `m[i].Method()` is potentially inserting if `Method` takes `me` by pointer,
and would have the same performance drawback as in C++.
