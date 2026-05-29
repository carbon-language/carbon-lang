# Replace keyword `is` with `impls`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2483)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [`T as C`](#t-as-c)
    -   [`T: C`](#t-c)

<!-- tocstop -->

## Abstract

Use the keyword `impls` instead of `is` when writing a `where` constraint that a
type variable needs to implement an interface or named constraint.

What was previously (provisionally) written:

```
fn Sort[T:! Container where .ElementType is Ordered](x: T*);
```

will now be written:

```
fn Sort[T:! Container where .ElementType impls Ordered](x: T*);
```

## Problem

The `is` keyword has been used as a placeholder in `where` constraints
expressing that a type variable is required to implement an interface or named
constraint. It has been an open question what word should be used in that
position since its original adoption in
[#818](https://github.com/carbon-language/carbon-lang/pull/818). Since then,
reasons to use a different word in this position have been discovered:

-   The word "is" is unspecific and could represent many different
    relationships.
-   This specific relationship is not symmetric.
-   We potentially want to use `is` as a keyword for another purpose.
-   The precedent for `is` came from Swift where `x is T` means "`x` has the
    type `T`". With the changes to Carbon generic semantics, particularly
    [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360),
    that is increasingly a poor fit for what we mean by this condition.

## Background

The `is` keyword as a constraint operator was introduced in
[#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818),
along with the open question about how to spell it.

The choice of `is` in that proposal followed
[`is` being Swift's type check operator](https://docs.swift.org/swift-book/LanguageGuide/TypeCasting.html#ID340),
where `x is T` is `true` if `x` has type `T`. Note that there are differences
between the `is` operator in Swift and what we have used it for in Carbon. In
Swift, it is used to test whether a value dynamically has a specific derived
type, when you have a value of a base class type and are using inheritance. In
Carbon:

-   it is used on types instead of values;
-   it is about conformance to an interface (the equivalent of Swift's
    protocols), and not about inheritance; and
-   it is resolved at compile time.

## Proposal

Use the keyword `impls` instead of `is` when writing a `where` constraint that
at type variable needs to implement an interface or named constraint. The
specific changes are included in
[the same PR as this proposal](https://github.com/carbon-language/carbon-lang/pull/2483).

## Rationale

This proposal is working towards Carbon's
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal:

-   Examples read naturally using "implements" in the place of the `impls`
    keyword, commonly matching how a comment would describe the constraint.
-   More clearly communicates the relationship between the two sides and that
    the relationship is not symmetric.
-   If a function has a `where T impls C` constraint that is not satisfied for
    some calling type `T`, the fix is for the caller to add an `impl T as C`
    definition.

## Alternatives considered

One concern with using `impls` is the potential for confusion with the plural of
`impl`, meaning "implementations," rather than acting as the verb "implements."
We hope to mitigate that concern by avoiding use of "impls" to mean anything
other than `impls` in our documentation. For example, we would say "`impl`
declarations" instead of "`impl`s".

Alternatives were considered in
[#2495: Keyword to use in place of `is` in `where`...`is` constraints](https://github.com/carbon-language/carbon-lang/issues/2495).
A number of alternatives were considered:

-   `T is C`
-   `T isa C`
-   `T impls C`
-   `T implements C`
-   `T ~ C`
-   `T: C`
-   `T as C`
-   `T impl C`
-   `T impl as C`
-   `impl T as C`

The reasons against `is` were outlined in the [problem](#problem) and
[rationale](#rationale) sections. Reasons against other alternatives:

-   `isa` seems (much) too rooted in inheritance.
-   `implements` is long but otherwise fine. This is a specific place where
    being verbose has worrisome negative impact.
-   `~` is already being considered for something a bit more fitting --
    bidirectional convertibility of our type "equality" constraints.
-   `impl` seems a bit clunky, and surprising to see in this position when it
    usually isn't.
-   `impl as` seems even more clunky
-   `impl T as C` also seems even more clunky, and would be confusing with the
    rules around rewrite constraints (different here from the use of
    `impl T as C` in a type).

In general there were concerns that the alternatives were confusing and risk
appearing to mean something other than what it does.

### `T as C`

The keyword `as` received more consideration:

-   Already a thing, and names the _facet_ that is required to exist.
-   Already used in a facet-like-but-not-facet context for
    `impl T as C { ... }`.
-   Short, easy to read, etc.

It had some disadvantages, though:

-   Doesn't connect readers as directly and effectively to the need for an
    `impl` to satisfy the constraint.
-   Doesn't read as nicely in context:
    `T:! C where C.ElementType as AddWith(i32)`. This was a big consideration in
    deciding against using `as`.
-   Underlying the above disadvantage, it doesn't fit into the model of a
    boolean expression that should be true. Instead, it is a cast that should be
    _possible_ or _meaningful_, which is somewhat different from the rest of the
    things in a `where` clause.
    -   However, "rewrite" constraints don't quite fit this model either.
    -   When using `==` constraints, they don't actually imply any boolean
        expression that would return true. In fact, at least my understanding is
        that the `T == U` constraint _could_ be written as a boolean expression
        but it would return _false_ even when the constraint holds.

### `T: C`

`T: C` matches how Swift and Rust write this, and is similar to the way you'd
write the same constraint in an ordinary declaration, `T:! I`. This syntactic
similarity is also a liability due to the differences in semantics when used in
a `where` clause: the `where` clause doesn't make the names from `I` available
in `T`, and it doesn't propagate `where .A = B` rewrites from `I` to `T`.

There's also some concern that the use of `:` would make parameter lists hard to
read when they contain embedded `where` clauses, like
`T:! Container where .ElementType: Printable, U:! OtherConstraint`.
