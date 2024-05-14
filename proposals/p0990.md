# Generics details 8: interface default and final members

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/990)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Defaulting to less specialized impls](#defaulting-to-less-specialized-impls)
    -   [Allow default implementations of required interfaces](#allow-default-implementations-of-required-interfaces)
    -   [Don't support `final`](#dont-support-final)

<!-- tocstop -->

## Problem

Rust has found that allowing interfaces to define default values for its
associated entities is valuable:

-   Helps with evolution by reducing the changes needed to add new members to an
    interface.
-   Reduces boilerplate when some value is more common than others.
-   Addresses the gap between the minimum necessary for a type to provide the
    desired functionality of an interface and the breadth of API that user's
    desire.

Carbon would benefit in the same ways.

## Background

Rust supports specifying defaults for
[methods](https://doc.rust-lang.org/book/ch10-02-traits.html#default-implementations),
[interface parameters](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading),
and
[associated constants](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants-examples).

## Proposal

This proposal defines both how defaults for interface members are specified in
Carbon code as well as final interface members in the
[generics details design doc](/docs/design/generics/details.md#interface-defaults).

## Rationale based on Carbon's goals

This proposal advances these goals of Carbon:

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software):
    Final members of interfaces can avoid some dynamic dispatch overhead.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    Defaults simplify adding new members to an interface without having to
    simultaneously update all impls of that interface.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Defaults both reduce boilerplate, making code easier to read and write.
    Marking interface members as `final` makes the code more predictable to
    users of that member.

## Alternatives considered

### Defaulting to less specialized impls

Rust has observed
([1](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#default-impls),
[2](http://aturon.github.io/tech/2015/09/18/reuse/)) that interface defaults
could be generalized into a feature for reusing definitions between impls. This
would involve allowing more specific implementations to be incomplete and reuse
more general implementations for anything unspecified.

However,
[they also observed](http://smallcultfollowing.com/babysteps/blog/2016/09/29/distinguishing-reuse-from-override/):

> [To be sound,] if an impl A wants to reuse some items from impl B, then impl A
> must apply to a subset of impl B's types. ... This implies we will have to
> separate the concept of "when you can reuse" (which requires subset of types)
> from "when you can override" (which can be more general).

This is a source of complexity that we don't want in Carbon. If we do eventually
support inheritance of implementation between impls in Carbon, it will do this
by explicitly identifying the impl being reused instead of having it be
determined by their specialization relationship.

### Allow default implementations of required interfaces

Here are the reasons we considered for not allowing interfaces to provide
default implementations of interfaces they require:

-   This feature would lead to incoherence unless types implementing
    `TotalOrder` also must explicitly implement `PartialOrder`, possibly with an
    empty definition. The problem arises since querying whether `PartialOrder`
    is implemented for a type does not require that an implementation of
    `TotalOrder` be visible.
-   It would be unclear how to resolve the ambiguity of which default to use
    when two different interfaces provide different defaults for a common
    interface requirement.
-   It would be ambiguous whether the required interface should be external or
    [internal](/docs/design/generics/terminology.md#extending-an-interface)
    unless `PartialOrder` is implemented explicitly.
-   There would be a lot of overlap between default impls and blanket impls.
    Eliminating default impls keeps the language smaller and simpler.

The rules for blanket impls already provide resolution of the questions about
coherence and priority and make it clear that the provided definition of the
required interface will be external.

### Don't support `final`

There are a few reasons to support `final` on associated entities in the
interface:

-   Clarity of intent when default methods are just to provide an expanded API
    for the convenience of callers, reducing the uncertainty about what code is
    called.
-   Matches the functionality available to base classes in C++, namely
    non-virtual functions.
-   Could reduce the amount of dynamic dispatch needed when using an interface
    in a `DynPtr`.

The main counter-argument is that you could achieve something similar using a
`final` impl:

```
interface I {
  fn F();
  final fn CallF() { F(); }
}
```

could be replaced by:

```
interface IImpl {
  fn F();
}
interface I {
  extends IImpl;
  fn CallF();
}
final impl (T:! IImpl) as I {
  fn CallF() { F(); }
}
```

This is both verbose and a bit awkward to use since you would need to
`impl as IImpl` but use `I` in constraints.
