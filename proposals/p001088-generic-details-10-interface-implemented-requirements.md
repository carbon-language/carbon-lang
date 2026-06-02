# Generic details 10: interface-implemented requirements

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1088)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Less strict about requirements with `where` clauses](#less-strict-about-requirements-with-where-clauses)
    -   [Don't require `observe...is` declarations](#dont-require-observeis-declarations)

<!-- tocstop -->

## Problem

This proposal is to add the capability for an interface to require other types
to be implemented, not just the `Self` type. The interface-implemented
requirement feature also has some concerns:

-   If the interface requirement has a `where` clause, there are
    [concerns](#less-strict-about-requirements-with-where-clauses) about being
    able to locally check whether impls satisfy that requirement.
-   A function trying to make use of the fact that a type implements an
    interface due to an interface requirement, or a blanket impl, may require
    the compiler perform a search that we don't know will be bounded.

## Background

The first version of interface-implemented requirements for interfaces was
introduced in proposal
[#553: Generics details part 1](https://github.com/carbon-language/carbon-lang/pull/553).

## Proposal

This proposal adds two sections to the
[generics details design document](/docs/design/generics/details.md):

-   [Interface requiring other interfaces revisited](/docs/design/generics/details.md#interface-requiring-other-interfaces-revisited)
-   [Observing a type implements an interface](/docs/design/generics/details.md#observing-a-type-implements-an-interface)

## Rationale based on Carbon's goals

This proposal advances these goals of Carbon:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    The motivation for this expressive power of interface requirements comes
    from discussions about how to achieve symmetric behavior with interfaces
    like `CommonTypeWith` from
    [proposal #911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development):
    The requirement that the source provide the proof of any facts that would
    require a recursive search using `observe` declarations means that the
    expense of that search is avoided except in the case where there is a
    compiler error. If the search is successful, the results of the search can
    be copied into the source, and afterward the search need not be repeated.

## Alternatives considered

### Less strict about requirements with `where` clauses

We could allow
[requirements with `where` constraints](/docs/design/generics/details.md#requirements-with-where-constraints)
to be satisfied by implementations that could be specialized, as long as the
constraints were still satisfied. Unfortunately, this is not a condition that
can be checked locally. Continuing the example from that section, consider four
packages

-   A package defining the two interfaces

    ```
    package Interfaces api;
    interface A(T:! Type) {
      let Result:! Type;
    }
    interface B(T:! Type) {
      impl as A(T) where .Result == i32;
    }
    ```

-   A package defining a type that is used as a parameter to interfaces `A` and
    `B` in blanket impls:

    ```
    package Param api;
    import Interfaces;
    class P {}
    external impl [T:! Type] T as Interfaces.A(P) where .Result = i32 { }
    // Question:Is this blanket impl of `Interfaces.A(P)` sufficient
    // to allow us to make this blanket impl of `Interfaces.B(P)`?
    external impl [T:! Type] T as Interfaces.B(P) { }
    ```

-   A package defining a type that implements the interface `A` with a wildcard
    impl:

    ```
    package Class api;
    import Interfaces;
    class C {}
    external impl [T:! Type] C as Interfaces.A(T) where .Result = bool { }
    ```

-   And a package that tries to use the above packages together:

    ```
    package Main;
    import Interfaces;
    import Param;
    import Class;

    fn F[V:! Interfaces.B(Param.P)](x: V);
    fn Run() {
      var c: Class.C = {};
      // Does Class.C implement Interfaces.B(Param.P)?
      F(c);
    }
    ```

Package `Param` has an implementation of `Interfaces.B(Param.P)` for any `T`,
which should include `T == Class.C`. The requirement in `Interfaces.B` in this
case is that `T == Class.C` must implement `Interfaces.A(Param.P)`, which it
does, and `Class.C.(Interfaces.A(Param.P).Result)` must be `i32`. This would
hold using the blanket implementation defined in `Param`, but the wildcard impl
defined in package `Class` has higher priority and sets the associated type
`.Result` to `bool` instead.

The conclusion is that this problem would only be detected during
monomorphization, and could cause independent libraries to be incompatible with
each other even when they work separately. These were significant enough
downsides that we wanted to see if we could live with the restrictions that
allowed local checking first. We don't know if developers will want to declare
their parameterized implementations `final` in this situation anyway, even with
[the limitations on `final`](/docs/design/generics/details.md#libraries-that-can-contain-a-final-impl).

This problem was discussed in
[the #generics channel on Discord](https://discord.com/channels/655572317891461132/941071822756143115/941089885475962940).

### Don't require `observe...is` declarations

We could require the Carbon compiler to do a search to discover all interfaces
that are transitively implied from knowing that a type implements a set of
interfaces. However, we don't have a good way of bounding the depth of that
search.

In fact, this search combined with conditional conformance makes the question
"is this interface implemented for this type" undecidable
[in Rust](https://sdleffler.github.io/RustTypeSystemTuringComplete/). Note: it
is possible that
[the acyclic rule](/docs/design/generics/details.md#acyclic-rule) would avoid
this problem in Carbon for blanket impls, but it doesn't apply to interface
requirements.

This problem was observed in
[a discussion in #typesystem on Discord](https://discord.com/channels/655572317891461132/708431657849585705/938167784565792848).
