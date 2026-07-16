# Ensure `where` requirements in named constraints are visible to lookups

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7299)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Diagnosing constraints without a connection to the top-level `.Self` from a `require`](#diagnosing-constraints-without-a-connection-to-the-top-level-self-from-a-require)

<!-- tocstop -->

## Abstract

Require rewrite and same-type constraints that do not depend on `.Self` to be
satisfied when a facet type is identified, since those constraints may not be
found later.

## Problem

It is possible to construct a named constraint that, when used, produces a
rewrite or same-type constraint with no connection to the top-level facet. This
prevents us from finding the constraint.

```carbon
constraint N(T: type) {
  extend require impls Z where .Z1 = {};
}

// No relationship to `U` in `C impls Z where C.Z1 = {}`.
fn F(generic U: type where C impls N(.Self)) {
  {} as C.(Z.Z1);
}
```

The `extend require` statement constrains `Self` which in this case is `C`.
Since it does not use `T`, it has no connection to `.Self` in the facet type of
`U`. That makes the constraint impossible to find in queries that don't involve
`U` since we will not know to search its facet type. We limit searching to
components of the involved types to avoid a global search.

## Background

-   The
    [rules for rewrite constraints](/docs/design/generics/appendix-rewrite-constraints.md)
    in the design.
-   The rules are introduced by proposal
    [#2 for the rewrite left-hand-side that 173](https://github.com/carbon-language/carbon-lang/blob/358df53c482aeaefc8869ff36f8ef332ec34af3c/proposals/p002173-associated-constant-assignment-versus-equality.md).

## Proposal

When identifying a facet type, collect all rewrite and same-type constraints
found while also collecting `impls` constraints from the facet type and any
named constraints it depends on. Require that any rewrite or same-type
constraint which does not depend on `.Self` is satisfied immediately, in order
to identify the facet type.

During identify, rewrite constraints are collected as rewrite constraints when
they are part of the top-level facet type being identified, or come from an
uninterrupted chain of `extend requires` statements. Otherwise, rewrite
constraints that come from a `requires` statement are collected as same-type
constraints.

Rewrite and same-type constraints that do not depend on `.Self` must be
immediately satisfied by performing an impl lookup with that constraint as a
condition. In the following example, the identified facet type of `T` contains a
rewrite `C.(Z(U).Z1) = {}` and the same-type constraint `C.(Z(U).Z2) == ()`. So
an impl lookup for `C as Z(U) where .Z1 = {} and .Z2 == ()` must be satisfied to
identify the type of `T`. In this example, the lookup succeeds by finding a
witness with the required constraints in the facet type of `U`.

```carbon
interface Z(V: type) {
  let Z1: type;
  let Z2: type;
}

constraint N(T2: type, U2: type) {
  extend require impls Z(U2) where .Z1 = {} and .Z2 == ();
}

fn F(U: type where C impls Z(.Self) where .Z1 = {} and .Z2 == (),
     T: type where C impls N(.Self, U));
```

## Rationale

We now ensure that constraints introduced by a `requires` clause in a named
constraint are either visible for lookups, or a redundant with existing rules.
This avoids a global search for the rewritten value, which advances our
[low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).

## Alternatives considered

### Diagnosing constraints without a connection to the top-level `.Self` from a `require`

In the following, the rewrite constraint `C.(Z.Z1) = {}` has no connection to
the top-level facet `U` since it does not contain any reference to the top-level
`.Self`.

```carbon
class C;

constraint N(T:! type) {
  require impls Z where .Z1 = {};
}

// No relationship to `U` in `C impls Z where C.Z1 = {}`.
fn F(U:! type where C impls N(.Self)) {}
```

We could diagnose during identifying the type of `U` that the `where` contains a
constraint `C.(Z.Z1) = {}` which doesn't involve the self type. This would
reduce the expressiveness of the generics system and produce an error for some
uses of `N` but not for others.

```carbon
class G(T:! type);

// Accepted.
fn F(U:! type where G(.Self) impls N(.Self)) {}

// Rejected.
fn F(U:! type where G({}) impls N(.Self)) {}
```
