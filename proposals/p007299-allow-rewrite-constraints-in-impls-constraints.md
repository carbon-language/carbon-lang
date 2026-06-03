# Allow rewrite constraints in impls constraints

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
    -   [`where` clauses](#where-clauses)
    -   [`require` clauses](#require-clauses)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Allowing same-type constraints in impls constraints](#allowing-same-type-constraints-in-impls-constraints)
-   [Details](#details)
    -   [Coherence](#coherence)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Diagnosing constraints without a connection to the top-level `.Self` from a `require`](#diagnosing-constraints-without-a-connection-to-the-top-level-self-from-a-require)
        -   [Diagnosing the missing self type](#diagnosing-the-missing-self-type)
        -   [Decay the rewrite to a same-type constraint](#decay-the-rewrite-to-a-same-type-constraint)

<!-- tocstop -->

## Abstract

Allow rewrite constraints to be written in the nested facet type on the
right-hand side of `impls`. This applies to both `where` and `require` clauses.

In a `where` clause, each such constraint has a `.Self`, which will contain a
connection back to the top-level `.Self`, which allows them to always be found
through the top-level facet.

In a `require` clause, the `.Self` may not connect back to the top-level
`.Self`. This happens when the top-level `.Self` is not part of the self-type
nor an interface argument. Within a named constraint, we see the `require`
clause when identifying. If the `require` clause is not dependent on the
top-level `.Self` in its self-type or interface, it is required to be satisfied
as part of identifying the facet type. Within an interface, `require` clauses
are checked when the interface is `impl`d. If the `require` clause is not
dependent on the top-level `.Self` in its self-type or interface, it is required
to be satisfied for the `impl` to be valid.

When identifying a facet type, require any `impls` constraints that don't
constrain the self type to be satisfied immediately.

## Problem

### `where` clauses

We currently disallow writing a rewrite constraint or a same-type constraint on
the RHS of an `impls` constraint. They are only allowed in the "top" facet type,
which can act as the type of some facet.

For example, these are valid:

```carbon
// Valid, the first `where` contains a rewrite.
fn F(T:! Z where .Z1 = {});

// Valid, the first `where` contains a same-type.
fn F(T:! Z where .Z1 == {});
```

But these are not:

```carbon
// ❌ Error: rewrite on RHS of `impls`.
fn F(T:! type where .Self impls (Z where .Z1 = {}));

// ❌ Error: same-type on RHS of `impls`.
fn F(T:! type where .Self impls (Z where .Z1 == {}));
```

This limits the expresivity of a facet type without involving named constraints,
as it only allows rewrite and same-type constraints against interfaces extended
by the top-level facet. This prevents a user from writing a non-extending
relationship with an interface that also constrains an associated constant in
the interface.

In the below example, the user can constrain associated constants in `Z` since
the type of `T` extends `Z`. But they cannot constrain associated constants in
`Y`.

```carbon
// ❌ Error: same-type on RHS of `impls`.
fn F(T:! Z where .Z1 = {} and .Self impls (Y where .Y1 = {}));
```

### `require` clauses

In a similar way to `where` clauses, we allow a `require` clause to apply a
rewrite or same-type constraint to an associated constant in an interface only
if it _extends_ the interface, though all rewrite constraints are treated as-if
they are same-type constraints.

```carbon
constraint N {
  extend require impls Y where .Y1 = {};
}
fn F(T:! Z where .Z1 = {} and .Self impls N);
```

Rewrite constraints are currently rejected if the `require` clause does not
_extend_ the interface.

```carbon
constraint M {
  // ❌ Error: rewrite on RHS of non-extending `impls`.
  require impls Y where .Y1 = {};

  // ❌ Error: same-type on RHS of non-extending `impls`.
  require impls Y where .Y1 == {};
}
```

Regardless, it is possible to construct a named constraint that, when used,
produces a rewrite or same-type constraint with no connection to the top-level
facet. This prevents us from finding the constraint.

```carbon
constraint N(T:! type) {
  extend require impls Z where .Z1 = {};
}

// No relationship to `U` in `C impls Z where C.Z1 = {}`.
fn F(U:! type where C impls N(.Self), v: C.(Z.Z1)) {
  v as {};
}
```

The constraint `N` extends `Z`, which allows the rewrite constraint, but it is
applied to `C` and has no connection to `U` inside it. The type `C.(Z.Z1)` of
`v` should know its value has a same-type relationship to `{}`. However, since
there is no connection in `C.(Z.Z1)` to `U`, there is no way to find that
constraint in the type of `U`. We limit searching to components of the involved
types to avoid a global search.

## Background

Proposal
[#2173](https://github.com/carbon-language/carbon-lang/blob/358df53c482aeaefc8869ff36f8ef332ec34af3c/proposals/p002173-associated-constant-assignment-versus-equality.md)
introduces this restriction under
["Combining constraints with `require` and `impls`"](https://github.com/carbon-language/carbon-lang/blob/358df53c482aeaefc8869ff36f8ef332ec34af3c/proposals/p002173-associated-constant-assignment-versus-equality.md#combining-constraints-with-impl-as-and-is).

## Proposal

We propose the following, for both `where` and `require` clauses:

-   Remove the restriction on rewrite constraints on the right-hand side of
    `impls`.
-   Require any `impls` constraint that does not depend on the self-type to be
    satisfied as part of identifying the facet type. This includes that any
    rewrite constraints on the right-hand side of the `impls` are required to be
    known to be true.
-   If the rewritten associated constant is not in an interface that the top
    level facet extends, then the rewrite is treated as a same-type constraint.

When writing an `impl` as an `interface`, the `require` clauses within are
required to be satisfied. This will now include both those that do and don't
depend on the self-type.

### Allowing same-type constraints in impls constraints

We would like to allow same-type constraints in the same places as rewrite
constraints, however they differ in a key way. A member-access designator on the
LHS of a rewrite constraint is considered semantically as a direct member access
of the associated constant in the interface on the left-hand side of the nearest
`where`. But a member-access designator in a same-type constraint is treated
semantically as a member access into `.Self`. And `.Self` is rejected as
ambiguous on the right-hand side of an `impls` constraint. We have a separate
investigation into choosing an unambiguous value for `.Self` in this position,
the same as is effectively chosen for the left-hand side of a rewrite
constraint. This is in leads issue
[#7138](https://github.com/carbon-language/carbon-lang/issues/7138). Should the
`.Self` be made unambiguous there, this proposal can be read to apply equally to
same-type constraints as to rewrite constraints.

## Details

By allowing a rewrite constraint on the right-hand side of an `impls`
constraint, we allow a facet type to constrain its callers in more complex ways,
which can allow for less generic code. That is, generic code can specify
requirements on its callers that allow it to know more about its types, up to
knowing concrete types it is working with, which simplifies the generic
implementation.

The following generic function can know that any facet provided for `V` will
satisfy its requirement that `V.(Z.Z1).(Y.Y1)` will be the same type as `U`.
This means, for example, that it can call generic functions in `Y` that take a
value of type `.Y1` with a value of type `U`.

```carbon
interface Z {
  let Z1:! type;
}

interface Y {
  let Y1:! type;
  fn Make() -> Self;
  fn UseY1(y1: Y1);
}

fn F(U:! W, V:! Z where .Z1 impls (Y where .Y1 = U)) {
  let u: U = W.Make();
  let y: V.(Z.Z1) = V(Z.Z1).(Y.Make)();
  // Uses `u` as a value of type `Y.Y1`.
  x.UseY1(u);
}
```

Note that a constraint that `.Y1 impls W` would also establish a relationship
between `U` and `.Y1`, but would be insufficient to allow the call to `UseY1()`.
In order to call `UseY1()` the function would have to be rewritten to be
generic, with its parameter of given a symbolic type constrained by `W`.

### Coherence

One concern is that rewrites in this position may not be seen by code using the
symbolic value being rewritten. This can cause coherence issues since different
code paths see different values for the symbolic value. The current design has
this issue with named constraints that contain a `extend require impls` clause
with a rewrite constraint but which does not make use of `.Self` from the
caller, as described in the [Problem section](#require-clauses).

First, we known that the rewrite constraint itself contains a dependency on, or
reference to, the top-level `.Self` in its left-hand-side by the rules requiring
`.Self` to be present in each constraint in a `where` clause.

An `impls` clause is required to contain a reference to the top-level `.Self` in
either its left-hand-side or the extended constraints on its right-hand-side. A
nested `where` inside `impls` (which is a prerequisite to writing a rewrite
constraint) then introduces a new inner `.Self` with a value of the
left-hand-side of the `impls` and a type of the extended constraints on the
right-hand-side of the `impls`. This means the inner `.Self` introduced by
`where` inside an `impls` is required to be dependent on the top-level `.Self`.

A rewrite constraint inside the `impls` clause will appear on the
right-hand-side of that nested `where`, and its left-hand-side must be a
member-access designator into that nested `.Self` which implies it also contains
a reference to the top-level `.Self`. We note a same-type constraint in the
nested `where` must also contain the inner `.Self` in order to be valid, which
means they similarly contain a reference to the top-level `.Self`.

This means in the facet type `Z where .Z1 impls (Y where .Y1 = U)`, the rewrite
constraint `.Y1 = U` contains a reference to the top-level `.Self` in its
left-hand-side. In this example, as `.Self.(Z.Z1).(Y.Y1) = U`.

The expanded left-hand-side of a rewrite constraint represents an absolute path
to the associated constant. As described above, the path is either rooted at or
contains a top-level `.Self` reference in the path.

When used to constrain a facet, the facet type is identified with that facet
replacing the `.Self`. For example with the binding
`T:! Z where .Z1 impls (Y where .Y1 = U)`, performing member access into `T`
identifies the type of `T` with `.Self` being replaced by `T`. That gives a
unique path to `.Y1` through `T` which is `T.(Z.Z1).(Y.Y1)`. As such, the access
of the associated constant `.Y1` through this path contains `T` which has a
facet type that contains the rewrite of `.Y1`. This allows any use of the
associated constant to also see its rewritten value by searching the type of
facets in the path to the associated constant.

This is similar to the fact that in an impl lookup `T as Z`, if a facet's type
provides the witness for the lookup, the facet will be part of the lookup query,
either in the source type or in the target interface's generic arguments.

The same holds for rewrite constraints in a named constraint that are visible in
the identified facet type, if the rewrite constraint contains a reference to
`.Self` passed into the named constraint through the self-type (for example
`C(.Self) impls N`) or through a generic parameter (for example
`C impls N(.Self)`). Of concern is a rewrite that does not contain a reference
to `.Self`. After identifying, any such constraint will not depend on the
self-type, and will be required to be satisfied immediately from another source.
If satisfied, it implies a rewrite of the associated constant can be found from
a facet or type in the path to the associated constant. Then a use of the
associated constant can perform the same search and find the rewritten value.

## Rationale

This proposal advances Carbon's goal of
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
by allowing generic code to introduce constraints that reduce their internal
complexity.

We now ensure that rewrites introduced by an `extend require impls` clause are
always visible to users of the rewritten associated constant, by searching the
path to the constant. This avoids a global search for the rewritten value, which
advances our
[low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).

can introduce a rewrite constraint that does not have any dependence on the
self-type in an identified facet type. Such a rewrite may not be found by uses
of the associated constant.

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

We considered the following options to handle this case.

#### Diagnosing the missing self type

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

#### Decay the rewrite to a same-type constraint

We could accept the constraint but decay the rewrite constraint to a same-type
constraint. However, this doesn't change the nature of the problem, where the
constraint has no connection to the top-level `.Self`. Given a conversion of a
value of type `C.(Z.Z1)` to `{}`, we need to find the same-type constraint in
`U`. But `U` is not present in the source type, so we do not find the same-type
relationship.

```carbon
fn F(U:! type where C impls N(.Self), v: C.(Z.Z1)) {
  // Rejected. Cannot find the constraint `C.(Z.Z1) = {}` in `N` from the
  // type of `v`, since it has no reference to `U`.
  v as {};
}
```
