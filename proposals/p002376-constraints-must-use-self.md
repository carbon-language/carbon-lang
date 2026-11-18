# Constraints must use `Self`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2376)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Require `impl as` constraints in an `interface` or `constraint` definition to
mention `Self` implicitly or explicitly. Require `where` clauses to refer to
`.Self` directly, or through a designator like `.Foo`.

## Problem

When trying to implement constraints in Carbon Explorer, we
[came up with an example](https://discord.com/channels/655572317891461132/941071822756143115/986054308179103765)
that raised questions:

```carbon
interface A {}
interface B {}
external impl forall [T:! Type] T as A where T is B {}
```

There were multiple possible interpretations for what the meaning of that
`where` clause was.

-   It could be equivalent to
    `external impl forall [T:! Type where .Self is B] T as A {}`. That is, this
    introduces an implementation of A for only those T that satisfy the where
    condition.
-   It could be equivalent to `external impl forall [T:! Type] T as A {}` but
    invalid if there is no `impl forall [T:! Type] T as B`. that is, this
    requires an implementation of `B` to exist for all `T`.
-   It could be equivalent to `external impl forall [T:! Type] T as A & B {}`.
    That is, this introduces an implementation of `B` for all `T`.
-   It could be invalid for various reasons.

That advantage of making this construction invalid is that it would force the
code into a form with a clearer meaning.

Other cases suggested that constraints that were modifying other types were in
general surprising,
[for example](https://discord.com/channels/655572317891461132/941071822756143115/986061766226214932):

```carbon
fn F[A:! Type, B:! Type, C:! Type where A == B](a: A, b: B, c: C);
```

would be better written as:

```carbon
fn F[A:! Type, B:! Type where A == .Self, C:! Type](a: A, b: B, c: C);
```

so the relationship between types `A` and `B` would be established from their
two declarations, not later modified by the declaration of `C`.

In summary, we ended up with a number of reasons to say a `where` clause should
be a constraint on the type being modified:

-   We prefer there only be [one way](/docs/project/principles/one_way.md) to
    write constraints. We believe that the examples that don't meet this
    restriction can always be rewritten to a form that meets this restriction.
-   We believe that after the rewrite, there is less ambiguity about what the
    code means.
-   We think it is valuable that the constraints on a type are complete when the
    type's declaration is complete.

We found similar restrictions are valuable for `impl as` constraints in an
`interface` or `constraint` definition. The restriction that they always involve
the `Self` type means that the search that compiler has to do to find relevant
constraints is limited to a finite number of definitions. Furthermore, without
this restriction, the set of interfaces known to implement a type would change
depending on which interfaces definitions are imported and known to be
satisfied, which is a coherence problem.

This restriction also allows interfaces and named constraints to be used while
incomplete, which allows some use cases that involve circular references,
including self reference. The logic goes like this:

-   we want to limit when information from a constraint is found,
-   to increase the cases where we don't need to look in a constraint,
-   so constraints are allowed to be incomplete in those cases.

Proposal [#2347](https://github.com/carbon-language/carbon-lang/pull/2347) lists
conditions when we want to allow constraints to be incomplete.

## Background

There are a number of earlier proposals related to or modified by this proposal:

-   [#553: Generics details part 1](https://github.com/carbon-language/carbon-lang/pull/553)
    introduced `impl as` restrictions in interfaces and named constraints.
-   [#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818)
    introduce `where` constraints.
-   [#1013: Generics: Set associated constants using `where` constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
    switched to using `where` constraints in `impl` declarations to specify
    associated constants.
-   [#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
    allowed forward declaration of interfaces and named constraints, explicitly
    supporting incomplete interfaces and named constraints beyond when they were
    being defined.
-   [#2107: Clarify rules around `Self` and `.Self`](https://github.com/carbon-language/carbon-lang/pull/2107)
    established some rules around `Self` and `.Self`, which this proposal adds
    to.
-   [#2347: What can be done with an incomplete interface](https://github.com/carbon-language/carbon-lang/pull/2347)
    clarifies what can be done with an incomplete interface or named constraint.
    Those rules rely on this proposal to be implementable.

## Proposal

`where` clauses must use a designator, either `.Self` or `.Foo` for some member
`Foo`. The designator may be used directly, or supplied as an argument to a
type, interface, or named constraint used in the `where` clause, as in these
examples:

-   `Container where .ElementType = i32`
-   `Type where Vector(.Self) is Sortable`
-   `Addable where i32 is AddableWith(.Result)`

`impl as` declarations in interfaces and named constraints must always involve
`Self`:

-   Can be the implicit `Self` when no type is specified, as in `impl as ...`,
    or the equivalent declarations with `Self` declared explicitly, as in
    `impl Self as ...`
-   Can be an argument to a type. The type can be what is to the left of the
    `as`, as in `impl Vector(Self) as ...`, or a type argument to the interface
    or constraint, as in `impl Vector(i32) as AddWith(Vector(Self))`.
-   Can be a parameter to the interface or constraint to the right of the `as`,
    as in `impl T as Bar(Self)`.

When the compiler looks to see if any constraints imply that an impl exists, the
only place it needs to look are the places that involve the type the impl is for
(`Self`). This means the compiler never needs to look in forward-declared (or
otherwise incomplete) constraints that don't involve that type. This applies
recursively. This allows incomplete interfaces and named constraints as
described in proposal
[#2347](https://github.com/carbon-language/carbon-lang/pull/2347).

This solves a problem: when doing impl lookup, what is the set of imlps that you
can look up? There may be an infinite set of constraints reachable through
interfaces, but with this rule, you only need to consider a finite subset.

## Details

The ["Generics: Details" design document](/docs/design/generics/details.md) has
been updated with this proposal. It includes clarification in the
[conditional conformance section](/docs/design/generics/details.md#conditional-conformance)
that an `impl` in a `class` definition can only be for the type being defined.

## Rationale

These restrictions are in support of the
["prefer providing only one way to do a given thing" principle](/docs/project/principles/one_way.md),
by reducing the number of equivalent ways of expressing a constraint.

As described in the [problem section](#problem), these restrictions make code
[easier to read and understand](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
by avoiding confusing or ambiguous constructions.

These restrictions reduce the search the compiler needs to perform to find
relevant constraints during impl lookup, in support of
[fast and scalable development](/docs/project/goals.md#fast-and-scalable-development).

## Alternatives considered

The main alternative we considered, was not imposing these restrictions. We
decided these restrictions were a good idea in these conversations:

-   [#generics-and-templates on 2022-06-13](https://discord.com/channels/655572317891461132/941071822756143115/986061509815844864)
-   [Open discussion on 2022-10-12](https://docs.google.com/document/d/1tEt4iM6vfcY0O0DG0uOEMIbaXcZXlNREc2ChNiEtn_w/edit#heading=h.q7afaawbc5k)
-   [#generics-and-templates 2022-10-24](https://discord.com/channels/655572317891461132/941071822756143115/1034198851059466292)
-   [2022-10-24 open discussion](https://docs.google.com/document/d/1tEt4iM6vfcY0O0DG0uOEMIbaXcZXlNREc2ChNiEtn_w/edit#heading=h.hb5qukkw7d3l)

The advantages of this proposal are outlined in the [problem section](#problem).

The main disadvantage of this proposal that
[we considered](https://discord.com/channels/655572317891461132/941071822756143115/986063589016215614)
is that it removes the option to use another name for the type than `.Self`. The
concern was that `.Self` might be seen as an advanced feature that is difficult
to understand, or it might be longer.
