# Semantic Identity and Order-Dependent Resolution for Rewrite Constraints

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5689)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
    -   [Canonicalization](#canonicalization)
-   [Alternatives considered](#alternatives-considered)
    -   [Order independence](#order-independence)
    -   [Syntactic equivalence](#syntactic-equivalence)

<!-- tocstop -->

## Abstract

The rules for rewrite constraint resolution say that when there are multiple
assignments to the same associated constant in a facet type, they must all be
identical. But it does not clearly identify whether that means syntactically
identical or semantically. And if semantically, whether that is before or after
substituting from the RHS of other rewrite constraints.

## Problem

Previously we had been working with the intention of requiring syntactical
equivalence for all rewrite constraints within a facet type that assign to the
same associated constant. However this is highly problematic for implementation
in the toolchain, since we lose syntactic information when we create the semir
of the rewrite constraints. In particular, once a facet type is fully
constructed, all syntactic information is lost, as individual instructions are
evaluated and their canonical constant values are what is stored in the facet
type in order to make facet types themselves canonical.

This problem becomes most visible when applying
[the `&` operation](https://docs.carbon-lang.dev/docs/design/generics/details.html#combining-interfaces-by-anding-facet-types)
to combine two facet types with rewrite constraints of the same associated
constant, such as: `(I where .X = ()) & (I where .X = .Y and .Y = ())`. Here the
facet type on the right will be resolved to `I where .X = () and .Y = ()` before
applying the `&` operation, so we have lost syntactic knowledge that `.X` was
being assigned `.Y`. That makes things inconsistent if we consider it an error
to give `.X` two differently-written values in a single facet type, such as
`I where .X = () and .X = .Y and .Y = ()`.

Therefore we would like to consider a semantic identity requirement instead,
however there are soundness issues, implementation-specific inconsistencies, or
diagnostic inconsistencies, unless we carefully specify how rewrite constraints
are applied through substitution when checking for conflicting assignments and
cycles between rewrite constraints.

For example, given `I where .Y = () and .Y = .X and .X = .Y`:

-   If `.X = .Y` is resolved to `.X = ()` first, then the two assignments to
    `.Y` are not in conflict and can be considered identical, and no cycle is
    found.
-   If `.Y = .X` is resolved to `.Y = .Y` first, however, we find a cycle and
    diagnose this as an error.
-   If `.X = .Y` is resolved to `.X = .X` by using the second constraint for
    `.Y` instead of the first one, we also find a cycle and diagnose an error.

That leaves us with the question of whether this facet type contains a cycle
`.Y = .Y` after being resolved, or contains two identical assignments of
`.Y = ()` and one of `.X = .Y`.

One question is whether we should have any reliance on ordering of rewrite
constraints in resolution. In other words, whether it's okay that reordering the
rewrite constraints changes whether a cycle is found or not.

If ordering should not affect the determination of cycles, then we have a choice
for what to look for:

-   Should we always use an ordering that finds a cycle, if such an ordering
    exists?
-   Should we always use an ordering that avoids a cycle, if such an ordering
    exists?

If we are okay with the ordering of rewrite constraints affecting the detection
of cycles, then we need to precisely specify how the ordering works in order to
make implementations consistent:

-   In what order are the RHS of rewrite constraints resolved?
-   When substituting for a reference to another rewrite constraint, in what
    order do we choose a constraint to substitute from?

## Background

-   [Design: Rewrite constraint resolution](https://github.com/carbon-language/carbon-lang/blob/67b67af7a61b9cea1d47c3a1009f78ac00790a47/docs/design/generics/appendix-rewrite-constraints.md?plain=1#L195-L304).
-   [Open discussion notes from 2025-06-17](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.qti4vn50zwy).
-   Conversation on
    [this proposal's PR](https://github.com/carbon-language/carbon-lang/pull/5689),
    in particular
    [this comment thread](https://github.com/carbon-language/carbon-lang/pull/5689#discussion_r2164881833).
-   Discord discussion from #generics-and-templates on
    [2025-06-17](https://discord.com/channels/655572317891461132/941071822756143115/1384632814196101171)
    and
    [2025-06-25](https://discord.com/channels/655572317891461132/941071822756143115/1387499929056055469).

## Proposal

We propose to allow ordering of rewrite constraints to affect cycle detection,
with the following orderings specified when resolving:

-   Rewrite constraints should be resolved left-to-right.
-   When a reference to another rewrite constraint is found, it should be
    substituted with the first matching rewrite constraint found from
    left-to-right.
-   When combining two facet types `LHS & RHS`, the LHS facet type's rewrite
    constraints are ordered before the RHS facet type's constraints when
    resolving the combined facet type. Effectively, the rewrite constraints from
    the RHS are concatenated onto the constraints from the LHS, then the facet
    type is resolved.

## Rationale

We believe that order dependence is in line with the design direction of the
language overall. In particular, this aligns with the
[information accumulation principle](/docs/project/principles/information_accumulation.md).
The principle says:

> If a program attempts to use information that has not yet been provided, the
> program is invalid.

This suggests that looking ahead for a rewrite constraint to avoid a cycle is an
invalid operation.

However, it also states that:

> Carbon programs are invalid if they would have a different meaning if more
> information were available.

This might seem to suggest that the meaning can not change when the order in
which code is written changes. But here we take _meaning_ to refer to the
resolved state of all associated constants when there are no errors present.
Using ordering to avoid cycles and prevent error diagnostics is a common
practice in Carbon, such as with the use of forward declarations. This proposal
makes the ordering effect whether a cycle is found and diagnosed, but in the
absence of a cycle, the ordering does not affect the resulting values of the
associated constants.

Using a prescribed ordering to resolve and find cycles simplifies the
implementation, avoiding the need to consider all `n!` orderings of `n`
constraints. It also simplifies explaining the language rules, as you can step
through the resolving algorithm without excessive branching choices, much like
you would step through lines of code in a debugger.

Using a left-to-right order is consistent with how Carbon is parsed and executed
generally. If there were multiple statements written on the same line, they
would also be executed left-to-right.

### Canonicalization

A key part of toolchain implementation is the ability to canonicalize constant
values, including facet types. This means that once resolved, the order of the
rewrite constraints must be able to be effectively lost, as different orderings
would all result in the same canonical facet type.

After being resolved, a facet type rewrite constraints have two key properties:

-   Each associated constant appears at most once on the LHS of a rewrite
    constraint. There are no duplicate assignments.
-   Each associated constant that appears on the RHS of a rewrite constraint
    does not appear on the LHS of any rewrite constraint. There are no
    relationships between rewrite constraints.

When combining two facet types with `LHS & RHS` we said that the rewrite
constraints of the LHS come before the rewrite constraints of the RHS. Since the
LHS and RHS are each fully resolved before being combined, canonicalization
requires that the rewrite constraints within the LHS and the RHS can be ordered
in arbitrary ways without changing whether a cycle is found.

Assume that we have two orderings such that one produces a cycle, and one does
not. That means we have a branching point from an associated constant, so there
are two rewrite constraints for that same constant, so one must appear on the
LHS and one on the RHS of the `&` operator. Now their order is fixed, as we have
specified that all rewrite constraints from the LHS come before those from the
RHS. So we will always choose the same constraint when replacing that associated
constant, and consistently find a cycle or not, regardless of how constraints
are ordered within the LHS and RHS facet types.

## Alternatives considered

### Order independence

This has two possibilities: eagerly finding cycles or eagerly avoiding cycles.
In either case, this complicates the toolchain implementation without making the
language more expressive. The user can always write constraints in a way that
does not produce a cycle in a left-to-right ordering if the constraints are
valid.

Eagerly finding cycles produces errors when unnecessary, which seems more
hostile than needed. Eagerly avoiding cycles could be seen as being helpful, as
we find a way to complete the concrete assignments whenever possible. In that
case, given a cycle of rewrite constraints, adding a concrete value for any
associated constant in the cycle, written anywhere in the facet type, breaks the
cycle. Whereas with left-to-right ordering, the concrete value needs to be the
first rewrite constraint for the associated constant.

With eager cycle breaking, this cycle `I where .Y = .X and .X = .Y` is broken by
adding `.X = ()` anywhere in the facet type:
`I where .Y = .X and .X = .Y and .X = ()`. For a human reading left to right, we
still read a cycle, but have to keep reading to understand why it's not.

With left-to-right cycle breaking, the same expressivity is kept, but the
concrete value must be written before other rewrites of the same associated
constant: `I where .Y = .X and .X = () and .X = .Y`. Now for a human reading
from left to right, we can read that `.Y = () and .X = ()` and can disregard
other later assignments to `.X`.

### Syntactic equivalence

We could choose to diagnose `I where .X = () and .X = .Y and .Y = ()` as a
conflicting assignment to `.X`, since the RHS of the two assignments are written
differently, even though the resolved value of the RHS can be found to be `()`
in both of them.

This is perhaps the simplest rule for humans to understand, but requires
maintaining syntactic information throughout the toolchain, complicating the
implementation and is at odds with canonicalization.

Given the following example:

```
(I where .X = .Y and .Y = ()) & (I where .X = ())
```

The combined facet type combines `.X = .Y` and `.X = ()` which are syntactically
different. But the facet type on either side of the `&` operator are resolved
and canonicalized. Canonicalization requires that `I where .X = .Y and .Y = ()`
is the same type as `I where .X = () and .Y = ()`, but when combined with
`I where .X = ()` one violates the syntactic rule and one does not.

Thus we would need to only apply syntactic identity within a single facet type,
but use a semantic identity when combining facet types. This makes for a more
inconsistent experience and implementation.
