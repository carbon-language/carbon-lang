# Reject nested `where` expressions to prevent ambiguous `.Self`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7437)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow nested `.Self` but it refers to the top level facet value](#allow-nested-self-but-it-refers-to-the-top-level-facet-value)
    -   [Allow ambiguous `.Self` but disambiguate based on context](#allow-ambiguous-self-but-disambiguate-based-on-context)
    -   [Disallow use of ambiguous `.Self` through name lookup](#disallow-use-of-ambiguous-self-through-name-lookup)
    -   [Disallow nested `where` syntactically but allow it from eval](#disallow-nested-where-syntactically-but-allow-it-from-eval)

<!-- tocstop -->

## Abstract

The `.Self` facet is introduced by `where` and it refers to the binding or the
left-hand-side of an `impls` that comes before the `where`, with its type being
the facet type immediately on the left-hand-side of the `where`. This allows
multiple `.Self` to be introduced within a facet type, each shadowing the other.

When these `.Self` become part of the canonical facet type, they are not
possible to differentiate, making it impossible to later replace `.Self`
coherently. To avoid having to recover from this situation, we propose to
prevent the ability to introduce an ambiguous `.Self` entirely, by disallowing
nesting `where` on the right-hand-side of another `where` expression.

## Problem

A `where` that introduces a shadowing `.Self` referring the same value (the same
binding, the same left-hand-side of `impls`) but a different type, gives more
visibility to the facet type being defined. But a `where` that introduces a
shadowing `.Self` referring to a different value creates an _ambiguous `.Self`_.

The Carbon reference toolchain is built on a compile-time evaluation model,
where all types have a canonicalized constant value in order to allow cheap type
comparison through the constant value identifier. In a canonical type
representation all `.Self` references look the same, as they can not refer to
their surrounding context. Any valid way of constructing the same facet type
must result in the same canonical constant, including building a facet type from
other non-local facet types.

These `.Self` references must then be replaced to form a more specific type
constant once the target facet they refer to is known. For an ambiguous `.Self`,
we don't know when we're supposed to replace the `.Self`, which can cause us to
replace it incorrectly. Doing so leads to incoherence as the internal
representation no longer matches the meaning of the code as written. Or it leads
to crashes in the toolchain implementation as inconsistency develops.

In the following example, `.Self` in the argument to `Z` refers to `T` and must
be replaced by `T` eventually. But there is a second `.Self` nested in the
rewrite constraint for `.Z1` that refers to whatever the `(Y where...)` facet
type ends up constraining. In order to use `T.Z1` we need the type of `T` with
all `.Self` references replaced by `T`.

```carbon
interface Z { let Z1: type; }
interface Y { let Y1: type; }

fn F(generic T: Z(.Self) where .Z1 = (Y where .Self.(Y.Y1) == ()),
     generic U: T.Z1);
```

The `.Self` in the same-type constraint looks the same as the `.Self` in the
argument to `Z` so they are both replaced with `T`, which is incorrect. The
second one should eventually be replaced with `U`, since that is the facet which
it is constraining. It is also incorrect because we don't know that `T`
implements `Y`.

Building on the previous example, in this call to `G`, we end up with `W` being
constrained by the facet type `T.Z1`. `T.Z1` refers to the type of `T` where
`.Self` is replaced by `T`, which is evaluated to `Y where T.(Y.Y1) == ()`. This
facet type incorrectly involves the symbolic type `T`, which `G` has no generic
parameter for, so it cannot be made concrete in a specific of `G`.

```carbon
fn G(generic V: type, generic W: V);

fn F(generic T: Z(.Self) where .Z1 = (Y where .Self.(Y.Y1) == ())) {
  G(T.Z1);
}
```

## Background

-   [#7138](https://github.com/carbon-language/carbon-lang/issues/7138): Leads
    issue "How to treat ambiguous .Self?"
-   [p2173](/proposals/p002173-associated-constant-assignment-versus-equality.md):
    Proposal "Associated constant assignment versus equality"

## Proposal

We disallow the introduction of a `.Self` that could be ambiguous by rejecting
the use of `where` nested in the right-hand-side of another `where` expression,
after evaluation. This rejects non-extend constraints in a facet type from
appearing inside the non-extend constraints of another facet type.

```carbon
// Allowed, no nested `where`.
fn F(generic T: Z(.Self));

// Allowed, no nested `where`.
fn F(generic T: Z where .Self impls Y);

// Allowed, `where` is nested on the left-hand-side of another `where`.
fn F(generic T: (Z where .Self impls Y) where .Self impls X);

// Rejected, `where` is nested on the right-hand-side of another `where`.
// - This introduces a non-extend impls constraint inside a non-extend impls constraint.
fn F(generic T: Z where .Self impls (Y where .Self impls X));

// Allowed, the nested facet type has no nested `where`.
fn F(generic T: Z where .Z1 = Y(.Self));

// Rejected, `where` is nested on the right-hand-side of another `where`.
// - This introduces a non-extend impls constraint inside a rewrite constraint.
fn F(generic T: Z where .Z1 = (Y where .Self impls X));

// Accepted, no nested `where`.
musteval fn GetY() -> type {
  return Y;
}
fn F(generic T: Z where .Self impls GetY());

// Rejected, `where` is nested on the right-hand-side of another `where`.
// - This introduces a rewrite constraint inside a non-extend impls constraint.
musteval fn GetYWithRewrite() -> type {
  return Y where .Y1 = {};
}
fn F(generic T: Z where .Self impls GetYWithRewrite());
```

Prior to this proposal, ambiguous `.Self` was already disallowed, but we
required diagnosing the _use_ of a `.Self` that is ambiguous. This limits
implementation strategies for the language. So we now reject the facet types
that allow the introduction of an ambiguous `.Self`.

In an implementation that uses compile-time evaluation, a facet type can be
constructed through eval which can introduce a `.Self` from another non-local
context, such as from the return of a compile-time function call or from an
`alias`. Once this is done, the `.Self` becomes ambiguous in the resulting facet
type.

The language retains the same expressivity in facet types as before. When a
facet type would have used an ambiguous `.Self` in a nested facet type, the
nested facet type can be moved to a named constraint where `.Self` can be used
without ambiguity internally, and the outer `.Self` can be passed in as a
generic argument.

For each rejected case above, we can demonstrate moving the nested `where` to a
named constraint to make it valid:

```carbon
// Now accepted, the `where` in `N` is no longer nested inside another `where`.
constraint N {
  // This can also be written as two separate `require` statements, for `U`
  // and `X`.
  require impls U where .Self impls X;
}
fn F(generic T: Z where .Self impls N);

// Now accepted, the `where` in `N` is no longer nested inside another `where`.
constraint N {
  require impls Y where .Self impls X;
}
fn F(generic T: Z where .Z1 = N);

// Now accepted, the `where` in `N` after evaluation is no longer nested inside
// another `where`.
musteval fn GetYWithRewrite() -> type {
  return Y where .Y1 = {};
}
constraint N {
  require impls GetYWithRewrite();
}
fn F(generic T: Z where .Self impls N);
```

This leads to a recommended best practice: Only use `where` expressions in
contexts where they are immediately used to constrain something. If you factor
out a `where` expression, turn it into a named `constraint` rather than using an
`alias` or `eval fn`, as named constraints can be composed more freely.

## Details

The name `.Self` is introduced by either of:

-   A compile-time binding, for its type: `T: Z(.Self)`.
-   The `where` keyword, in a facet type: `type where .Self impls Z(.Self)`.

Generally, if it is shadowing, the `.Self` introduced by the first `where`
keyword in a facet type always refers to the same value as the the compile-time
binding - to the binding itself. So it does not introduce an ambiguous `.Self`.
Though it is possible for evaluation to nest that `where` into the
right-hand-side of another `where` by composing facet types.

A `where` on the right-hand-side of another `where` expression can introduce a
new value for its `.Self` in multiple ways when writing a facet type in a
constraint's expression:

-   In an `impls` constraint, `... where C impls (... where ...)` the nested
    `where` introduces `.Self` referring to the type on the left-hand-side of the
    `impls`. In this example it would refer to `C`.
-   The right-hand-side of a rewrite constraint, `.X = (... where ...)` can
    contain an arbitrary compile-time expression. That expression may be a facet
    type with a `where` expression. That `where` expression would introduce a
    `.Self` that refers to an unknown facet value until the associated constant
    being rewritten was used as the constraining facet type for some facet value.
    In this example, if a type `T` was constrained by the `.X` associated
    constraint, the nested `.Self` would refer to that `T`.
-   In a same-type constraint, it is possible to write an arbitrary compile-time
    expression on either side of the `==` operator. The expression may contain a
    facet type with a `where` expression. That `where` expression would introduce
    a `.Self` that does not refer to any facet value.

It is possible for `.Self` to be introduced syntactically, by writing a `where`
expression directly on the right-hand-side of another `where`. These are more
straightforward to reject, and can be rejected by pointing directly to the
invalid `where` keyword.

```carbon
fn F(generic T: Z where .Z1 = (Y where .Y1 = {}))
//                               ^^^^^
//                               Invalid `where` on the right-hand-side of
//                               another `where`.
```

A `where` can also be introduced through evaluation. For example, a reference to
an `alias` does not contain a `where` in its syntax, but the alias may evaluate
to a facet type that contains a `where`, and this may contain `.Self` references
that become ambiguous in the larger facet type. To prevent this, we also reject
non-extend constraints in a facet type from appearing inside the non-extend
constraints of another facet type. Non-extend constraints all come from the
right-hand-side of a `where` expression. When they appear in a nested facet type
that is inside a non-extend constraint, it implies a `where` nested on the
right-hand-side of another `where`, and the potential for an ambiguous `.Self`
in the nested facet type.

```carbon
alias A = Y where .Y1 = {};

fn F(generic T: Z where .Z1 = A)
//                            ^
//                            Contains an invalid `where` on the right-hand-
//                            side of another `where`.
```

## Rationale

The proposal advances the principle of
[Low context-sensitivity](/docs/project/principles/low_context_sensitivity.md).
By avoiding ambiguity in the syntax, we avoid requiring context to disambiguate.

The proposal also advances our goal of
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
and the sub-goal to "Design features to be simple to implement" in particular.
We are reducing the complexity of behaviour required for `.Self` in order to give
more implementation choices, and to make the constant evaluation model possible.

## Alternatives considered

### Allow nested `.Self` but it refers to the top level facet value

The design requires that constraints all constrain the "current type" in some
way. In a facet type, that means they contain a reference to `.Self`, and that
`.Self` refers to the "current type". This alternative would have nested `.Self`
refer to the top-level type being constrained, which would prevent writing
constraints other than rewrite constraints on the right-hand-side of a nested
`where`.

If a nested facet type on the right-hand-side of a rewrite constraint contains
`.Self` and it refers to the top level facet value we defer the ambiguity until
the rewrite constraint's value is used elsewhere.

In this example the facet type of `U` contains a `.Self.(Z.Z2)` that would refer
to `T` and `.Self.(Y.Y1)` that would refer to `U`. In the canonical facet type,
these `.Self` references are ambiguous.

```carbon
fn F(generic T: Z where .Z1 = (Y where .Z2 == ()),
     U: T.Z1 where .Y1 == ())
```

While the design does not allow writing `.Self` on the left-hand-side of a
rewrite constraint, our implementation based on compile-time evaluation does
insert `.Self` into its model of the rewrite constraint. In particular, in this
example, we have two `.Self` in the left-hand-side of the rewrite constraint:
The `.Z1` is treated as a reference to an associated constant `Z1` in the
interface `Z(.Self)` for the facet `.Self`.

```carbon
fn F(generic T: Z(.Self) where .Z1 = ());
```

And we can nest the rewrite constraint on the right-hand-side of another
`where`:

```carbon
fn F(generic T: Y where .Y1 impls (Z(.Self) where .Z1 = ()));
```

This alternative would have all `.Self` references refer to `T` but the
semantics of the language require the one of the two `.Self` references in `.Z1`
to refer to `T.Y1`. It is a reference to an associated constant `Z1` in the
interface `Z(.Self = T)` for the facet `.Self = .Y1`. This creates a
contradiction in the implementation, and leaves us with the same problem as we
began with.

### Allow ambiguous `.Self` but disambiguate based on context

Leads issue [#7138](https://github.com/carbon-language/carbon-lang/issues/7138)
originally proposed 4 different ways to allow an ambiguous `.Self` and resolve
it to a unique value (options 1, 1b, 3, 4).

To do so requires "disambiguating" the `.Self` references, which we attempted to
do numerous times with various strategies over 2025 and 2026:

-   [2026-06-04](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.4yc1be253c33#heading=h.xiezkm442a2e):
    Numbering distance of `.Self` through generic arguments?
-   [2026-04-28](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.h6lecvcwu214#heading=h.twhpk59ki1vo):
    Change how `.Self` is introduced?
-   [2026-04-13](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.h6lecvcwu214#heading=h.pjdpgj7z1v1j):
    Restrict use of `.Self` in impls constraints?
-   [2025-10-16](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.ay4or45dwwxd):
    Attempting to use different instruction types for `.Self`.
-   [2025-09-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4r024sqi9b4b):
    Replace `.Self` by modifying side-car data instead of modifying binding types?
-   [2025-09-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rg3417ebho7q):
    Numbering distance of `.Self` from the binding/facet it refers to?
-   [2025-08-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.8dh7ckcudkx2):
    Special FacetAccessType for `.Self` usage to avoid toolchain cycles?
-   [2025-07-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.7urbxcq23olv):
    Numbering strategies for disambiguating `.Self` uses.
-   [2025-06-30](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4qd5dkyfn2k3):
    Determining `.Self` distances.

Each numbering strategy has failed, as
[we can always construct](https://github.com/carbon-language/carbon-lang/issues/7138#issuecomment-4709736280)
a facet type with ambiguous `.Self` and then
[use that facet type as a generic argument](https://discord.com/channels/655572317891461132/941071822756143115/1514762686045487154).
And the use of the facet type in the new generic context does not have the
information available to it needed to disambiguate the `.Self` references.

The complexity involved in tracking and replacing `.Self` becomes arbitrarily
complex, with `.Self` representing free variables in
[a lambda calculus](https://discord.com/channels/655572317891461132/941071822756143115/1514765334488547459)
embededed in a facet type.

Even detecting if a `.Self` is ambiguous from context is challenging since
evaluation can distort the context. In
[this example](https://github.com/carbon-language/carbon-lang/issues/7138#issuecomment-4752416144),
the facet type contains a `.Self` reference which is ambiguous, but which we
can't determine it is so from local context alone:

```carbon
interface I {}
class C(T: type);

class X {
  impl type as Core.MulWith(X) where .Result = type {
    eval fn Op(unused self, _: X) -> type {
       return I where .Self == i32;
    }
  }
}

alias x = {} as X;

fn G(generic T: type where C(.Self) impls type * x) {}
```

Because of these issues, we chose to pursue disallowing nested `where` to
prevent the _introduction_ of an ambiguous `.Self` instead of rejecting the
_use_ of an ambiguous `.Self`.

### Disallow use of ambiguous `.Self` through name lookup

We can use surrounding context to determine if a `.Self` name would be
ambiguous, and then diagnose the use of it and prevent the name lookup. However
because evaluation can introduce `.Self` into a facet type too, through an alias
or a function call, local context is not enough. We need to find the ambiguous
`.Self` in the final facet type, but by then they are ambiguous so we can't
tell.

### Disallow nested `where` syntactically but allow it from eval

If we disallow writing `where` in the right-hand-side of another `where`, then
we prevent writing an ambiguous `.Self` directly. While a `.Self` can also be
introduced from eval, at this time the `.Self` can not be ambiguous, as its type
can not be `.Self`-dependent. This comes from:

-   A call to an `eval fn` is not evaluated until all inputs are concrete. So
    there can't be a `.Self` in the inputs. That means the function can not return
    a facet type that involves `.Self` on the left-hand-side of the `where`. So
    any use of `.Self` on the right-hand-side can be disamiguated.
-   An alias has no input parameters, so it can not evaluated to a facet type that
    involves `.Self` on the left-hand-side of the `where`. So any use of `.Self`
    on the right-hand-side can be disamiguated.

As such, we could say that eval is allowed to add a nested `where` since the
facet type will be concrete, and not `.Self`-dependent (has no dependency on the
`.Self` from the outer facet type).

However we do plan to add generic parameters to aliases, which will allow
passing `.Self` as an argument, and thus allow aliases to evaluate to a
`.Self`-dependent facet type. To avoid churn on this rule, we just designed it
to also handle parameterized aliases, and ban nesting a `where` on the
right-hand-side of another `where` in all cases, even through evaluation.
