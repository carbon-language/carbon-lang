# Interface variance (`implicit_into` and `implicit_from`)

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8186)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Summary](#summary)
-   [Proposal](#proposal)
    -   [`Core.ConvertsInto`](#coreconvertsinto)
    -   [`implicit_into` interface parameter variance](#implicit_into-interface-parameter-variance)
        -   [Implied constraints](#implied-constraints)
        -   [Modifiers for coherence](#modifiers-for-coherence)
        -   [Facets create adapters](#facets-create-adapters)
    -   [`implicit_from` variance](#implicit_from-variance)
        -   [`Core.ConvertsFrom`](#coreconvertsfrom)
        -   [Covariant](#covariant)
    -   [`match_first` in named constraints](#match_first-in-named-constraints)
    -   [Interaction between `anchor`, `exact`, and named constraints](#interaction-between-anchor-exact-and-named-constraints)
-   [Details](#details)
    -   [Coherence modifier examples](#coherence-modifier-examples)
    -   [`impl` lookup for an interface with variant parameters](#impl-lookup-for-an-interface-with-variant-parameters)
    -   [Conversions when calling a function argument](#conversions-when-calling-a-function-argument)
-   [Future work](#future-work)
    -   [Impl prioritization](#impl-prioritization)
    -   [Allow more variance](#allow-more-variance)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Other ways to determine variance in interfaces](#other-ways-to-determine-variance-in-interfaces)
    -   [Support implicit conversions in impls](#support-implicit-conversions-in-impls)
    -   [Infer `anchor` in `impl` declarations](#infer-anchor-in-impl-declarations)

<!-- tocstop -->

## Abstract

We introduce contravariant (`implicit_into`) and covariant (`implicit_from`)
variance on `Core.ExtType` interface parameters and associated constants,
expressed in terms of new `Core.ConvertsInto` and `Core.ConvertsFrom`
constraints. This allows `impl` lookup to perform implicit extended-type
conversions when specified by the interface, replacing the `like` mechanism on
`impl`s. To preserve coherence and allow disambiguation, we introduce `anchor`
and `exact` modifiers, adapter creation for facets of variant interfaces,
disabling implied constraints for variant interfaces, and `match_first` blocks
in named constraints.

## Problem

Prior to this proposal, implicit conversions during `impl` lookup (such as for
operator overloading) were handled by marking types with `like` on individual
`impl` declarations, as introduced in
[proposal #1144](https://github.com/carbon-language/carbon-lang/pull/1144). That
mechanism has significant limitations:

-   It cannot provide implicit conversions when an interface is used as a
    generic constraint without a blanket `impl` — for example, calling a
    generic callable parameter `f` constrained to accept `i32` with an integer
    literal `7` fails because `7` has type `Core.IntLiteral`, not `i32`.
-   It operates only on types rather than extended types (`Core.ExtType`), so it
    cannot handle category conversions (such as calling a `val self` `impl` on a
    `ref` expression, or converting a `ref` return to a `val` return in
    indexing).
-   It only supports contravariant input parameters, not covariant return
    extended types (such as an associated constant `.ResultExtType`).

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal depends on pending proposals also split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
    (`Core.ExtType`, `exttype(...)`, `.type`, `fwd`, `where .type = ...`), and
-   [proposal #8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185)
    (extended type interfaces, extended types in `impl`s, named extended type
    constraints).

This proposal provides the variance mechanism used by pending proposals also
split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8187: Extended-type call interface and `call` syntax](https://github.com/carbon-language/carbon-lang/pull/8187)
    (`Core.CallPrimitive` and `call(...)`), and
-   [proposal #8188: Extended-type operator, conversion, and member binding interfaces](https://github.com/carbon-language/carbon-lang/pull/8188)
    (`Core.IndexWithPrimitive`, `Core.MemberBind`, `Core.AsPrimitive`,
    `Core.ImplicitAsPrimitive`, and other primitive operator interfaces).

This proposal also builds on and updates:

-   [Proposal #1144: Generic details 11: operator overloading](https://github.com/carbon-language/carbon-lang/pull/1144)
    (whose `like` mechanism is replaced by this proposal).
-   [Proposal #5337: Interface extension and `final impl` update](p005337-interface-extension-and-final-impl-update.md#impl-selection-algorithm)
    (the `impl` selection algorithm updated by this proposal).

## Summary

In short, this proposal introduces:

-   `Core.ConvertsInto(PET: Core.ExtType)` and
    `Core.ConvertsFrom(RET: Core.ExtType)` which are `Core.ExtType` constraints:
    `ConvertsInto(PET)` matches all extended types that can be converted to a
    parameter with extended type `PET`, and `ConvertsFrom(RET)` matches all
    extended types that can be converted from a return with extended type `RET`.
-   `implicit_into` which may be used to mark `Core.ExtType` interface
    parameters, including the `Self` parameter of an extended type interface.
    Those parameters are called "variant", specifically "contravariant." The
    arguments to a query in that position may be converted, and are described by
    the `Core.ConvertsInto` constraint.
    -   An `impl` of such an interface can have multiple type structures,
        depending on whether the types in the query match or need to be
        converted.
    -   An `impl` may use `anchor` or `exact` to limit the conversions allowed,
        which can be needed to satisfy the orphan rule. `anchor` may also modify
        `implicit_into` on interface parameters.
    -   Implied constraints are not added for interfaces with variant
        parameters.
    -   When normally we would create a facet of the argument type for the
        interface, if that interface has variant parameters, we instead create
        an adapter for the argument type that implements the interface. This
        adapter performs the conversion between the argument and parameter
        extended types, and allows the implementations of the interface to
        differ from the original type without introducing coherence concerns.
-   `implicit_from` which similarly may be used to mark an interface parameter
    or associated constant. These parameters are also "variant," specifically
    "covariant." These allow conversions in the opposite direction as
    `implicit_into`, and are described by the `Core.ConvertsFrom` constraint.
-   `match_first` blocks inside named constraints to prioritize between
    requirements that overlap due to variance.
-   removal of the `like` operator on `impl`s.

## Proposal

### `Core.ConvertsInto`

The `Core` package defines the `ExtType` facet type
`ConvertsInto(PET: ExtType)`, which says which `ExtType`s describe a binding
that would be allowed to convert to the binding described by `PET`. So
`exttype(val i32)` is a value of `ConvertsInto(exttype(val i64))` since there is
an implicit conversion from `i32` to `i64`. More precisely, if we create a
function `F` with `PET` as the extended type of its parameter, and `y` a binding
with extended type `SRC` (also an `ExtType` value), as in:

```carbon
fn F(fwd x: PET);
fn G(fwd y: SRC) {
  F(fwd ~y);
}
```

then `SRC` is a value of `ConvertsInto(PET)` if and only if the call
`F(fwd ~y)` is legal.

Note that this depends on the meaning of the planned move operator `~`. In this
case, we need:

-   `~v` when `v` is a value parameter gives you the value of `v` before the
    expression;
-   `~r` when `r` is a `ref` parameter gives you a reference expression that
    references whatever `r` did before the expression;
-   `~i` when `i` is a `var` parameter gives you an initializing expression that
    initializes an object to the state that `i` was in before the expression.

Defining `ConvertsInto` in terms of forwarding a parameter `fwd y: SRC` by way
of `fwd ~y` ensures that the notional `impl`s and
[adapters](#facets-create-adapters) for
[`implicit_into`](#implicit_into-interface-parameter-variance) work by
definition. Alternatively, `ConvertsInto` could be framed in terms of passing an
expression of extended type `SRC` directly to `F` (such as `F(fwd Y())` for
`fn Y() -> fwd SRC;` when `SRC` is a valid extended return type). We expect the
upcoming definition of moves to make these two formulations equivalent, because
input-to-input forwarding `fwd ~y` of a `var` parameter `y` will be able to
reuse `y`'s caller-allocated `var` storage without requiring `Core.Move`.

As an example, `ConvertsInto(exttype(var i64))` includes the facets of
`exttype(val i32)`, `exttype(ref i16)`, and `exttypeof(3)`, since this code is
legal:

```carbon
fn F(var x: i64);

fn G1(y1: i32) {
  F(~y1);
}

fn G2(ref y2: i16) {
  F(~y2);
}

fn G3(fwd y3: exttypeof(3)) {
  F(~y3);
}
```

Also, the facet of `exttype(var i64)` is also a member of
`ConvertsInto(exttype(var i64))`. In fact `PET` is a member of
`ConvertsInto(PET)` for all `PET`.

Note that this is different from extended type facet type conversions (in
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)),
which are about converting between one extended type facet type and another.
This is about what conversions would be allowed to a binding described by a
particular value of `Core.ExtType`.

This is also different from `ImplicitAs` (and `ImplicitAsPrimitive` in
[proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)):

-   With `ImplicitAs`, the parameter specifies the destination type, not the
    destination extended type, and the `impl` determines the resulting extended
    type. `ConvertsInto` uses `ImplicitAs` to determine if types convert, but in
    combination with other interfaces, such as `Copy` and `Move`, which
    determine if other aspects of the extended type such as category convert.
    For example, a reference to an object of a derived class implicitly converts
    to a reference to its base class. It additionally `ConvertsInto` a _value_
    of the base class, since we can follow that type conversion with a
    [value acquisition](/docs/design/values.md#value-acquisition).
-   You can't implement `ConvertsInto`, it may only be used as a constraint.

### `implicit_into` interface parameter variance

An extended type interface parameter (including `Self` for an extended type
interface) may be marked with the `implicit_into` keyword binding modifier, as
in this example:

```carbon
interface I(implicit_into Self: Core.ExtType,
            implicit_into P: Core.ExtType) {
  fn F(fwd self: Self, fwd a: P);
}

impl exttype(ref u64) as I(exttype(var i64)) {
  fn F(ref self: u64, var a: i64) {
    self += a;
  }
}
```

This means `impl` lookup for queries involving `I` will consider `impl`
declarations that don't match exactly, unless the corresponding extended type
values in the `impl` declarations have
[a mark that requires them to match, such as `exact`](#modifiers-for-coherence).
In particular, in those positions a query `X` will match `Y` if `X` is a value
of type `ConvertsInto(Y)`. For example, the `impl` defined above acts as if it
was replaced by a collection of `impl`s for purposes of `impl` lookup:

```carbon
// `ForwardTo` is the `F` member of the `impl`
// above. It is equivalent to:
//   fn (ref self: u64, var a: i64) {
//     self += a;
//   }
alias ForwardTo =
  (exttype(ref u64) as I(exttype(var i64))).F;

impl forall [PET1: ConvertsInto(exttype(ref u64)),
             PET2: ConvertsInto(exttype(var i64))]
    PET1 as I(PET2) {
  fn F(fwd self: PET1, fwd a: PET2) {
    // This is allowed, by the definition of
    // `ConvertsInto` and that the extended type of `self`
    // of `ForwardTo` is `ref _: u64` and
    // the extended type of its argument is `var _: i64`.
    (~self).(ForwardTo)(fwd ~a);
  }
}

impl forall [PET1: ConvertsInto(exttype(ref u64))]
   PET1 as I(exact exttype(var i64)) { ... }

impl forall [PET2: ConvertsInto(exttype(var i64))]
   exact exttype(ref u64) as I(PET2) { ... }

impl exact exttype(ref u64)
    as I(exact exttype(var i64)) {
  fn F(ref self: u64, var a: i64) = ForwardTo;
}
```

Note that these `impl` definitions are not created, but `impl` lookup acts as if
they exist. This replaces the `like` mechanism introduced in
[proposal #1144](https://github.com/carbon-language/carbon-lang/pull/1144), that
would add this parameterization to the `impl` itself, rather than making it part
of the `interface`.

Consequences:

-   The type structure of the match for purposes of prioritization counts
    `implicit_into` parameters as a pair of a type and the complete extended
    type, and changes depending on how well the query matches the `impl`.
    -   If the type of the `impl` doesn't match the type in the query, then both
        parts of the pair are considered to be `?`.
    -   If the type matches but the rest of the extended type doesn't, the type
        structure is considered to be `(T, ?)`.
    -   If the whole extended type matches, the type structure is considered to
        have a concrete match for both parts of the pair.
    -   This allows a type to implement the same extended type interface for
        multiple categories. For example, if a type implements an interface once
        with a value extended type and once with a reference extended type, a
        value or reference query will choose the `impl` with the matching
        category, while an initializing or constant query will choose the
        value-extended-type `impl` (the only one that applies, since we don't
        want to bind a reference parameter to a temporary). A separate
        `var`-extended-type `impl` can also be provided.
-   As a result, a single `impl` written in the source can correspond to
    multiple type structures. If any of the type structures of two different
    `impl`s are the same, they must be put together in a `match_first` block.
-   These additional `impl`s can only cover cases where we can reliably forward
    a source extended type parameter as an argument to the original function.
    The job of the `ConvertsInto` constraints is to restrict to those cases.
-   The number of notional `impl`s generated triples for every variant parameter
    (minus one for every variant parameter that is marked
    [`anchor`](#modifiers-for-coherence)).
-   An `implicit_into` parameter may only be used in a
    [contravariant position](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
    within the interface definition, as either:

    -   the extended type of an associated function `fwd` parameter, or
    -   the argument to
        [an `implicit_from` interface parameter](#implicit_from-variance) that
        is used as the type of a parameter.

    Note that parameters of a class are always _invariant_, even if that type is
    used as the type of a parameter.

Now consider a generic call to invoke `I.F`:

```carbon
fn CallF[T: Core.RefSelf(I(exttype(val i16)))]
    (ref x: T) {
  x.F(3);
}

var y: u64 = 2;
CallF(ref y);
```

A few things to note here:

-   The constraint on `T` requires `exttype(ref T)` to implement
    `I(exttype(val i16))`. The `impl` defined above is for
    `exttype(ref u64) as I(exttype(var i64))`, which doesn't match in the
    parameter extended type. However since that parameter is variant, it acts as
    if there is an impl that does satisfy that constraint:

    ```carbon
    impl forall [PET2: ConvertsInto(exttype(var i64))]
        exact exttype(ref u64) as I(PET2) {

      fn F(ref self: u64, fwd a: PET2) {
        // Performs an implicit conversion of `a`
        // to call the original implementation that
        // has one `var _: i64` parameter.
      }
    }
    ```

-   We could not pass in a reference to a `u32` object here, since although a
    `u32` value will implicitly convert to `u64`, a `u32 ref` won't convert to a
    `u64 ref`.
-   Inside of `CallF`, `x.F` takes a `val i16` parameter. The literal `3` gets
    implicitly converted to `i16` to make that call, and then gets converted
    again to `i64` to call the original implementation of `I`. See
    [the "Conversions when calling a function argument" section](#conversions-when-calling-a-function-argument).

#### Implied constraints

Consider a class that uses the interface with a variant parameter from before as
a parameter constraint:

```carbon
interface I(implicit_into Self: Core.ExtType,
            implicit_into P: Core.ExtType) {
  fn F(fwd self: Self, fwd a: P);
}

class C(P: I(exttype(val i16))) { ... }

fn G[P: I(exttype(val i32))](c: C(P));
```

Unlike with interfaces with invariant parameters, multiple different `impl`s
could satisfy that constraint. In this case, the `C(P)` parameter type in `G`
would be legal with the existing constraint of `I(exttype(val i32))` on `P`, but
implied constraints would normally add another constraint that
`P impls I(exttype(val i16))`. When
[we considered this on 2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dj0nqhfqo67w),
we decided we didn't want implied constraints adding constraints when there was
a choice of how to do it, so implied constraints don't apply to interfaces with
variant parameters.

#### Modifiers for coherence

Note that, unless modified by the `impl`, parameters marked `implicit_into`
can't be used to satisfy the orphan rule for coherence. As such we allow these
modifiers on those parameters, which can be used to ensure that there is a local
type in the `impl`'s type structure:

-   `exact`: the type and category must match between the query and `impl`,
    restricting the effect of `implicit_into` to other aspects of extended type
    for this parameter in this `impl`
-   `anchor`: at least one of the parameters marked `anchor` must have a
    matching type between the query and `impl`.

See [examples in the "Details" section](#coherence-modifier-examples).

We also allow using `anchor` to modify `implicit_into` on the _interface_
parameters to only allow conversions of extended type that don't change the
type. This can be useful since otherwise an interface that uses `implicit_into`
on all its parameters always has to use `anchor` when defining `impl`s.

> FIXME: **QUESTION:** Is it `anchor implicit_into` or `implicit_into anchor`?

Either use of `anchor` in effect adds a `where .type = T` constraint (from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)) on
the `ConvertsInto(`...`)` extended type facet type used to allow conversions to
use that `impl`.

Note that the presence or absence of `anchor`, `exact`, `implicit_into`, or
`implicit_from` is not included in `Core.ExtType` values.

> **Alternative considered:** We considered having `exact` mean exact extended
> type match, but this had some concerns such as:
>
> -   We want to allow evolution of what is included in an extended type. For
>     example we might want to include the source location of the argument
>     expression in the extended type.
> -   Additional knowledge of the compile-time value in the query should not
>     prevent matching.
>
> We also considered allowing `exact` to only match on type, excluding category,
> but that didn't match the expected interpretation of the word "exact," where
> it should definitely include anything explicitly written in an extended type
> literal value.

#### Facets create adapters

The facets of an interface without variant parameters preserve the original
type. When forming a facet of an interface with variant parameters, however, we
instead get an adapter type that also implements the interface. In this case,
the implementation of the interface for the adapter type performs the `i16` to
`i64` conversion and then delegates to the original implementation. This change
in type occurs anytime the compiler can't prove it can use the original type
with its original `impl` unchanged, and gives the freedom to have differences
from the original `impl` without violating coherence. A few differences can
occur:

-   Using the original type may mean different implicit conversions occur. In
    our example, the original type's `impl` would have directly converted `3` to
    `i64`, without going through `i16`.
-   As a result of using different implicit conversions, they may accept
    different input types. The original version could be called with `i32`, but
    the version inside `CallF` could be called with a type that converts to
    `i16` but not to `i64`.
-   The original type might have another `impl` of the same interface that could
    apply. The adapted type only has the ones specified by the constraint. Only
    the `impl` used to satisfy the constraint will be used by the adapted type.
-   Use of `anchor` or `exact` on the original `impl` will be ignored, only
    aspects of the `impl` that are explicitly in the constraint are preserved by
    the adapter.

For `template` parameters, if the call can be resolved at definition time using
the adapter type, then when the template is instantiated the call must be
resolved the same way. This follows the general consistency required of template
parameters with their constraints, see for example in
[member access](/docs/design/expressions/member_access.md#lookup-ambiguity).

This was discussed on
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dj0nqhfqo67w).

### `implicit_from` variance

Extended type associated constants and interface parameters of an interface may
be marked `implicit_from`. This allows variance as part of `impl` lookup, but in
the opposite direction from
[`implicit_into` parameters](#implicit_into-interface-parameter-variance). This
example shows an `implicit_from` associated constant:

```carbon
interface Factory {
  let implicit_from Result: Core.ExtType;
  fn Create(self) -> fwd Result;
}

fn UseFactory
    [T: Factory where .Result = exttype(var i64)]
    (x: T) -> i64 {
  return x.Create();
}

class HasFactory {
  impl as Factory {
    where Result = exttype(val i32);
    fn Create(self) -> val i32 {
      return 1;
    }
  }
}

let f: HasFactory = {};
UseFactory(f);
```

Like for `implicit_into` parameters, `implicit_from` associated constants mean
that creating a facet for a type may
[create an adapter type](#facets-create-adapters). In this case though, the
adapter converts between the extended return types. In our example, it is as if
the `UseFactory(f);` call was replaced by:

```carbon
class __HasFactory_adapter {
  adapt HasFactory;
  impl as Factory {
    where Result = exttype(var i64);
    fn Create(self) -> i64 {
      // Does implicit conversion i32 -> i64.
      return (self as HasFactory).Create();
    }
  }
}

UseFactory(f as __HasFactory_adapter);
```

This example shows an `implicit_from` interface parameter, using an interface
with an `implicit_into` parameter (such as `Core.CallPrimitive` from
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)):

```carbon
interface Registry(implicit_from P: Core.ExtType) {
  fn Register[FN: Core.CallPrimitive(P)](ref self, f: FN);
  fn DefaultArg() -> fwd P;
}
```

#### `Core.ConvertsFrom`

The mismatch between the extended type value in the constraint and `impl` is
allowed when this transformation is valid. In particular, there is a
`Core.ConvertsFrom` constraint that is the mirror of
[`Core.ConvertsInto`](#coreconvertsinto). If we create a function `F` with `RET`
as the extended type of its return, and `G()` a function returning extended type
`DEST` (also an `ExtType` value), as in:

```carbon
fn F() -> fwd RET;
fn G() -> fwd DEST {
  return F();
}
```

then `DEST` is a value of `ConvertsFrom(RET)` if and only if `return F();` is
legal.

For example, an `impl` of `Registry`:

```carbon
class R {}
impl R as Registry(exttype(val i32)) {
  fn Register
      [FN: Core.CallPrimitive(exttype(val i32))]
      (ref self, f: FN) {
    f(R.DefaultArg());
  }
  fn DefaultArg() -> val i32 {
    return 7;
  }
}
```

is as if it was written:

```carbon
class R {}
impl forall [RET: Core.ConvertsFrom(exttype(val i32))]
    R as Registry(RET) { ... }
```

This means that since `i32` implicitly converts to `i64`,
`R impls Registry(exttype(val i64))` as well, and the following use is allowed:

```carbon
fn TakesI64(x: i64);

// In the `DoRegister(ref r)` call below, `T`
// is an adapter for `R` that adds the
// `i32` -> `i64` conversion.
fn DoRegister[T: Registry(exttype(val i64))]
    (ref r: T) {
  // Calls `TakesI64` with `7 as i32`
  // which gets converted to an `i64`.
  r.Register(TakesI64);
}

var r: R;
DoRegister(ref r);
```

#### Covariant

An `implicit_from` interface parameter or associated constant may only be used
in a
[covariant position](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
within an interface definition, either:

-   the extended type of an associated function `fwd` return, as in
    `Factory.Create` and `Registry.DefaultArg`
    [earlier](#implicit_from-variance), or
-   the argument to
    [an `implicit_into` interface parameter](#implicit_into-interface-parameter-variance)
    that is used as the type of a parameter, as in `Registry.Register`
    [earlier](#implicit_from-variance).

### `match_first` in named constraints

Consider a situation where an interface with variant parameters is used twice in
the constraint of a generic parameter, as in this example (using `call` from
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)):

```carbon
fn H[T: call(_: i32) & call(_: i64)]
    (fwd g: T, x: i16) {
  // ❌ Ambiguous whether it uses
  // `call(_: i32)` or `call(_: i64)`.
  g(x);
}
```

Due to the allowed implicit conversions, it can be ambiguous which
implementation to use. In this example, the different implementations of `call`
could have different return types. Without a way to choose between them, the
compiler is forced to issue a diagnostic that the call is ambiguous.

To resolve this ambiguity, we need additional information from the developer in
the constraint. Due to the difficulty of expressing this inline in a constraint
and a desire to avoid a second syntax for expressing the prioritization between
`impl`s, we've decided to express this out-of-line in a named constraint using
the same `match_first` syntax used to prioritize `impl`s outside this context.

```carbon
constraint X(Self: Core.ExtType) {
  match_first {
    require Self impls call(_: i32);
    require Self impls call(_: i64);
  }
}

fn J[T: X](fwd g: T, x: i16) {
  // ✅ Uses `call(_: i32)`.
  g(x);
}
```

This was discussed on
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?pli=1&tab=t.0#heading=h.dj0nqhfqo67w).

### Interaction between `anchor`, `exact`, and named constraints

Without named constraints, we would only use `anchor` and `exact` in impls and
only in the position of a variant parameter to the interface. With named
constraints, we could want to use them in two other cases:

-   A type parameter in a named constraint may end up being a variant extended
    type argument to the underlying interface. For example, this would happen
    with the named constraint `AddWith`, which delegates to `AddWithPrimitive`
    (in
    [proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)).
    To support this, we might need to allow types to be marked with `anchor`.
    Further, Carbon would have to propagate whether an expression uses `anchor`
    as extra-type information that isn't used by type checking except if the
    expression ends up being the variant argument to an interface. We might also
    want to diagnose using `anchor` on a type in a way that doesn't end up being
    used as a variant argument, but that seems hard to determine.
-   If you have a constraint like `Add` which takes only the `Self` parameter
    and uses it to set all of the variant interface parameters, it is better to
    put `anchor` in `Add` than in all impls of `Add`. We could support this by
    allowing `anchor` or `exact` before interface arguments in
    `extend require impls` declarations in a named constraint. This would be
    ignored when using it as a constraint, but is used in an `impl` of the named
    constraint.

> FIXME: **QUESTION:** Is there a way to avoid some of this complexity?

This was discussed on
[2025-06-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52ru7ner80b4).

## Details

### Coherence modifier examples

Given this package:

```carbon
package P;

interface J
    (implicit_into P1: Core.ExtType,
     implicit_into P2: Core.ExtType,
     P3: Core.ExtType) { }

class C {}

// ✅ Allowed since `J` is local, even
// though `i32` is not.
impl i32 as
    J(exttype(val i32), exttype(val i32), exttype(val i32)) {}
// As if 4 impls with type structure:
// - i32 as J(i32, i32, i32)
// - i32 as J(?, i32, i32)
// - i32 as J(i32, ?, i32)
// - i32 as J(?, ?, i32)
```

A library using the interface from that package needs to ensure all of the
notional `impl`s implied by these `impl` declarations reference a local type.
This means that there needs to be a reference to a local type:

-   in a parameter that is not in an `implicit_into` parameter of the interface,
-   in a parameter that is an `exact` parameter of the `impl`, or
-   in every parameter that is an `anchor` parameter of the `impl`.

Note that the `Self` parameter before the `as` counts as a parameter for these
purposes, but local types in a `forall` clause do not.

```carbon
import P;

// ❌ Invalid since nothing is local.
impl bool as P.J(exttype(val P.C),
                 exttype(val P.C),
                 exttype(val P.C)) {}

class D {}

// ❌ Invalid since only local type is
// `D`, which is in `implicit_into` parameters.
impl i16 as P.J(exttype(val D),
                exttype(val D),
                exttype(val P.C)) {}
// As if 4 impls with type structure:
// - i16 as J(D, D, P.C)
// - i16 as J(?, D, P.C)
// - i16 as J(D, ?, P.C)
// - i16 as J(?, ?, P.C) <- problem


// ✅ Allowed: `D` is local and in a
// non-`implicit_into` parameter.
impl i8 as P.J(exttype(val P.C),
               exttype(val P.C),
               exttype(val D)) {}
// As if 4 impls with type structure:
// - i8 as J(P.C, P.C, D)
// - i8 as J(?, P.C, D)
// - i8 as J(P.C, ?, D)
// - i8 as J(?, ?, D) <- okay, has D

// ✅ Allowed: `D` is local and in an
// `exact` parameter.
impl i64 as P.J(exact exttype(val D),
                exttype(val P.C),
                exttype(val P.C)) {}
// As if 2 impls with type structure:
// - i64 as J(D, P.C, P.C)
// - i64 as J(D, ?, P.C) <- okay, has D

class E {}

// ✅ Allowed: every `anchor` parameter
// has a local type.
impl u16 as P.J(anchor exttype(val D),
                anchor exttype(val E),
                exttype(val P.C)) {}
// As if 3 impls with type structure:
// - u16 as J(D, E, P.C)
// - u16 as J(?, E, P.C) <- okay, has E
// - u16 as J(D, ?, P.C) <- okay, has D
// No J(?, ?, P.C) due to `anchor`

// ❌ Invalid since only one `anchor` is
// required to have type matching the
// query, and if it is the second
// `anchor`, then this declaration has
// no local type.
impl u32 as P.J(anchor exttype(val D),
                anchor exttype(val P.C),
                exttype(val P.C)) {}
// - u32 as J(D, P.C, P.C)
// - u32 as J(?, P.C, P.C) <- problem
// - u32 as J(D, ?, P.C)
// No J(?, ?, P.C) due to `anchor`
```

### `impl` lookup for an interface with variant parameters

There are a few changes needed to
[the current `impl` selection algorithm from proposal #5337](p005337-interface-extension-and-final-impl-update.md#impl-selection-algorithm):

-   The initial filter for `impl`s that have a compatible type structure for the
    query is changed to allow differences in the type structure in the
    [variant parameters](#implicit_into-interface-parameter-variance) to the
    interface.
-   If the interface or `impl` has [`anchor`](#modifiers-for-coherence)
    parameters, at least one must have a type match with the query.
-   If an `impl` passes those tests, its type structure must be determined so
    the `impl` may be prioritized against other candidates. This is done
    following the rules
    [described in the proposal](#implicit_into-interface-parameter-variance).

Note that the only observable semantics of `impl` lookup are the sequence of
`impl`s that are considered (including their order), and the `impl` that is
eventually matched. This allows the compiler to avoid work on other `impl`s if
it can prove they won't affect what's observable.

### Conversions when calling a function argument

One advantage of the approach of
[marking the variance on the interface parameter](#implicit_into-interface-parameter-variance)
instead of the approach prior to this proposal where the `impl` would be
parameterized to allow implicit conversions, is we can get implicit conversions
even when we can't have parameterized `impl`s. This comes up with function
parameters (using `call[val]` from
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)), and
this proposal allows them to work similarly to ordinary function calls:

```carbon
fn IntArg(x: i32);

fn ConcreteCall() {
  // ✅ Performs implicit conversion from
  // `Core.IntLiteral` to `i32`.
  IntArg(7);
}

fn GenericCall[FNT: call[val](_: i32)](f: FNT) {
  // Good to get the same conversion here, even
  // though we don't (and can't) have a blanket
  // `impl` that `f` accepts anything that
  // implicitly converts to `i32`.
  f(7);
}

GenericCall(IntArg);
```

## Future work

### Impl prioritization

We still need to have a mechanism for picking an `impl` to use when there are
multiple with the most preferred type structure. This issue is more acute with
extended types and variant parameters, since multiple notional `impl`s and
non-type aspects of extended types can participate in `impl` selection.

### Allow more variance

If motivated by use cases, we could add support for `implicit_into` variance on
associated constants, or variance on types, not just extended types. This would
be above and beyond types that are eventually used to construct an extended
type, as considered in
[the "Interaction between `anchor`, `exact`, and named constraints" section](#interaction-between-anchor-exact-and-named-constraints).

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Declaring variance on interfaces allows implicit conversions to work
    uniformly in generic code (such as calling a generic callable with an
    integer literal) as well as in concrete operator calls, without losing
    coherence.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    Requiring explicit `implicit_into`, `implicit_from`, `anchor`, and `exact`
    annotations keeps coherence local and avoids evolution hazards when new
    implicit conversions are introduced in downstream libraries.
-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    Interface variance allows library-defined callables, indexing, conversions,
    and operators (in follow-up proposals
    [#8187](https://github.com/carbon-language/carbon-lang/pull/8187) and
    [#8188](https://github.com/carbon-language/carbon-lang/pull/8188)) to
    support the same implicit conversions as built-in language operations.

## Alternatives considered

### Other ways to determine variance in interfaces

We first considered `in` and `out` (such as in discussions on
[2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5)
and
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?pli=1&tab=t.0#heading=h.dj0nqhfqo67w))
to mark the different kinds of variance instead of
[`implicit_into`](#implicit_into-interface-parameter-variance) and
[`implicit_from`](#implicit_from-variance), following other languages such as
Scala, Kotlin, TypeScript, and C#. We
[decided to switch 2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dqgfbeb33txh)
that `in` and `out` were particularly expensive words to reserve, and which
corresponded to each kind of variance was hard to remember. The chosen keywords
are intended to emphasize what implicit conversion will be used.

We also considered making the variance keyword part of the parameter name, much
like `each` on variadic parameters. This seemed unnecessary verbosity given
that:

-   the compiler would enforce that the variance was consistent with usage,
    there was no choice about what variance to use in a given position; and
-   the `impl` you write is the actual `impl` you get, there is just another
    layer around it.

We considered inferring variance, or just marking that there was variance and
inferring only the direction of variance. This had some downsides:

-   Variance comes with significantly different semantics, causing
    [facets to create adapters](#facets-create-adapters), so better if it is opt
    in.
-   Don't always want variance, for example since it requires extra annotations
    on impls to avoid coherence problems.
-   It goes against our
    ["information accumulation" principle](/docs/project/principles/information_accumulation.md)
    and general policy of not determining properties of parameters based on how
    they are used in the body of the definition:
    -   This allows forward declarations to have the information needed to be
        used.
    -   Avoids action at a distance. For example, having to look at the body of
        the interface and referenced interfaces.
    -   Makes the contract visible and explicit.
    -   Avoids potentially expensive reasoning when there are recursive
        references to the containing type.

### Support implicit conversions in impls

The support for implicit conversions prior to this proposal was provided by
parameterizing an `impl` so that it also applied to types that could be
implicitly converted from. This had the downside that the following code is an
error:

```carbon
fn F[T: call[val](_: i32) -> bool](f: T) -> bool {
  // Error: `T` supports `i32` arguments, not
  // `Core.IntLiteral`.
  return f(7);
}
```

Given Carbon's reliance on implicit conversions in various contexts, this was
considered important to address. We did not see a path to addressing this by
adding `forall` requirements that the impls providing the implicit conversions
exist. The issue is that those would be very hard to satisfy in the presence of
specialization. In addition, a `forall` constraint would preclude being called
with an overload set with overlap in arities. In general, the concern is that
there is nothing preventing the creation of a new library with a new type that
implicitly converts to the parameter types of both members of the overload set.
This would be an evolution hazard.

This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

### Infer `anchor` in `impl` declarations

The `anchor` annotation could be inferred instead of marked explicitly by which
types mentioned in the `impl` declaration are local to that file. That has some
downsides:

-   It causes the `impl` declaration to be sensitive to declarations that can be
    anywhere earlier in the same file. This causes readers of the code to need
    to scan the rest of the file to understand which arguments need a type
    match, and makes the impact of changes less local. This is against the
    ["low context-sensitivity" principle](/docs/project/principles/low_context_sensitivity.md).
-   This would cause semantic changes when concatenating files, for example to
    make a bug report. In particular, more types could then be used as anchors
    for satisfying the orphan rule. This is against the
    ["file concatenation should preserve meaning" principle](/docs/project/principles/file_concatenation.md).

This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).
