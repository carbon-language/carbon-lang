# Extended type interfaces and named constraint extensions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8185)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Summary](#summary)
-   [Proposal](#proposal)
    -   [Extended type interfaces](#extended-type-interfaces)
    -   [Extended types in `impl`s](#extended-types-in-impls)
    -   [`where`...`impls`](#whereimpls)
    -   [`extend`](#extend)
    -   [Converting from an extended type facet type to a type facet type](#converting-from-an-extended-type-facet-type-to-a-type-facet-type)
    -   [Named extended type constraints](#named-extended-type-constraints)
-   [Future work](#future-work)
    -   [`extend` in `where`...`impls` expressions](#extend-in-whereimpls-expressions)
    -   [Impl prioritization](#impl-prioritization)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Replace `require` for interface-interface requirements](#replace-require-for-interface-interface-requirements)
    -   [Related alternatives in other proposals](#related-alternatives-in-other-proposals)

<!-- tocstop -->

## Abstract

We extend Carbon's interfaces, `impl`s, `where` clauses, `extend` declarations,
and named constraints to support extended types (`Core.ExtType`):

-   Extended type interfaces and named extended type constraints may explicitly
    declare `Self` with `Core.ExtType` or an extended type facet type (or
    explicitly `Self: type`).
-   `impl`s of extended type interfaces or interfaces with `Core.ExtType`
    parameters use the `.type` property of extended types for the orphan rule
    while prioritizing on the full extended type.
-   `where`...`impls` supports extended types and extended type facet types.
-   `extend require`, `extend impl`, and `extend final impl` allow an explicit
    type or extended type to be specified before `impls` or `as`.
-   `Core.ValSelf`, `Core.RefSelf`, and `Core.VarSelf` convert extended type
    facet types into ordinary `type` facet types with the specified `Self`
    category.

## Problem

Ordinary interfaces in Carbon have an implicit `Self: type` parameter, so an
interface's `self` parameter category (or return category when `Self` is
returned) is fixed by the interface definition. However, many interfaces —
including callbacks, factories, callables, indexing, conversions, and operators
— need `impl`s to control whether `self`, parameters, or returns use `val`,
`ref`, `var`, or compile-time extended types.

At the same time, under
[progressive disclosure](/docs/project/principles/progressive_disclosure.md),
everyday callers and implementers who only want to work with ordinary types need
convenient ways (such as named constraints and `Core.ValSelf`/`RefSelf`/
`VarSelf`) to wrap extended type interfaces as ordinary `type` facet types.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal depends on pending
[proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
(`Core.ExtType`, `exttype(...)`, `.type`, extended type facet types,
`where .type = ...`, and `fwd` bindings and returns). It also complements
pending
[proposal #8189: Associated constants in named constraints](https://github.com/carbon-language/carbon-lang/pull/8189),
which is used alongside this proposal when wrapping extended type interfaces in
type-oriented named constraints.

This proposal provides the generics foundation for pending proposals also split
out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](https://github.com/carbon-language/carbon-lang/pull/8186),
-   [proposal #8187: Extended-type call interface and `call` syntax](https://github.com/carbon-language/carbon-lang/pull/8187),
    and
-   [proposal #8188: Extended-type operator, conversion, and member binding interfaces](https://github.com/carbon-language/carbon-lang/pull/8188).

This proposal also builds upon:

-   [Proposal #5337: Interface extension and `final impl` update](https://github.com/carbon-language/carbon-lang/pull/5337)
    (`extend require`, `extend impl`, and `extend final impl`).
-   [Proposal #7016: Updating `self` syntax and adding `static` member variables](p007016-updating-self-syntax-and-adding-static-member-variables.md)
    (putting `self` as the first parameter in parentheses `(`...`)`).
-   [Idea in open discussion on 2025-05-07 to allow parameterizing interfaces by extended types](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

## Summary

In short, this proposal introduces:

-   extended type interfaces, where `Self` has type `Core.ExtType` or an
    extended type facet type instead of `type` (as well as allowing explicit
    `Self: type`).
-   that an `impl` of an extended type interface, or an interface with extended
    type parameters, uses the `.type` property of extended type arguments to
    determine the type to use for the orphan rule, while using the full extended
    type value for prioritization during `impl` lookup.
-   `where A impls B` constraints where `A` is an extended type or extended type
    facet and `B` is an extended type facet type.
-   allowing `Self` or an extended type using `Self` to be specified in
    `extend require`, `extend impl`, and `extend final impl` declarations that
    previously required `Self` to be omitted.
-   `Core.ValSelf`, `Core.RefSelf`, and `Core.VarSelf` functions that take a
    `Core.ExtType` facet type and return an ordinary `type` facet type with the
    specified category used for `Self`.
-   named extended type constraints, allowing an optional `Self:`... declaration
    as the first parameter in `(`...`)` on named constraints.

## Proposal

### Extended type interfaces

Previously, interfaces had an implicit `Self` type parameter, set to the type
that is implementing the interface. We add the capability to explicitly declare
the `Self` parameter with an extended type facet type as the first parameter in
parentheses `(`...`)`, similar to the `self` parameter of methods in
[proposal #7016](p007016-updating-self-syntax-and-adding-static-member-variables.md):

```carbon
interface Callback(Self: Core.ExtType) {
  // Uses `Self` as the extended type of a
  // parameter using `fwd`.
  fn Go(fwd self: Self);
}

interface DefaultConstructible
    (Self: Core.ExtType) {
  // Uses `Self` as the extended type of the
  // return using `fwd`.
  fn MakeDefault() -> fwd Self;
}
```

`Callback` and `DefaultConstructible` are _extended type interfaces_
(specifically `Core.ExtType` interfaces, since `Self` is declared with
`Core.ExtType` rather than a more restricted extended type facet type from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)). A
value of an extended type interface is an _extended type facet_. Extended type
interfaces and extended type facets work the same way as ordinary "type"
interfaces and type facets, except `Self` is an extended type value satisfying
`Self`'s declared extended type facet type rather than a `type` value, and an
extended type facet implicitly converts to `Core.ExtType` (and to `Self`'s
declared extended type facet type) rather than to `type`.

An ordinary "type" interface acts as if `Self` is declared `Self: type` as the
first parameter in `(`...`)`, and `Self: type` may also be written explicitly.
Other `type` facet types (such as another interface, named constraint, or
`where` expression) are not allowed as the type of `Self`; express such
requirements with `require` declarations in the body instead (see
[the alternative considered](#replace-require-for-interface-interface-requirements)).

`Self` is bound to the subject of the `impl` or facet type, and is omitted from
the argument list when referencing the interface.

Let's break down what happens with an ordinary "type" interface:

```carbon
// Equivalent to: interface I(Self: type)
interface I {
  fn F(self);
}

// Since `I` is effectively declared `I(Self: type)`, that means
// `T` is a facet of a type, which means a type that is known to
// implement `I`. Since `T` is a facet of a type, it has an implicit
// conversion to `type`, so it can be used to the right of a `:`
// to give a type to `x`.
fn A[T: I](x: T) {
  x.F();
  // - Evaluates `typeof(x).F`, which is `T.F`:
  //   - `T` is a facet of type `I`, so look in `I` for `F`, finding `I.F`
  //     (which has signature `I(Self: type).fn F(self)`)
  //   - Since `I.F` is an associated entity, perform `impl` lookup with
  //     `Self = T as type`
  // - Requires that the type of `x` implements `I`
  //   - Satisfied by the bindings `x: T` and `T: I`, giving `T.impl(I.F)`
  // - Performs instance binding of `x` to that member, then calls it
}
```

Now look at how this works analogously with the `Core.ExtType` interface
`Callback`:

```carbon
// Since `Callback` is declared `Callback(Self: Core.ExtType)`, that means
// `CET` is a facet of a `Core.ExtType`, which means a `Core.ExtType` that is
// known to implement `Callback`. Since `CET` is a facet of a `Core.ExtType`,
// it has an implicit conversion to `Core.ExtType`, so it can be used in a
// `fwd` binding to give an extended type to `c`.
fn InvokeCallback[CET: Callback](fwd c: CET) {
  c.Go();
  // - Looks up `Go` using the extended type of `c`:
  //   - If the extended type is a paren extended type, allow `.0`, `.1` access
  //   - If the extended type is a curly brace extended type, allow `.x`, `.c` access
  //   - If the extended type is primitive, look in its type (as above)
  //   - Since the extended type `CET` is symbolic, look in `CET`'s type
  //     (`Callback`), finding `Callback.Go` (which has signature
  //     `Callback(Self: Core.ExtType).fn Go(fwd self: Self)`)
  //   - Since `Callback.Go` is an associated entity, perform `impl` lookup
  //     using the *extended type* of `c` (`CET as Core.ExtType`) as `Self`
  // - Requires the *extended type* of `c` to implement `Callback`
  //   - Satisfied by the bindings `fwd c: CET` and `CET: Callback`
  // - Performs instance binding of `c` to that member, then calls it
}
```

Other operations such as implementing an extended type interface also work
analogously, using extended type values instead of type values.

```carbon
class V {}
// `Core.ExtType` values implement `Callback`, instead of type
// values, since `Callback` is a `Core.ExtType` interface (and
// a `Core.ExtType` facet type).
impl exttype(val V) as Callback {
  // Notice the extended type of the `self` parameter is
  // `exttype(val V)`, the value of `Self`. This is the
  // same as `fn Go(fwd self: Self)`...
  fn Go(self: V) {
    Core.Print("Printed!");
  }
}

let v: V = {};
// The extended type of the expression `v` is `exttype(val V)`.
// This `Core.ExtType` value implements `Callback` so this
// call typechecks with `CET` deduced to be
// `exttype(val V) as Callback`.
//
// The `c.Go()` call in `InvokeCallback` prints "Printed!".
InvokeCallback(v);
```

With an extended type interface, we can change other aspects of the extended
type beyond the type. Here we show a similar example except with a `ref`
`Core.ExtType`:

```carbon
class R {
  var m: i32;
}
impl exttype(ref R) as Callback {
  // Notice the extended type of the `self` parameter is
  // `exttype(ref R)`, the value of `Self`.
  fn Go(ref self: R) {
    self.m += 2;
  }
}
var r: R = {.m = 3};
// This time `r` is a reference expression so `CET` is
// deduced to be `exttype(ref R) as Callback`.
InvokeCallback(ref r);
Assert(r.m == 5);
```

Finally, here is an example using `DefaultConstructible` instead of `Callback`:

```carbon
// Initializing extended type
impl exttype(var R) as DefaultConstructible {
  fn MakeDefault() -> R {
    return {.m = 7};
  }
}
r = (exttype(var R) as DefaultConstructible)
      .MakeDefault();
Assert(r.m == 7);

var global: R = {.m = 6};
// `ref` extended type
impl exttype(ref R) as DefaultConstructible {
  fn MakeDefault() -> ref R {
    return global;
  }
}
(exttype(ref R) as DefaultConstructible)
      .MakeDefault().m += 3;
Assert(global.m == 9);
```

### Extended types in `impl`s

Interfaces can also use `Core.ExtType` (or an extended type facet type) as the
types of their parameters or associated constants.

```carbon
interface OneArg(P: Core.ExtType) {
  let Result: Core.ExtType;
  fn Fun(self, fwd a: P) -> fwd Result;
}

impl MyType as OneArg(exttype(val i32)) {
  where Result = exttype(var bool);
  fn Fun(self, a: i32) -> bool;
}
```

To satisfy coherence by
[the orphan rule](/docs/design/generics/details.md#orphan-rule), extended types
are converted to types in the type structure using the `.type` property from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183) (as
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq)).
The extended type value itself is used, though, when prioritizing during `impl`
lookup.

Since extended types in an `impl` use `.type` to get the types for the orphan
rule, we end up with a situation where the type to satisfy the orphan rule can
come from the `forall` clause using `where .type = ...` (from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)). In
this example:

```carbon
impl [PET: Core.ExtType where .type = MyClass] PET as I;
```

The type structure for the orphan rule is `PET.type as I` which, after applying
the rewrite from `PET`'s constraint, becomes `MyClass as I`.

### `where`...`impls`

A `where` constraint may now contain a requirement `A impls B` where `A` is an
extended type or facet of an extended type, and `B` is an extended type facet
type. For example, a generic binding
`T: type where exttype(val .Self) impls Callback` will bind to types `T` such
that there is an implementation of `Callback` for `exttype(val T)`.

### `extend`

Prior to this proposal, the `extend` modifier on some declarations means that
some type argument must be omitted, and is presumed to be `Self`, although
there's a syntactic position available for it:

-   In an interface or named constraint, `extend require`, as defined in
    [proposal #5337](https://github.com/carbon-language/carbon-lang/pull/5337),
    must be followed by `impls` instead of allowing a type to be specified.
-   In a class, `extend impl` and `extend final impl` must be followed by `as`
    instead of allowing a type to be specified.
-   Similarly, in an interface, `extend impl` and `extend final impl`, as
    defined in
    [proposal #5337](https://github.com/carbon-language/carbon-lang/pull/5337),
    must be followed by `as` instead of allowing a type to be specified.

In all of these cases, we drop that restriction, and allow a type or an extended
type to be specified in the expected position. These will generally only be
useful if the `Self` type or extended type can implicitly convert to what is
placed in that position, though we allow anything in the spirit of supporting
the extension method approach of
[leads issue #1345](https://github.com/carbon-language/carbon-lang/issues/1345)
and
[deferred proposal #1122](https://github.com/carbon-language/carbon-lang/pull/1122).
This doesn't override the need for `Self` to appear in a deducible position for
an `impl` in a generic `class` scope, as described in
[leads issue #5251](https://github.com/carbon-language/carbon-lang/issues/5251#issuecomment-2779776355).

### Converting from an extended type facet type to a type facet type

Given an extended type facet type, we can construct an ordinary `type` facet
type in a few different ways. One is to construct a named constraint. The `Self`
values of the named constraint are types, but the requirements of the constraint
can be on extended type values using the `Self` type.

```carbon
// Equivalent to `constraint ValCallback(Self: type)`
constraint ValCallback {
  extend require exttype(val Self) impls Callback;
}

constraint RefCallback {
  extend require exttype(ref Self) impls Callback;
}

constraint VarCallback {
  extend require exttype(var Self) impls Callback;
}
```

Another way is with [a `where`...`impls` expression](#whereimpls):

-   `type where exttype(val .Self) impls Callback`
-   `type where exttype(ref .Self) impls Callback`
-   `type where exttype(var .Self) impls Callback`

This however doesn't give a way to have name lookup into the facet type find
names in `Callback`, like the named constraints accomplish with
[`extend`](#extend). For that, we introduce functions that take a `Core.ExtType`
facet type and return an ordinary `type` facet type: `Core.ValSelf`,
`Core.RefSelf`, and `Core.VarSelf`. They are used like:

```carbon
fn InvokeValCallback
    [T: Core.ValSelf(Callback)](c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(self: T)`.
  c.Go();
}

fn InvokeRefCallback
    [T: Core.RefSelf(Callback)](ref c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(ref self: T)`.
  c.Go();
}

fn InvokeVarCallback
    [T: Core.VarSelf(Callback) & Core.Move]
    (var c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(var self: T)`.
  // The upcoming move proposal will define
  // how this works, but is expected to make
  // `~c` an initializing expression in this
  // case.
  (~c).Go();
}

fn InvokeWithResultOfFunctionCall
    [T: Core.Call() where
         .Result impls VarSelf(Callback)]
    (f: T) {
  // Not `f().Go()` since `.Result impls`
  // doesn't give `.Result` names.
  f().(VarSelf(Callback).Go)();
}
```

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5);
generalizing this with
[`extend` in `where`...`impls` expressions](#extend-in-whereimpls-expressions)
is deferred to future work.

### Named extended type constraints

We want named constraints to provide
[progressive disclosure](/docs/project/principles/progressive_disclosure.md) for
extended types, providing alternative ways of accessing functionality that uses
extended types at the lowest level, but can use a named constraint with types
instead in common scenarios. This is similar to how we use named constraints in
other circumstances to hide complexity, such as providing `Add` as an easier
alternative to `AddWith` when you don't need the additional flexibility.

We allow an optional `Self:`... declaration as the first parameter in `(`...`)`
on named constraints, with the same allowed types for `Self` as on
[extended type interfaces](#extended-type-interfaces) (including explicit
`Self: type`, the default when `Self` is omitted). A `require impls` or
`require Self impls` in a constraint is allowed when the type of `Self` matches
the type of the `Self` of the interface or constraint being required.

```carbon
constraint DefaultConstructibleCallback
    (Self: Core.ExtType) {
  // `Self` is `Core.ExtType`, matches the `Self`
  // of `Callback` and `DefaultConstructible`.
  extend require impls Callback;
  extend require impls DefaultConstructible;
}
```

## Future work

### `extend` in `where`...`impls` expressions

A core-language syntax allowing `extend` inside a
[`where`...`impls`](#whereimpls) expression, such as
`type where exttype(val .Self) extend impls Callback`, could expose names from
the required interface directly on the resulting facet type (for example, in
`InvokeWithResultOfFunctionCall`
[above](#converting-from-an-extended-type-facet-type-to-a-type-facet-type), as
well as replacing `Core.ValSelf`, `Core.RefSelf`, and `Core.VarSelf`). We defer
designing general `where`...`extend impls` semantics as future work in order to
limit the scope of this proposal.

### Impl prioritization

We still need to have a mechanism for picking an `impl` to use when there are
multiple with the most preferred type structure. This issue becomes more acute
with extended types, since there is more non-type information that can now
appear in an `impl` signature.

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Extended type interfaces allow generic APIs to abstract over parameter and
    return categories while preserving familiar member-access and `impl`
    semantics.
-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    Enabling interfaces and `impl`s to parameterize on `Core.ExtType` allows
    callables, indexing, conversions, and operators (in follow-up proposals
    [#8187](https://github.com/carbon-language/carbon-lang/pull/8187) and
    [#8188](https://github.com/carbon-language/carbon-lang/pull/8188)) to be
    defined as library interfaces rather than compiler built-ins.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Named constraints, `extend` with explicit extended types, and
    `Core.ValSelf`/`RefSelf`/`VarSelf` let library authors wrap extended type
    interfaces in ordinary type-level interfaces for common callers.

## Alternatives considered

### Replace `require` for interface-interface requirements

In some cases, a `require` declaration in an interface to add a constraint that
another interface must be implemented, as in:

```carbon
interface J {}

interface I {
  require impls J;
}
```

is equivalent to an interface with a constraint on `Self`:

```carbon
interface I(Self: J) {}
```

This however had concerns which led us to keep using `require` and disallow
interface, named constraint, or `where` facet types on `Self` (avoiding
[two ways to express the same requirement](/docs/project/principles/one_way.md)):

-   It raised questions about how qualified name lookup into `Self` would work.
-   We preferred these requirements to be expressed in the body of the interface
    definition, not declaration, from a reading perspective.
-   Having the requirements in the declaration added restrictions on `I`'s
    declaration.

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).

### Related alternatives in other proposals

For related alternatives that considered avoiding separate extended type
interfaces altogether — decoupling type from the rest of the extended type, or
representing all extended type information in ordinary types — see
[proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183).
