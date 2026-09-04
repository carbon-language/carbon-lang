# Updates to member access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7697)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Background](#background)
-   [Problem](#problem)
    -   [Callables representing the result of `impl` lookup](#callables-representing-the-result-of-impl-lookup)
    -   [Accessing names with instance and non-instance overloads](#accessing-names-with-instance-and-non-instance-overloads)
    -   [Facets with members associated with different interfaces](#facets-with-members-associated-with-different-interfaces)
    -   [C++ pointer-to-member values](#c-pointer-to-member-values)
    -   [Properties](#properties)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Non-instance members](#non-instance-members)
    -   [Callables for member functions](#callables-for-member-functions)
    -   [Overloading](#overloading)
    -   [Explicit about instance binding](#explicit-about-instance-binding)
    -   [C++ pointer-to-member values](#c-pointer-to-member-values-1)
    -   [`typeof`](#typeof)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Different way to distinguish whether instance binding occurs](#different-way-to-distinguish-whether-instance-binding-occurs)
    -   [Other member access operators](#other-member-access-operators)
    -   [Bind interfaces only used for compound member access](#bind-interfaces-only-used-for-compound-member-access)

<!-- tocstop -->

## Abstract

Update the rules for member access:

-   Simple member access `a.b`
    -   If `a` names a scope, performs name lookup and optionally `impl` lookup.
    -   Otherwise, `a.b` is shorthand for `a.(typeof(a).b)` and always performs
        instance binding.
-   Compound member access `a.(m)` does optional `impl` lookup and always
    performs instance binding.
    -   This is a change from only performing instance binding if `m` is an
        instance member.
-   New operation `a.impl(m)` is introduced. It always performs `impl` lookup,
    and nothing else.
-   The `BindToType` interface is removed. Only instance binding may be
    customized (using the `BindToValue` and `BindToRef` interfaces).

As a result, member access doesn't use whether the right operand is an instance
member anymore. Instead, instance binding is performed whenever it would be
plausible, and a new syntax is used to opt out.

## Background

-   The
    ["qualified names and member access" design document](/docs/design/expressions/member_access.md)
    reflects the design up to and including:
    -   [Proposal #989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
    -   [Proposal #2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
    -   [Proposal #3646: Tuples and tuple indexing](https://github.com/carbon-language/carbon-lang/pull/3646)
-   [Proposal #3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720)
    updated the member access rules to add customization of how member access
    worked by implementing binding interfaces. It defined some simple member
    accesses as rewrites into compound member access.
-   [Pull request #7557](https://github.com/carbon-language/carbon-lang/pull/7557)
    attempted to update the
    [member access design doc](/docs/design/expressions/member_access.md) to
    reflect the changes in
    [Proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720).
-   There were several discussions to figure out how to resolve the ambiguous
    points and resolve the problems we discovered:
    -   [#generics-and-templates discussion on 2026-08-20 on Discord](https://discord.com/channels/655572317891461132/941071822756143115/1540063686201311342)
    -   [discussion on 2026-08-24](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.3ot8c9eu3e1h#heading=h.bpukwg8e3446)
    -   [#typesystem discussion starting 2026-08-27 on Discord](https://discord.com/channels/655572317891461132/708431657849585705/1542650207525929070)
    -   [discussion on 2026-08-28](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?pli=1&tab=t.3ot8c9eu3e1h#heading=h.s780u75i71d1)
    -   [discussion on 2026-08-31](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?pli=1&tab=t.3ot8c9eu3e1h#heading=h.4ij84uxqftu5)

## Problem

[Proposal #3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720)
introduced some problems discovered when trying to update the design documents
in [PR #7557](https://github.com/carbon-language/carbon-lang/pull/7557):

-   No clear story for what use cases are solved by `BindToType` and when that
    customization would be needed.
-   Simple member access `a.b` was defined as a rewrite to compound member
    access `a.(M)`, but the definition of compound member access depended on
    whether `M` was an instance member, and so would not always have the desired
    behavior.
-   Unclear story around how we support overloading between instance and
    non-instance members. It seem to involve a default implementation of the
    binding interfaces for all types for use by non-instance members. Could
    non-instance members be repeatedly bound?

We thought it should be more obvious in which cases instance binding would
occur. In this example,

```
class C {
  extend base: (i32, i32);
  static let n: i32 = 0;
}

var x: C = {.base = (1, 2)};
```

Does `x.n` have the value `0` like `C.n` or `1` like `x.(0)`? The interpretation
depends on whether the binding interfaces are implemented for these types. In a
generic context, that could be unknown at checking time, as in this example:

```carbon
class D(T: Core.Default) {
  let x: T = T.Op();
}

fn F[T: type](d: D(T)) {
  // Is the result here have value `T`, or is it the
  // result of binding `T.Op()` to `d`?
  d.x
}
```

This would lead to awkward constraints on types in order to get expected normal
behavior in generic code.

The implementation of #3720 in the toolchain also looked to be expensive, with
broad blanket implementations of interfaces to get the expected default
behavior, leading to lots of `impl` lookups. As much as possible, builtin
`impl`s should be `final` or narrow.

### Callables representing the result of `impl` lookup

We want some way of creating callables from methods and member function from
interfaces, with the option of binding or not binding `self` for associated
methods.

```carbon
interface I {
  fn F();
  fn M(self);
}

class C {}
impl C as I { ... }

fn G(c: C) {
  // Would like to make callables for:
  // - impl lookup of `I.F` for `C`
  // - impl lookup of `I.M` for `C` taking a `C` parameter for `self`
  // - impl lookup of `I.M` for `C` where the `self` parameter is bound to `c`.
}
```

Before this proposal, the behavior of compound member access was different for
instance methods and non-instance member functions associated with an interface:

-   `C.(I.F)` would produce the result of looking up `I.F` for `C`.
-   `c.(I.F)` was invalid, since `c` is not a type, and so can't implement `I`.
-   `C.(I.M)` was invalid, since `I.M` is an instance member and can't perform
    instance binding to `C`.
-   `c.(I.M)` would perform `impl` lookup and then instance binding of `I.M` to
    `c`.

And there was no way, beyond writing a lambda, to get the result of `impl`
lookup of `I.M` in `C` without performing instance binding.

### Accessing names with instance and non-instance overloads

[Proposal #3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720)
erased some of the differences between instance and non-instance members in
order to support names that had both as overloads, to pave the way for
overloading to be added to the language, as in (using the function overload
syntax from
[discussion on 2025-03-28](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.t6l733mu79i7))

```carbon
interface I {
  overload F {
    fn (self) -> f32;
    fn (i32) -> bool;
  }
}

class C {
  overload G {
    fn (self) -> f32;
    fn (i32) -> bool;
  }
}
```

However, not all of the differences were eliminated, so it was unclear whether
the rewrite from simple to compound member access should use the type or value
of the left operand.

### Facets with members associated with different interfaces

We would like to support members of facets that are associated entities of
different interfaces, as in this example:

```carbon
interface I {
  fn F();
  fn M(self);
}

interface J {
  require impls I;
  alias I_F = I.F;
  alias I_M = I.M;
}

fn G[T: J](x: T) {
  // `T` is a facet of `J`, but the names `I_F` and `I_M`
  // from `J` refer to members of `I`. Access to those
  // members by way of `x` or `T` should work and use the
  // implementation of `I` by `T`.
}
```

This means member access into facets still needs to perform `impl` lookup, even
though in many cases the facet has the implementation in its witness.

### C++ pointer-to-member values

[C++ pointer-to-member](https://en.cppreference.com/cpp/language/pointer)
values should be usable from Carbon once bound to an instance.

```carbon
import Cpp inline '''
struct A {
  int m;
  auto F() -> int;
};

int A::* p = &A::m;
int (A::* q)() = &A::F;
''';

fn G(ref a: Cpp.A) -> i32 {
  // Equivalent to `a.*p + (a.*q)()` in C++.
  // Evaluates to `a.m + a.F()`.
  return a.(Cpp.p) + a.(Cpp.q)();
}
```

### Properties

See the
[future work section on properties in proposal #3720](/proposals/p003720-member-binding-operators.md#future-properties).
For purposes of this proposal, we want it clear that the custom code for
producing values for a property would be invoked by instance binding, and so it
is important that instance binding happen when writing expressions that looked
like ordinary member access. Ideally we would also have another syntax available
to be able to talk about the property before instance binding.

## Proposal

We provisionally define `typeof(x)` to give the static type of the expression
`x` without any runtime evaluation of `x`.

Simple member access `a.b` depends on what kind of entity `a` is:

-   If `a` is a namespace or package, only name lookup for `b` is performed.
-   If `a` names a facet, then `b` is looked up in the type of `a` (which by
    definition is a facet type such as an interface). If the lookup finds an
    associated entity, then `impl` lookup is performed. This lookup prefers and
    first looks in the facet `a`. This `impl` lookup is needed to address the
    ["facets with members associated with different interfaces" problem](#facets-with-members-associated-with-different-interfaces).
-   If `a` names a facet type, then `a.b` performs name lookup for `b` in `a`.
-   If `a` names another type (including any class), then `a.b` performs name
    lookup for `b` in `a`. If the result of lookup is an associated entity, then
    `impl` lookup is performed.
-   Otherwise, `a.b` is rewritten to the compound member `a.(typeof(a).b)`.
    -   `typeof(a)` will always be a facet or other type, so `typeof(a).b` will
        always be resolved using one of the above rules, and won't require
        further rewrites.
    -   If `b` is an associated entity, `typeof(a).b` will perform `impl` lookup
        using `typeof(a)`, so no `impl` lookup will happen during the compound
        member access from the rewrite.
    -   The compound member access will always perform instance binding, unlike
        prior to this proposal.

Compound member access `a.(m)` performs two steps:

-   If `m` is an associated entity, perform `impl` lookup with the `Self` type
    set to the type of `a`. This must succeed and be valid.
    -   For an interface `I` and class `C`, both `C.(I.F)` and `I.(I.F)` will
        fail due to `typeof(a) == type` not implementing `I`.
-   Instance binding is performed, using either the `BindToValue` or `BindToRef`
    interfaces implemented by `typeof(a)`. As in
    [proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720),
    the compiler provides `final` builtin implementations to provide the
    previous instance binding behavior.

Instead of writing `C.(I.F)` to perform `impl` lookup, we introduce new syntax
`C.impl(I.F)`. This always performs `impl` lookup with the the `Self` type equal
to `C` and nothing else. It is invalid unless `C` is known to implement `I`
(this check is delayed until the expression is no longer template dependent).
As a result, the left argument must always be a type or facet.

## Details

Simple member access `a.b` requires knowing what kind of entity the first `a`
operand is. In generic code, it might not be known whether a symbolic value
represents a type or some other kind of value. In that case, though, simple
member access isn't useful since we don't know enough about `a` to perform name
lookup into it.

### Non-instance members

Non-instance members of types (including classes and interfaces) no longer
implement the binding interfaces, and so may not be used with instance binding.

```carbon
interface I {
  // Non-instance member function
  fn F();
}

class C {
  // Non-instance member function
  fn G();

  // Non-instance static data member
  static var s: i32;

  impl as I;
}

fn PreviouslyAllowedNowInvalid(x: C) {
  // Previously allowed, but now invalid:
  // ❌ x.F();
  // ❌ x.G();
  // ❌ x.s = 1;
}

fn Instead(x: C) {
  // Instead, these should be written:
  C.F();  // ✅
  C.impl(I.F)();  // ✅
  typeof(x).F();  // ✅

  C.G();  // ✅
  typeof(x).G();  // ✅

  C.s = 1;  // ✅
  typeof(x).s = 1;  // ✅
}
```

We require that the caller distinguish whether they are performing instance
binding, which means that changing a method to a non-instance member function
requires updating callers.

### Callables for member functions

Thanks to the new `a.impl(m)` syntax, we can now produce callables for all of
the cases in the
["callables representing the result of `impl` lookup" section](#callables-representing-the-result-of-impl-lookup):

```carbon
interface I {
  fn F();
  fn M(self);
}

class C {}
impl C as I { ... }

fn G(c: C) {
  // impl lookup of `I.F` for `C`: `C.impl(I.F)`
  C.impl(I.F)();

  // impl lookup of `I.M` for `C` taking a `C` parameter for `self`: `C.impl(I.M)`.
  // This may be called with `c` passed in for `self` using:
  C.impl(I.M)(c);
  // or:
  c.(C.impl(I.M))(c)

  // impl lookup of `I.M` for `C` where the `self` parameter is bound to `c`:
  // `c.(I.M)`
  c.(I.M)();
}
```

Note how this changes the meaning of compound member access from before this
proposal:

-   `C.(I.F)` used to produce the result of looking up `I.F` for `C`, but is no
    longer valid since instance binding to `C` fails. The new syntax
    `C.impl(I.F)` is used instead.
-   `c.(I.F)` and `C.(I.M)` remain invalid.
-   The common case of `c.(I.M)` retains its previous meaning.

### Overloading

The
[overloading problem](#accessing-names-with-instance-and-non-instance-overloads)
is addressed by not using anything about the second operand to decide whether to
perform instance binding or whether to use the value or type of the left operand
for `impl` lookup. The only fact about the right operand that is used in the
proposed rules is whether it names an associated entity.

### Explicit about instance binding

With this proposal, `a.(m)` always performs instance binding, and `a.b` performs
instance binding unless `a` is a kind of entity where we never perform instance
binding such as packages and namespaces. Cases where you want to avoid instance
binding now have a separate syntax (`a.impl(m)`).

This means that generic code has a clear meaning, and transforming non-generic
code to be generic won't change behavior.

For [properties](#properties), this means that the normal ways of accessing
members will perform the instance binding that triggers the evaluation of the
property, but there is an opt-out syntax (`a.impl(m)`) when that is not desired.

### C++ pointer-to-member values

The solution to
[the "C++ pointer-to-member values" problem](#c-pointer-to-member-values) from
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720)
continues to work.

### `typeof`

`typeof(x)` has no runtime side effects, and produces a compile-time result.
This may involve compile-time evaluation, but all runtime effects from that
evaluation are discarded before code generation, as if the code is in an
`if (false)` block. For example:

```carbon
musteval fn P(T: type) -> type {
  return T*;
}

fn F[template T: type](ref x: T) -> P(T) {
  x += 1;
  return &x;
}

fn Call() {
  var y: i32 = 0;
  // Involves the compile-time evaluation of `P(i32)`,
  // and forming a specific instance of `F`. However,
  // `F(ref y)` is not called at runtime.
  StaticAssert(typeof(F(ref y)) == i32*);
  Assert(y == 0);
}
```

## Rationale

This proposal makes the behavior of member access more explicit, reducing
context sensitivity in accordance with
[the Carbon principle](/docs/project/principles/low_context_sensitivity.md).
Reducing ambiguity improves
[code readability](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
as does eliminating needing
[extra constraints for generic code](#explicit-about-instance-binding).

## Alternatives considered

### Different way to distinguish whether instance binding occurs

Instead of using `a.impl(b)` to skip instance binding, other syntax ideas were
[considered on 2026-08-28](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.3ot8c9eu3e1h#heading=h.s780u75i71d1)

-   `a.(b)` would not do instance binding, and `a.[b]` would. This made the more
    common case of instance binding look more unusual, and didn't provide a
    keyword in the rarer case that could be used to search the documentation or
    the web to understand what it meant.
-   `a.static(b)` was considered instead of `a.impl(b)`. `impl` was preferred
    since it better conveyed that `impl` lookup is the only thing that happens
    in that operation.

### Other member access operators

We considered introducing a `::` that primarily did qualified name lookup. This
didn't address the root causes of the problem, though, which was the varying
behavior after name lookup completed.

### Bind interfaces only used for compound member access

The bind interfaces needed to be used with simple member access as well,
otherwise properties would appear different than other members.
