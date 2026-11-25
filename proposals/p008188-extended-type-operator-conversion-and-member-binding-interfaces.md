# Extended-type operator, conversion, and member binding interfaces

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8188)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Summary](#summary)
-   [Proposal](#proposal)
    -   [Primitive indexing interface](#primitive-indexing-interface)
    -   [Primitive member binding interface](#primitive-member-binding-interface)
    -   [Implicit conversions of `self`](#implicit-conversions-of-self)
    -   [Primitive conversion interfaces](#primitive-conversion-interfaces)
        -   [with `partial`](#with-partial)
        -   [with `const`](#with-const)
    -   [Primitive versions of other operator interfaces](#primitive-versions-of-other-operator-interfaces)
        -   [Arithmetic operators](#arithmetic-operators)
        -   [Bitwise and shift operators](#bitwise-and-shift-operators)
        -   [Comparison operators](#comparison-operators)
        -   [Assignment operators](#assignment-operators)
        -   [Marking `ref` and `fwd` on operator operands](#marking-ref-and-fwd-on-operator-operands)
-   [Future work](#future-work)
    -   [Implementing multiple interfaces together](#implementing-multiple-interfaces-together)
    -   [`ref?` for `val`/`ref` accessor overloads](#ref-for-valref-accessor-overloads)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Disallow `ref` parameters on primitive operator interfaces](#disallow-ref-parameters-on-primitive-operator-interfaces)

<!-- tocstop -->

## Abstract

We replace Carbon's type-oriented operator, indexing, conversion, and member
binding interfaces with extended-type `Primitive` interfaces
(`Core.IndexWithPrimitive`, `Core.MemberBind`, `Core.AsPrimitive`,
`Core.ImplicitAsPrimitive`, and `Primitive` versions of arithmetic, bitwise,
shift, comparison, and assignment interfaces) that give `impl`s control over
operand and result extended types. For progressive disclosure, the existing
type-oriented names (`Core.IndexWith`, `Core.IndirectIndexWith`,
`Core.BindToVal`, `Core.BindToRef`, `Core.As`, `Core.ImplicitAs`,
`Core.AddWith`, and so on) are preserved as named constraints. We also define
how `partial` and `const` conversions interact with expression category and how
`ref` and `fwd` markers apply to operator operands.

## Problem

Prior to extended-type interfaces and interface variance, Carbon's operator,
indexing, conversion, and member binding interfaces were limited to ordinary
types:

-   Operators could not specify `var`, `ref`, or compile-time parameters or
    control the extended type of their results. In particular, C++ interop
    needs `ref` operands on operators such as `std::ostream& operator<<`.
-   Some conversions, such as adapter conversions and derived-to-base
    conversions, can return a reference to the original object with a new type.
    Previously, that was only possible for built-in conversions, since `Core.As`
    and `Core.ImplicitAs` always returned an initializing expression. Likewise,
    `partial` and `const` conversions could not express their category
    constraints through the conversion interfaces.
-   Subscript indexing (`lhs[index]`) required two separate interfaces
    (`Core.IndexWith` and `Core.IndirectIndexWith`) and complex rewrites that
    depended on the category of `lhs` and on which interface was _not_
    implemented.
-   Member binding (`x.(y)`) required three interfaces (`Bind`, `BindToValue`,
    and `BindToRef`) just to handle value versus reference categories.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal depends on pending proposals also split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
    (`Core.ExtType`, `exttype(...)`, `exttypeof(...)`, `.type`, `fwd` bindings,
    operands, and returns),
-   [proposal #8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185)
    (extended type interfaces, `extend`, `Core.ValSelf`/`RefSelf`/`VarSelf`),
-   [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](https://github.com/carbon-language/carbon-lang/pull/8186)
    (`implicit_into`, `implicit_from`, `anchor`, and `match_first` in named
    constraints), and
-   [proposal #8189: Associated constants in named constraints](https://github.com/carbon-language/carbon-lang/pull/8189)
    (used by the type-oriented named constraints wrapping each `Primitive`
    interface).

Examples in this proposal also reference pending
[proposal #8184: Redefining `alias`, `using`, and namespace aliases](https://github.com/carbon-language/carbon-lang/pull/8184)
(`alias` and `using`) and
[proposal #8187: Extended-type call interface and `call` syntax](https://github.com/carbon-language/carbon-lang/pull/8187)
(`Core.CallPrimitive` and `call[val]` for bound methods).

This proposal updates designs originally introduced in:

-   [Proposal #702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
-   [Proposal #820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
-   [Proposal #845: `as` expressions](https://github.com/carbon-language/carbon-lang/pull/845)
-   [Proposal #1083: Arithmetic](https://github.com/carbon-language/carbon-lang/pull/1083)
-   [Proposal #1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)
-   [Proposal #1191: Bitwise and shift operators](https://github.com/carbon-language/carbon-lang/pull/1191)
-   [Proposal #2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
-   [Proposal #2511: Assignment statements](https://github.com/carbon-language/carbon-lang/pull/2511)
-   [Proposal #3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720)
    and
    [proposal #7697: Updates to member access](https://github.com/carbon-language/carbon-lang/pull/7697).

## Summary

In short, this proposal introduces:

-   a change to the ...`[`...`]` indexing operator. It now uses the
    `Core.IndexWithPrimitive` extended type interface instead of a combination
    of `Core.IndexWith` and `Core.IndirectIndexWith` type interfaces.
    -   `Core.IndirectIndexWith` becomes a named constraint that requires
        `Core.IndexWithPrimitive`.
    -   `Core.IndexWith` becomes a named constraint that requires both
        `Core.IndexValWith` and `Core.IndexRefWith`. `Core.IndexWith` can't be
        implemented directly; implement `Core.IndexValWith` and
        `Core.IndexRefWith` separately instead.
-   a change to the
    [member binding interfaces](/docs/design/expressions/member_access.md#member-binding-interfaces).
    Those type interfaces (`Bind`, `BindToValue`, `BindToRef`) are replaced with
    named constraints (`BindToVal`, `BindToRef`) that `extend require`
    `MemberBind`, a single interface using extended types that is used for all
    cases.
-   extended type interfaces `Core.AsPrimitive` and `Core.ImplicitAsPrimitive`
    that replace type interfaces `Core.As` and `Core.ImplicitAs`. This gives
    `impl`s the ability to control the resulting category. `Core.As` and
    `Core.ImplicitAs` become named constraints requiring the corresponding
    primitive extended type interfaces.
    -   With this greater control, we define how conversions with `partial`- and
        `const`-qualified types interact with expression category.
-   new `Primitive` extended type interfaces for arithmetic, bitwise, shift,
    comparison, and assignment operators that allow control over extended type,
    not just type. The previous interface names are replaced with named
    constraints that allow you to continue to use types.
-   marking operator operands with `ref` (when selecting an `impl` taking a
    `ref` extended type) or `fwd` (for operands with generic extended type).

## Proposal

### Primitive indexing interface

Proposal
[#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
introduced the `IndexWith` and `IndirectIndexWith` interfaces. The rewrites from
a `lhs[index]` expression into invoking methods of interfaces depended on both
the category of the `lhs` part of the expression, and which interfaces were
implemented. This was complex due to lacking the tools for dispatching on
category, and resulted in undesirable dependence on the information that a
particular interface was not implemented, which meant that the rewrite could
change if another constraint was added. We compensated with a `final impl` to
make the two rewrites have type consistency. With extended type interfaces and
variance, however, we can take a simpler approach of having a single interface,
`IndexWithPrimitive`, that uses extended types, and two different use cases
expressed using named constraints:

```carbon
package Core;

interface IndexWithPrimitive(
    implicit_into anchor Self: ExtType,
    implicit_into Subscript: ExtType) {
  let implicit_from ResultExtType: ExtType;
  fn Op(fwd self: Self, fwd subscript: Subscript)
      -> fwd ResultExtType;
}
```

The expression "_lhs_ `[` _index_ `]`" is rewritten to
"_lhs_`.(IndexWithPrimitive(exttypeof(`_index_`)).Op)(`_index_`)`".

-   The `IndirectIndexWith` constraint has a single requirement: given a value
    `Self` the indexing operator produces a reference result.
-   The `IndexWith` constraint requires that both the `IndexValWith` and
    `IndexRefWith` constraints be implemented. The `IndexValWith` constraint
    provides a value result from indexing given a value `Self`, and the
    `IndexRefWith` constraint provides a reference result given a reference
    `Self`. In effect, using two implementations provides `ref?` semantics.
    `IndexValWith` and `IndexRefWith` use `alias`es to give different names
    (`At` and `Ref`) to the `Op` functions of `IndexWithPrimitive` with
    different parameters. During `impl` lookup, the `IndexRefWith` version will
    be preferred for reference expressions due to the rules for `implicit_into`
    interface parameter variance (in
    [proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186)).
    Note that `IndexWith` may be used as a parameter's constraint, but can't be
    directly implemented, since we don't support implementing multiple
    interfaces together (see
    [leads issue #4566: Implementing multiple interfaces with a single `impl` definition](https://github.com/carbon-language/carbon-lang/issues/4566)
    and
    [proposal #5168: Forward `impl` declaration of an incomplete interface](https://github.com/carbon-language/carbon-lang/pull/5168)).
    This gap in the design affects this operator, but will need to be addressed
    separately as future work, possibly by either changing the rules for
    implementing multiple interfaces or by using mixins to address this use
    case. This was
    [discussed on 2025-06-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52ru7ner80b4).

```carbon
// `lhs[index]` is a value expression when `lhs`
// is a value expression.
constraint IndexValWith(SubscriptType: type) {
  let ElementType: type;

  require exttype(val Self) impls
     IndexWithPrimitive(exttype(val SubscriptType))
     where .ResultExtType = exttype(val ElementType);
  alias At = ValSelf(IndexWithPrimitive(
      exttype(val SubscriptType))).Op;
}

// `lhs[index]` is a ref expression when `lhs`
// is a ref expression.
constraint IndexRefWith(SubscriptType: type) {
  let ElementType: type;

  require exttype(ref Self) impls
     IndexWithPrimitive(exttype(val SubscriptType))
     where .ResultExtType = exttype(ref ElementType);
  alias Ref = RefSelf(IndexWithPrimitive(
      exttype(val SubscriptType))).Op;
}

// `lhs[index]` is a reference expression or value
// depending on the category of `lhs`.
constraint IndexWith(SubscriptType: type) {
  let ElementType: type;
  match_first {
    extend require impls IndexRefWith(SubscriptType)
        where .ElementType = ElementType;
    extend require impls IndexValWith(SubscriptType)
        where .ElementType = ElementType;
  }
}

// `lhs[index]` is always a reference expression, even
// when `lhs` is a value.
constraint IndirectIndexWith(SubscriptType: type) {
  let ElementType: type;

  require exttype(val Self) impls
     IndexWithPrimitive(exttype(val SubscriptType))
     where .ResultExtType = exttype(ref ElementType);
  alias Ref = ValSelf(IndexWithPrimitive(
      exttype(val SubscriptType))).Op;
}
```

Observe that a type that defines an `impl` of `IndirectIndexWith` will also
satisfy the `IndexWith` constraint:

-   The "value `Self` returning a value" requirement can be satisfied due to
    `implicit_from` associated constant variance allowing the reference return
    to be converted to a value return.
-   The "reference `Self` returning a reference" requirement takes advantage of
    `implicit_into` variance to support converting the reference `Self` to the
    value expected by the `impl`.

Both `IndexValWith` and `IndirectIndexWith` require
`exttype(val Self) as IndexWithPrimitive(exttype(val SubscriptType))` with
different result extended types, and so conflict; a type can implement at most
one of them.

To support coherence, `IndexWithPrimitive` uses `implicit_into anchor` on the
`Self` parameter, since that is the parameter expected to be used to find the
`impl`.

Examples:

```carbon
class Array(template T: type, template N: i64) {
  impl as IndexValWith(i64) {
    where ElementType = T;
    fn At(self, subscript: i64) -> val T;
  }
  impl as IndexRefWith(i64) {
    where ElementType = T;
    fn Ref(ref self, subscript: i64) -> ref T;
  }
}

class Span(T: type) {
  impl as IndirectIndexWith(i64) {
    where ElementType = T;
    fn Ref(self, subscript: i64) -> ref T;
  }
}
```

Switching the indexing interface to extended types was
[discussed on 2025-06-10](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz).

### Primitive member binding interface

[Member binding](/docs/design/expressions/member_access.md#member-binding-interfaces)
(introduced in
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720) and
updated in
[proposal #7697](https://github.com/carbon-language/carbon-lang/pull/7697)) also
uses multiple interfaces (`Bind`, `BindToValue`, and `BindToRef`) in order to
handle different expression categories. We can now redefine them as named
constraints, in terms of a general `MemberBind` interface:

```carbon
interface MemberBind(implicit_into LHS: Core.ExtType) {
  let implicit_from ResultExtType: Core.ExtType;
  fn Op(self, fwd lhs: LHS) -> fwd ResultExtType;
}

// Formerly BindToValue
constraint BindToVal(T: type) {
  let Result: type;
  extend require impls
      MemberBind(exttype(val T))
      where .ResultExtType = exttype(val Result);
}

constraint BindToRef(T: type) {
  let Result: type;
  extend require impls
      MemberBind(exttype(ref T))
      where .ResultExtType = exttype(ref Result);
}
```

Observe that this replicates the semantics of `BindToValue` and `BindToRef`: in
particular, the `implicit_into` variance of the `LHS` parameter supports
[implicit conversions](/docs/design/expressions/member_access.md#inheritance-and-other-implicit-conversions).

However, there are a few differences:

-   There is no longer a `Bind` interface used to guarantee that the result type
    of `BindToVal` and `BindToRef` would be the same.
-   `BindToVal` will be used for reference expressions if `BindToRef` is not
    implemented, avoiding the need to routinely implement both.

In the future, `MemberBind` could be made into an extended type interface, but
currently there is no motivation to use anything other than the type of `Self`
for `impl` selection.

### Implicit conversions of `self`

Consider a class with an alias to a method in another class, with an implicit
conversion between the two classes:

```carbon
class Dog {
  // Equivalent to:
  // fn Bark(self: Dog);
  fn Bark(self) {
    DoBark();
  }
}

class Poodle {
  impl as ImplicitAs(Dog) {
    fn Convert(self) -> Dog {
      return {};
    }
  }
  // Equivalent to
  // `alias Bark = Dog.Bark;`
  using Dog.Bark;
}

var poodle: Poodle = {};
// Equivalent to:
// ((poodle as Dog).(Dog.Bark))()
poodle.Bark();
```

The implicit conversion of `poodle` to type `Dog` does not happen because of the
`Core.CallPrimitive` interface (in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)).
That interface declares its `Self` parameter as `implicit_into anchor`, which
means it won't look for different types as part of `impl` lookup, which is
important for coherence. Instead, that implicit conversion happens as a result
of the
[instance binding](/docs/design/expressions/member_access.md#instance-binding)
of `poodle` to `Dog.Bark`, when calling the `Op` method of the
[member binding interface](#primitive-member-binding-interface). In fact, there
is another conversion after that, to an adapter of `Dog` that implements
`Core.CallPrimitive` with the body of the `Dog.Bark` method.

```carbon
class __TypeOf_Dog_Bark {}
let generic __Dog_Bark: __TypeOf_Dog_Bark = {};

class __Binding_Dog_Bark {
  adapt Dog;
  impl as call[val]() -> () {
    fn Op(self) {
       DoBark();
    }
  }
}

impl __TypeOf_Dog_Bark as BindToRef(Dog) {
  where Result = __Binding_Dog_Bark;
  fn Op(unused self, ref r: Dog)
      -> ref __Binding_Dog_Bark {
    return r as __Binding_Dog_Bark;
  }
}

impl __TypeOf_Dog_Bark as BindToVal(Dog) {
  where Result = __Binding_Dog_Bark;
  fn Op(unused self, x: Dog)
      -> val __Binding_Dog_Bark {
    return x as __Binding_Dog_Bark;
  }
}
```

Only after binding is complete and `poodle` has been converted to a `Dog` is the
`CallPrimitive` implementation looked up as part of resolving the `()`
operation. That lookup will only find calls where the type of `Self` is `Dog`.

This works with
[our approach of allowing implicit conversions with the `self` parameter](/docs/design/expressions/member_access.md#inheritance-and-other-implicit-conversions).
See extension methods
[leads issue #1345: "How is the type of me restricted, and where can methods be declared?"](https://github.com/carbon-language/carbon-lang/issues/1345)
and
[deferred proposal #1122](https://github.com/carbon-language/carbon-lang/pull/1122).

### Primitive conversion interfaces

We also replace the conversion interfaces `As` and `ImplicitAs` with extended
type interfaces `AsPrimitive` and `ImplicitAsPrimitive`. These continue to each
have a parameter that specifies destination _type_, but an associated constant
(set by the `impl`) determines the destination _extended type_. The destination
extended type is constrained to have its `.type` match the parameter. We also
define named constraints `As` and `ImplicitAs` that operate on types, as before,
for
[progressive disclosure](/docs/project/principles/progressive_disclosure.md).

```carbon
package Core;

interface AsPrimitive(implicit_into Self: ExtType,
                      Dest: type) {
  // FIXME: Need to get the unqualified version
  // of `Dest` without `const`/`partial` here.
  let implicit_from ResultExtType: ExtType where .type = Dest;
  fn Convert(fwd self: Self)
      -> fwd ResultExtType;
}
constraint As(Dest: type) {
  extend require exttype(val Self) impls
      AsPrimitive(Dest)
      where .ResultExtType = exttype(var Dest);
}

// FIXME: `implicit_into` is concerning here, since we
// generally only want to perform a single implicit
// type conversion. Perhaps it should have `anchor`?
interface ImplicitAsPrimitive(
    implicit_into Self: ExtType, Dest: type) {
  final extend impl as AsPrimitive(Dest);
  // Inherited from AsPrimitive(Dest):
  // let implicit_from ResultExtType: ExtType where .type = Dest;
  // fn Convert(fwd self: Self)
  //     -> fwd ResultExtType;
}
constraint ImplicitAs(Dest: type) {
  extend require exttype(val Self) impls
      ImplicitAsPrimitive(Dest)
      where .ResultExtType = exttype(var Dest);
}
```

The expression `x as U` is rewritten to `x.(AsPrimitive(U).Convert)()`, and an
implicit conversion of `x` to type `U` is rewritten to
`x.(ImplicitAsPrimitive(U).Convert)()`.

These new capabilities allow us to support conversions that preserve category,
without workarounds unavailable to libraries, supporting our principle that
["All APIs are library APIs"](/docs/project/principles/library_apis_only.md).

-   Adapter types automatically implement `AsPrimitive`, which allows those
    conversions to preserve category.
-   Derived-to-base implicit conversions use `ImplicitAsPrimitive` to preserve
    category.

This was
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq).

#### with `partial`

Using category allows both preserving category and limiting to those cases that
are safe when casting between a `partial` class type and the whole class type:

-   Implicit conversion `C` -> `partial C` is allowed for any category, and
    preserves category, so `exttype(val C)`, `exttype(ref C)`, and
    `exttype(var C)` all implement `ImplicitAs(partial C)`, with a corresponding
    `.ResultExtType`:
    -   For `exttype(val C)`, `.ResultExtType = exttype(val partial C)`.
    -   For `exttype(ref C)`, `.ResultExtType = exttype(ref partial C)`.
    -   For `exttype(var C)`, `.ResultExtType = exttype(var partial C)`.
-   Implicit conversion `partial C` -> `C` is allowed only when the source is an
    initializing expression. A `partial C` initializing expression always has
    dynamic type `C`. With other expression categories, it could be a derived
    type.

    -   `exttype(var partial C)` `impls`
        `ImplicitAsPrimitive(C) where .ResultExtType = exttype(var C)`

    ```carbon
    fn F() -> partial C;
    fn G(c: C);
    // ✅ Allowed
    G(F());
    ```

Other result categories are possible by doing a category conversion after the
implicit type conversion.

This was discussed on
[2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.d985djkv6qti).

#### with `const`

Similarly with `const`, we can capture the fact that value expressions are
immutable, and therefore imply `const`.

-   Conversion `C` -> `const C` is allowed for any category and preserves
    category. This is since we are using a meaning of `const` like C++ `const`
    which means "not changed through this reference" rather than "reference to
    something immutable."
-   Conversion `const C` -> `C` is allowed for values.
    -   `exttype(val const C)` `impls`
        `ImplicitAsPrimitive(C) where .ResultExtType = exttype(val C)`
    -   A value with `exttype(ref const C)` will be able to use that conversion
        due to `implicit_into` variance, requiring a
        [value acquisition](/docs/design/values.md#value-acquisition).

This was discussed on
[2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.d985djkv6qti).

### Primitive versions of other operator interfaces

In general, operator interfaces from before this proposal are replaced with a
named constraint to a new interface with a `Primitive` suffix. The new interface
uses `Core.ExtType` where appropriate for parameters and associated constants
describing returns. Operators are rewritten to these new "Primitive" interfaces
instead.

For example, the expression `a + b` used to be rewritten to
`a.(AddWith(typeof(b)).Op)(b)`, calling a member of the `AddWith` interface. It
is now rewritten to `a.(AddWithPrimitive(exttypeof(b)).Op)(b)`, where
`AddWithPrimitive` is defined like so:

```carbon
package Core;

interface AddWithPrimitive(
    implicit_into Self: ExtType,
    implicit_into U: ExtType) {
  default let implicit_from ResultExtType: ExtType
      = exttype(var Self.type);
  fn Op(fwd self: Self, fwd other: U) -> fwd ResultExtType;
}
```

This "Primitive" interface has some changes:

-   Uses `Core.ExtType` instead of `type`.
-   The parameters are marked `implicit_into`. In some cases the left parameter
    is also marked `anchor`.
-   The associated constant controlling the return is marked `implicit_from`.
-   Uses `fwd` bindings and `fwd` returns.

For
[progressive disclosure](/docs/project/principles/progressive_disclosure.md), we
continue to provide `Core.AddWith` that uses types instead of extended types,
but it is now a named constraint:

```carbon
constraint AddWith(U: type) {
  let Result: type;

  extend require exttype(val Self) impls
      AddWithPrimitive(exttype(val U))
      where .ResultExtType = exttype(var Result);
}
constraint Add {
  extend AddWith(Self) where .Result = Self;
}
```

Note that the common type interfaces are unchanged.

#### Arithmetic operators

Updated in
[Arithmetic](/docs/design/expressions/arithmetic.md#extensibility): `Negate`,
`AddWith`, `SubWith`, `MulWith`, `DivWith`, `ModWith` (along with `Add`, `Sub`,
`Mul`, `Div`, `Mod`).

-   Unary `-`: `NegatePrimitive(implicit_into anchor Self: ExtType)` with
    `default let implicit_from ResultExtType: ExtType = exttype(var Self.type);`
    and `fn Op(fwd self: Self) -> fwd ResultExtType;`, plus named constraint
    `Negate`. `-x` is rewritten to `x.(NegatePrimitive.Op)()`.
-   Binary `+`, `-`, `*`, `/`, `%`: `AddWithPrimitive`, `SubWithPrimitive`,
    `MulWithPrimitive`, `DivWithPrimitive`, and `ModWithPrimitive` follow the
    `AddWithPrimitive` pattern above, with corresponding `OpWith` and `Op` named
    constraints. `x $ y` is rewritten to
    `x.(OpWithPrimitive(exttypeof(y)).Op)(y)`.

#### Bitwise and shift operators

Updated in
[Bitwise and shift operators](/docs/design/expressions/bitwise.md#extensibility):
`BitComplement`, `BitAndWith`, `BitOrWith`, `BitXorWith`, `LeftShiftWith`,
`RightShiftWith` (along with `BitAnd`, `BitOr`, `BitXor`, `LeftShift`,
`RightShift`).

-   Unary `^`: `BitComplementPrimitive(implicit_into anchor Self: ExtType)`
    (matching `NegatePrimitive`), rewritten as
    `x.(BitComplementPrimitive.Op)()`.
-   Binary `&`, `|`, `^`: `BitAndWithPrimitive`, `BitOrWithPrimitive`,
    `BitXorWithPrimitive` (matching `AddWithPrimitive`).
-   Shift `<<`, `>>`: `LeftShiftWithPrimitive` and `RightShiftWithPrimitive`
    anchor on the type of the left argument:

    ```carbon
    interface LeftShiftWithPrimitive(
        implicit_into anchor Self: ExtType,
        implicit_into U: ExtType) {
      default let implicit_from ResultExtType: ExtType
          = exttype(var Self.type);
      fn Op(fwd self: Self, fwd other: U)
          -> fwd ResultExtType;
    }
    ```

#### Comparison operators

Updated in
[Comparison operators](/docs/design/expressions/comparison_operators.md#extensibility):
`EqWith`, `OrderedWith` (along with `Eq`, `Ordered`).

```carbon
interface EqWithPrimitive(
    implicit_into Self: Core.ExtType,
    implicit_into U: Core.ExtType) {
  fn Equal(fwd self: Self, fwd u: U) -> bool;
  default fn NotEqual(fwd self: Self, fwd u: U) -> bool {
    return not ((fwd self) == (fwd u));
  }
}
constraint EqWith(U: type) {
  extend require impls
      ValSelf(EqWithPrimitive(exttype(val U)));
}
// FIXME: anchor
constraint Eq {
  extend EqWith(Self);
}

interface OrderedWithPrimitive(
    implicit_into Self: Core.ExtType,
    implicit_into U: Core.ExtType) {
  fn Compare(fwd self: Self, fwd u: U) -> Ordering;
  default fn Less(fwd self: Self, fwd u: U) -> bool {
    return self.Compare(fwd u) == Ordering.Less;
  }
  default fn LessOrEquivalent
      (fwd self: Self, fwd u: U) -> bool {
    let c: Ordering = self.Compare(fwd u);
    return c == Ordering.Less
        or c == Ordering.Equivalent;
  }
  default fn Greater(fwd self: Self, fwd u: U) -> bool {
    return self.Compare(fwd u) == Ordering.Greater;
  }
  default fn GreaterOrEquivalent
      (fwd self: Self, fwd u: U) -> bool {
    let c: Ordering = self.Compare(fwd u);
    return c == Ordering.Greater
        or c == Ordering.Equivalent;
  }
}
constraint OrderedWith(U: type) {
  extend require impls
      ValSelf(OrderedWithPrimitive(exttype(val U)));
}
// FIXME: anchor
constraint Ordered {
  extend OrderedWith(Self);
}
```

Comparison operators rewrite to `x.(EqWithPrimitive(exttypeof(y)).Equal)(y)`,
`x.(OrderedWithPrimitive(exttypeof(y)).Less)(y)`, and so on.

#### Assignment operators

Updated in [Assignment](/docs/design/assignment.md#extensibility): `AssignWith`,
`AddAssignWith`, `SubAssignWith`, `MulAssignWith`, `DivAssignWith`,
`ModAssignWith`, `BitAndAssignWith`, `BitOrAssignWith`, `BitXorAssignWith`,
`LeftShiftAssignWith`, `RightShiftAssignWith` (along with `Assign`, `AddAssign`,
and so on). These use `ref self` for the left argument so the left operand's
extended type does not vary, while the right operand uses
`implicit_into U: ExtType`:

```carbon
interface AssignWithPrimitive
    (implicit_into U: ExtType) {
  fn Op(ref self, fwd other: U);
}
constraint AssignWith(U: type) {
  extend require impls
      AssignWithPrimitive(exttype(val U));
}
constraint Assign {
  extend require impls AssignWith(Self);
}
```

The statement `x = y;` is rewritten to
`x.(AssignWithPrimitive(exttypeof(y)).Op)(y);`, and compound assignments rewrite
analogously. `Inc` (`++`) and `Dec` (`--`) are unchanged.

#### Marking `ref` and `fwd` on operator operands

Arguments to operators that are not always references (unlike the left operand
of assignment) but select an `impl` using a `ref` extended type must be marked
with `ref` at the callsite, as in `(ref out_stream) << x;`. This was discussed
in
[#interop on 2026-01-23](https://discord.com/channels/655572317891461132/768530752592805919/1464427960026005648).
Similarly, operator operands with generic extended type are marked with `fwd`,
as in `(fwd self) == (fwd u)`.

## Future work

### Implementing multiple interfaces together

As noted in [the indexing section](#primitive-indexing-interface), `IndexWith`
cannot currently be implemented directly because Carbon doesn't support
implementing multiple interfaces together
([leads issue #4566](https://github.com/carbon-language/carbon-lang/issues/4566),
[proposal #5168](https://github.com/carbon-language/carbon-lang/pull/5168)).
Addressing this — whether by changing multiple-interface implementation rules or
through mixins — is left as future work.

### `ref?` for `val`/`ref` accessor overloads

Supporting a concise way to implement both `val` and `ref` versions of an
extended type interface (such as `IndexValWith` and `IndexRefWith`) without
duplication is discussed as future work (`ref?`) in
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183).

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms):
    The increased expressivity of extended types allows us to prevent bugs — for
    example, limiting implicit conversion from `partial C` to `C` to
    initializing expressions avoids misuses of partially constructed values.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    Allowing `ref` extended types on primitive operator interfaces supports C++
    operators that take reference parameters (such as
    `std::ostream& operator<<`), and `IndexWithPrimitive` cleanly supports both
    owning containers (`std::vector`) and reference views (`std::span`).
-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    Conversions that preserve category (such as adapter casts, derived-to-base
    conversions, `partial`, and `const`) and unified indexing/member binding can
    be expressed directly in the language rather than as special built-in
    capabilities.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Preserving `As`, `ImplicitAs`, `AddWith`, `IndexWith`, `BindToVal`, and
    `BindToRef` as type-oriented named constraints ensures developers don't need
    to engage with extended types for ordinary operator and conversion uses.

## Alternatives considered

### Disallow `ref` parameters on primitive operator interfaces

We originally had restrictions (using `Core.NoRefExtType`) that the extended
type parameters to primitive operator interfaces not be references. This avoided
questions about whether the caller would have to write `ref` before the argument
in that case, as in `(ref x) + (ref y)` for the addition operator. It was also
intended to avoid having the possibility of surprising mutating side effects.
However, this would cause an interop problem since C++ operators can take
reference parameters. This comes up in practice with the `<<` operator when
applied to streams, as in:

```cpp
std::ostream& operator<<(std::ostream& lhs, Foo rhs);
```

Instead, we allow `Core.ExtType` (including `ref`) on primitive operator
interfaces and require call sites selecting a `ref` `impl` on a non-assignment
operator to mark the operand with `ref` (as in `(ref out_stream) << x;`),
preserving visibility of mutation at the call site without blocking C++ interop.

This was discussed in `#interop` on
[2026-01-20](https://discord.com/channels/655572317891461132/768530752592805919/1463317216739983505)
and
[-23](https://discord.com/channels/655572317891461132/768530752592805919/1464391757038096489).
