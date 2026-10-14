# Abbreviated `interface` and `impl` syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7896)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Primary interface functions](#primary-interface-functions)
    -   [Abbreviated `impl` declarations, `impl fn`, and signature deduction](#abbreviated-impl-declarations-impl-fn-and-signature-deduction)
        -   [Abbreviated `impl` syntax](#abbreviated-impl-syntax)
        -   [Replacing `impl as` with `impl fn` and `impl Self as`](#replacing-impl-as-with-impl-fn-and-impl-self-as)
        -   [Deducing interface arguments and associated constants from the signature](#deducing-interface-arguments-and-associated-constants-from-the-signature)
    -   [Interface extension, anonymous aliases, and named constraints](#interface-extension-anonymous-aliases-and-named-constraints)
    -   [Out-of-line definitions](#out-of-line-definitions)
    -   [Updating `Op` interfaces](#updating-op-interfaces)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep placeholder names like `Op` or repeat the interface name](#keep-placeholder-names-like-op-or-repeat-the-interface-name)
    -   [Keep `impl as` alongside or instead of `impl fn`](#keep-impl-as-alongside-or-instead-of-impl-fn)
    -   [Also abbreviate `interface` declarations](#also-abbreviate-interface-declarations)
    -   [Omit the function name in compound member access](#omit-the-function-name-in-compound-member-access)
    -   [Omit the function name in standalone out-of-line `fn` definitions](#omit-the-function-name-in-standalone-out-of-line-fn-definitions)
    -   [Make the implicit function name visible to unqualified lookup](#make-the-implicit-function-name-visible-to-unqualified-lookup)
    -   [Call primary functions without naming the member](#call-primary-functions-without-naming-the-member)
    -   [Deduce associated constants in braced `impl` definitions](#deduce-associated-constants-in-braced-impl-definitions)
    -   [Dedicated `op` keyword](#dedicated-op-keyword)

<!-- tocstop -->

## Abstract

This proposal introduces _primary functions_ in `interface` and `impl`
declarations, along with _abbreviated `impl` syntax_ that omits the enclosing
`{ ... }` braces when implementing a single function of an interface or named
constraint, and replaces `impl as` with `impl fn` (and `impl Self as` for other
`impl` declarations):

```carbon
interface Core.Negate {
  default let Result: type = Self;
  fn (self) -> Result;
}

interface Core.AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}

class Point {
  var x: f32;
  var y: f32;

  impl fn Core.Negate(self) -> Self {
    return {.x = -self.x, .y = -self.y};
  }
  impl fn Core.Add(self, other: Self) -> Self {
    return {.x = self.x + other.x, .y = self.y + other.y};
  }
  impl fn Core.MulWith(self, scale: like f32) -> Self {
    return {.x = self.x * scale, .y = self.y * scale};
  }
  impl fn Core.Eq.Equal(self, other: Self) -> bool {
    return self.x == other.x and self.y == other.y;
  }
}
```

An unnamed `fn` declaration in an `interface` implicitly takes the name of the
enclosing `interface` for qualified member lookup (such as
`p.(Core.AddWith(Point).AddWith)(q)`,
`(Point as Core.AddWith(Point)).AddWith`, or `p.Negate()` and `p.AddWith(q)` in
generic code or when the `impl` is extended). In an abbreviated `impl`
(`impl <SelfType> as <FacetType> fn [Name](...)`), omitting the function name
implements the interface's primary function, while specifying `Name` implements
the named function member of an interface whose other members are defaulted or
deduced. Within a `class` (or `interface`), `impl fn X.Y.Z(...)` (where each
component may include arguments) is defined as the same as
`impl Self as X.Y.Z fn (...)` when `X.Y.Z` names an interface (or named
constraint) with a primary function (such as `impl fn Core.Add(...)`), and as
`impl Self as X.Y fn Z(...)` when `X.Y.Z` has more than one component and names
a function member `Z` of an interface `X.Y` (such as
`impl fn Core.Eq.Equal(...)`), replacing `impl as` (while non-`impl fn`
declarations write `impl Self as`). Omitted interface arguments and associated
constants that appear in the function's signature are deduced from the `impl`'s
`fn` signature (with `like` permitted on parameter types to deduce `like` on the
corresponding interface argument).

This replaces placeholder method names like `Op` across operator, callable,
lifecycle, and other single-operation interfaces whose names work as method
names on implementing types.

## Problem

Many interfaces in Carbon represent a single operation: arithmetic and bitwise
operators (`Core.AddWith`, `Core.Negate`), assignment and compound assignment
(`Core.AssignWith`, `Core.AddAssignWith`, `Core.Inc`, `Core.Dec`), callables
(`Core.Call`), pointer dereferencing (`Core.Deref`), lifecycle operations
(`Core.Copy`, `Core.Destroy`), and pattern matching (`Core.Match`). In the
current design, declaring and implementing these interfaces has three main
problems:

1.  Placeholder or redundant function names: Because every associated function
    in an interface requires an explicit identifier, single-function interfaces
    either use a placeholder name like `Op` or repeat the interface's name
    (`interface EqualWith(U: type) { fn EqualWith... }`). When an interface is
    implemented with `extend impl` in a class, a placeholder name like `Op`
    cannot be usefully extended into the class's scope without colliding with
    other interfaces using `Op` or exposing an unhelpful `x.Op()` method.
    Conversely, repeating the interface name in both the `interface` and `impl`
    declarations is verbose and causes the function name to shadow the interface
    name in unqualified lookup inside the interface body.
2.  Boilerplate in single-function `impl` declarations: Implementing a
    single-operation interface requires repeating operand types in both the
    interface argument list and the function parameter list, writing a
    `where .Result = ...` clause on the facet type whenever the return type
    differs from the default, and wrapping the single `fn` definition in braces:

    ```carbon
    // Current syntax:
    impl Meters as Core.DivWith(Seconds) where .Result = MetersPerSecond {
      fn Op(self, other: Seconds) -> MetersPerSecond {
        return {.value = self.value / other.value};
      }
    }
    ```

    Here, `Seconds` is written twice (once in `Core.DivWith(Seconds)` and again
    in `other: Seconds`), `MetersPerSecond` is written twice (once in
    `where .Result = MetersPerSecond` and again in `-> MetersPerSecond`), and
    programmers writing an operator implementation must learn associated type
    rewrite constraints before returning a type other than `Self`.
3.  Awkward `impl as` shorthand in class bodies: Inside a `class`, omitting
    `Self` before `as` (`impl as Core.AddWith(Self) { ... }`) leaves a unary
    prefix `as` and, when combined with an abbreviated `fn`, places an unnamed
    `fn (` mid-line
    (`impl as Core.MulWith(like f32) fn (self, scale: f32) -> Self`) rather than
    reading like a member function declaration
    (`impl fn Core.MulWith(self, scale: like f32) -> Self`).

## Background

-   [Issue #4711: Syntax for implementing a single-function interfaces](https://github.com/carbon-language/carbon-lang/issues/4711)
    proposed omitting the redundant function name in single-function interfaces
    and omitting `{ ... }` braces and `where .Result = ...` in single-function
    `impl` declarations.
-   [Leads issue #1058: How should interfaces for core functionality be named?](https://github.com/carbon-language/carbon-lang/issues/1058)
    established `fn Op` as a temporary convention for operator and core
    interfaces, adopted across
    [#1083: Arithmetic](/proposals/p001083-arithmetic-expressions.md),
    [#1178: Rework operator interfaces](/proposals/p001178-rework-operator-interfaces.md),
    [#1191: Bitwise and shift operators](/proposals/p001191-bitwise-and-shift-operators.md),
    [#2187: Update sum types design](/proposals/p002187-update-sum-types-design.md),
    [#2511: Assignment statements](/proposals/p002511-assignment-statements.md),
    [#2875: Functions, function types, and function calls](/proposals/p002875-functions-function-types-and-function-calls.md),
    and
    [#3720: Member binding operators](/proposals/p003720-member-binding-operators.md).
-   [Proposal #3848: Lambdas](/proposals/p003848-lambdas.md) introduced
    omitting the function name after `fn` for anonymous functions (lambdas).
-   [Proposal #3762: Merging forward declarations](/proposals/p003762-merging-forward-declarations.md)
    and
    [Proposal #5337: Interface extension and `final impl` update](/proposals/p005337-interface-extension-and-final-impl-update.md)
    defined how interfaces and named constraints extend and alias members of
    other interfaces.
-   [Proposal #5366: The name of an `impl` in `class` scope](/proposals/p005366-the-name-of-an-impl-in-class-scope.md)
    defined how `impl` declarations inside a class are named and redeclared
    out-of-line.
-   [Proposal #6008: Replace `impl fn` with `override fn`](/proposals/p006008-replace-impl-fn-with-override-fn.md)
    replaced `impl fn` for virtual method overrides with `override fn`.
-   [Proposal #7016: Updating `self` syntax and adding `static` member variables](/proposals/p007016-updating-self-syntax-and-adding-static-member-variables.md),
    [Proposal #7254: Replace `:!` and `:?` with keywords and contextual defaults](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md),
    and
    [Proposal #7697: Updates to member access](/proposals/p007697-updates-to-member-access.md)
    updated method receiver syntax (`fn (self, ...)` and `fn (ref self, ...)`),
    generic bindings, and member access.
-   The design in this proposal was explored in
    [Dedicated syntax for interface functions](https://docs.google.com/document/d/1l3AdRSEbC7uWU6eRzRLu47IXF83-PR6f7tBz_Rqb2EY/edit)
    and in open discussions on
    [2025-03-28](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.t6l733mu79i7)
    and
    [2025-04-30](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.a7jqjxpzyrp4).

## Proposal

### Primary interface functions

An `interface` may contain at most one unnamed `fn` declaration, which omits the
function name between `fn` and its parameter list:

```carbon
interface Core.AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}
```

This function is the _primary function_ of the interface:

-   It implicitly has the same name as the enclosing `interface` for qualified
    member lookup (such as `x.AddWith(y)`, `Self.AddWith(x, y)`,
    `x.(Core.AddWith(U).AddWith)(y)`, `(T as Core.AddWith(U)).AddWith(x, y)`, or
    `Core.AddWith(U).AddWith(x, y)`).
-   Its implicit name is never found by unqualified name lookup in any scope,
    including inside the `interface`, an extending `interface`, `constraint`, or
    `class`, or an `impl`. Within those scopes, the unqualified identifier
    `AddWith` refers to the interface itself. To call or reference the primary
    function, code uses qualified member lookup (such as `self.AddWith(other)`
    in a class that extends the implementation,
    `self.(Core.AddWith(U).AddWith)(other)` in a non-extending `impl`, or
    `Self.AddWith`).

An `impl` of an interface with a primary function may omit the function name on
at most one `fn` declaration to implement the primary function, or it may write
the interface's name explicitly (`fn AddWith(self, other: U) -> Self`).

### Abbreviated `impl` declarations, `impl fn`, and signature deduction

#### Abbreviated `impl` syntax

An `impl` declaration (including `extend impl` and `final impl`) may omit the
`{ ... }` braces and follow the facet type directly with a `fn` declaration or
definition:

-   Omitting the function name after `fn` (`impl <SelfType> as <FacetType> fn
    (...)`) implements the primary function of the interface or named
    constraint.
-   Specifying a function name after `fn` (`impl <SelfType> as <FacetType> fn
    Name(...)`) implements the associated function `Name` of the interface or
    named constraint (such as `fn Equal` in `Core.EqWith`, `fn Compare` in
    `Core.OrderedWith`, `fn Convert` in `Core.As` and `Core.ImplicitAs`, or the
    explicit name of a primary function). That member must be a function and
    must be the only non-defaulted member of the interface or named constraint
    (aside from members satisfied by a `where` clause or deduced from the
    signature).

```carbon
impl Meters as Core.DivWith fn (self, other: Seconds) -> MetersPerSecond {
  return {.value = self.value / other.value};
}

impl like MyFloat as Core.OrderedWith
    fn Compare(self, other: like MyInt) -> Core.Ordering {
  return ReverseOrdering(other.(Core.OrderedWith(MyFloat).Compare)(self));
}
```

#### Replacing `impl as` with `impl fn` and `impl Self as`

The `impl as` shorthand (omitting `Self` before `as` in a `class` or
`interface`) is replaced by `impl fn` for single-function implementations and
`impl Self as` for other `impl` declarations:

-   Whenever `as` appears in an `impl` declaration, a type expression before
    `as` is required. Inside a `class` or `interface`, non-`impl fn`
    declarations for the enclosing type write `Self` explicitly (`impl Self as
    Printable { ... }`, `extend impl Self as Printable;`, `impl Self as
    Printable = Adapter;`, and `extend impl Self as BaseInterface;` in an
    interface).
-   Within a `class` (or `interface`), `impl fn` provides a concise shorthand
    for an abbreviated `impl` for `Self`. Given `impl fn X.Y.Z(...)` (where the
    name has one or more dot-separated components, each potentially including
    arguments):
    -   If the `X.Y.Z` formation names an interface (or a named constraint with
        a primary function, such as `Printable`, `Core.Add`, or `Core.MulWith`),
        it must have a primary function, and `impl fn X.Y.Z(...)` is the same as
        `impl Self as X.Y.Z fn (...)` (so `impl fn X(...)` is `impl Self as X fn
        (...)` and `impl fn Core.Add(...)` is `impl Self as Core.Add fn (...)`).
    -   If `X.Y.Z` has more than one component and names a function member `Z`
        of an interface `X.Y` (such as `Comparable.Less`, `Core.Eq.Equal`,
        `Core.Ordered.Compare`, `Core.As(bool).Convert`, or
        `Core.ImplicitAs.Convert`), the last component (`Z`) is moved to the
        function name after `fn`: `impl Self as X.Y fn Z(...)` (so `impl fn
        X.Y(...)` is `impl Self as X fn Y(...)` and `impl fn Core.Eq.Equal(...)`
        is `impl Self as Core.Eq fn Equal(...)`). That member `Z` must be a
        function and must be the only non-defaulted member of the interface
        `X.Y` (aside from members that can be deduced from the signature).

```carbon
class Vec2 {
  var x: f32;
  var y: f32;

  impl fn Core.Default() -> Self { return {.x = 0.0, .y = 0.0}; }
  impl fn Core.Copy(self) -> Self { return {.x = self.x, .y = self.y}; }
  impl fn Core.Assign(ref self, other: Self) {
    self.x = other.x;
    self.y = other.y;
  }

  impl fn Core.Negate(self) -> Self { return {.x = -self.x, .y = -self.y}; }
  impl fn Core.Add(self, other: Self) -> Self {
    return {.x = self.x + other.x, .y = self.y + other.y};
  }
  impl fn Core.MulWith(self, scale: like f32) -> Self {
    return {.x = self.x * scale, .y = self.y * scale};
  }
  impl f32 as Core.MulWith fn (self, v: Vec2) -> Vec2 {
    return {.x = self * v.x, .y = self * v.y};
  }

  impl fn Core.Eq.Equal(self, other: Self) -> bool {
    return self.x == other.x and self.y == other.y;
  }
  impl fn Core.As(bool).Convert(self) -> bool {
    return self.x != 0.0 or self.y != 0.0;
  }
}
```

#### Deducing interface arguments and associated constants from the signature

In an abbreviated `impl` declaration (`impl <SelfType> as <FacetType> fn ...`
or its `impl fn ...` shorthand), omitted interface arguments and unassigned
associated constants are deduced by matching the `impl`'s `fn` signature against
the interface function's signature:

-   **Interface argument deduction**: When the interface is parameterized and
    its argument list `(U, ...)` is omitted (for example, `impl fn
    Core.MulWith(self, scale: f32) -> Self`, `impl fn Core.Call(self, x: f64) ->
    f64`, `impl fn Core.ImplicitAs.Convert(self) -> StringView`, or `impl Meters
    as Core.DivWith fn (self, other: Seconds) -> MetersPerSecond`), the omitted
    interface arguments are deduced by matching the parameter types and return
    type of the `impl`'s `fn` signature against the interface's `fn` signature.
    Every omitted interface parameter must be determined by the signature (or
    have a default). Interface arguments may also be written explicitly on the
    interface (such as `impl fn Core.As(bool).Convert(self) -> bool` or `impl
    f32 as Core.MulWith(Vec2) fn (self, v: Vec2) -> Vec2`).
-   **`like` on parameter types**: In the `fn` parameter list of an abbreviated
    `impl`, a parameter's type may be written `like T` (such as `impl fn
    Core.MulWith(self, scale: like f32) -> Self`). When matched against an
    omitted interface parameter `U: type`, `U` is deduced as `like T` (so `impl
    fn Core.MulWith(self, scale: like f32) -> Self` is equivalent to `impl Self
    as Core.MulWith(like f32) fn (self, scale: f32) -> Self`), while within the
    function signature and body the parameter (`scale`) has type `T` (`f32`).
-   **Associated constant deduction**: Any non-`final` associated constants
    declared by the interface itself (including those with `default` values,
    such as `Result` in `Core.AddWith`) that appear in deducible positions of
    the implemented function's signature in the interface are deduced from the
    `impl`'s `fn` signature. For example, in `impl Meters as Core.DivWith fn
    (self, other: Seconds) -> MetersPerSecond`, matching `-> MetersPerSecond`
    against `-> Result` deduces `.Result = MetersPerSecond` without a `where
    .Result = MetersPerSecond` clause.
-   Omitting the `->` return type clause on the `impl`'s `fn` signature is
    treated as `-> ()` for deduction, so matching an omitted return clause
    against `-> Result` deduces `.Result = ()` (as in `impl Func(Arg) as
    Core.Call fn (self, arg: Arg)`). Return type deduction (`-> auto` or `=>
    expr`) cannot be used to deduce an interface argument or associated
    constant.
-   Associated constants belonging to a required interface (such as
    `Bind(T).Result` in `BindToValue(T)`) are not defined by the `impl` of the
    extending interface; the `impl`'s `fn` signature must match the value
    established by the implementation of the required interface.
-   Deduced interface arguments and associated constants belong to the `impl`,
    so they cannot depend on generic parameters of the `fn` itself (though they
    may depend on `forall` parameters of the `impl` or enclosing generic
    parameters).
-   Signature deduction only occurs in the abbreviated `impl ... fn ...` (and
    `impl fn ...`) forms. When `{ ... }` braces are used on an `impl`, interface
    arguments must be written on the interface and associated constants must be
    specified with a `where` constraint or use their defaults.
-   In a class-scope `impl` that specifies an explicit type before `as` other
    than `Self` (such as `impl f32 as Core.MulWith fn (self, v: Vec2) -> Vec2`
    inside `class Vec2`), `Self` after `as` refers to the `impl`'s self type
    (`f32`) rather than the enclosing class (`Vec2`). To prevent confusion with
    the enclosing class, using `Self` anywhere after `as` in a class-scope
    `impl` whose explicit self type is not `Self` is an error.

### Interface extension, anonymous aliases, and named constraints

When an `interface` or `constraint` extends another interface `I` that has a
primary function, the two extension mechanisms from
[#5337](/proposals/p005337-interface-extension-and-final-impl-update.md) behave
as follows:

-   `extend require impls I`: Aliases `I`'s primary function into the extending
    scope under its qualified name `I` (`alias I = I.I;`). It does not become
    the primary function of the extending `interface` or `constraint`. This
    allows a constraint or interface to extend multiple interfaces with primary
    functions (such as `AddWith(Self)` and `SubWith(Self)`) without their
    primary functions colliding. If a scope extends multiple instantiations of
    the same parameterized interface (such as `AddWith(Self)` and
    `AddWith(Other)`), the conflicting `AddWith` names from `extend` are dropped
    unless disambiguated with an explicit `alias`.
-   `extend impl Self as I` (or `extend final impl Self as I`) in `interface J`:
    Copies `I`'s primary function into `J` as `J`'s primary function (taking the
    qualified name `J`), and the generated blanket `impl` of `I` forwards `I`'s
    primary function to `J`'s primary function (`U.impl(J.J)`). An interface `J`
    cannot use `extend impl Self as` for multiple interfaces that have primary
    functions.

In addition, an `interface` or `constraint` may declare at most one _anonymous
`alias`_ by omitting the alias name before `=`:

```carbon
constraint Core.Add {
  extend require impls Core.AddWith(Self) where .Result = Self;
  alias = Core.AddWith(Self).AddWith;
}
```

Like a primary function, an anonymous `alias` implicitly takes the name of the
enclosing `interface` or `constraint` for qualified member lookup only (such as
`x.Add(y)`, `x.(Core.Add.Add)(y)`, or `(T as Core.Add).Add(x, y)`), and is never
found by unqualified lookup. Implementing a named constraint uses the
constraint's names and supports abbreviated `impl` syntax
(`impl fn Core.Add(self, other: Self) -> Self` or
`impl T as Core.Add fn (self, other: Self) -> Self`) when the constraint has a
primary function by way of an anonymous `alias`.

### Out-of-line definitions

An abbreviated `impl` ending in `;` defines the `impl` (fixing its deduced
interface arguments and associated constants) and forward-declares its function
for out-of-line definition, whereas `impl Self as I;` (without `fn`)
forward-declares the `impl` itself:

-   To define a class-scope `impl fn` (or a forward-declared single-function
    `impl`) out-of-line, the class scope is re-entered with `Class.(...)` using
    either `impl fn Class.(X.Y.Z)(...) { ... }` or
    `impl Class.(Self as ...) fn ... { ... }`:
    -   In `impl fn Class.(X.Y.Z)(...) { ... }`, the parenthesized name
        `.(X.Y.Z)` follows the same rule as in-class `impl fn X.Y.Z(...)`: it is
        equivalent to `impl Class.(Self as X.Y.Z) fn (...) { ... }` when `X.Y.Z`
        names an interface (or named constraint) with a primary function (such
        as `Core.Copy` or `Core.AddWith`), and to
        `impl Class.(Self as X.Y) fn Z(...) { ... }` when `X.Y.Z` has more than
        one component and names a function member `Z` of an interface `X.Y`
        (such as `Core.Eq.Equal` or `Core.As(i32).Convert`).
    -   Consistent with
        [compound member access](/docs/design/expressions/member_access.md#compound-member-access)
        (`x.(Core.Copy)()`) and
        [class-scope `impl` redeclarations](/proposals/p005366-the-name-of-an-impl-in-class-scope.md)
        (`impl Class.(Self as Core.Copy)`), parentheses are required around
        `.(X.Y.Z)` because the interface name is typically package- or
        namespace-qualified (such as `Core.Copy`, `Core.AddWith`, or
        `Core.Eq.Equal`) rather than a member of `Class`. The parentheses
        unambiguously separate the enclosing class `Class` (which may itself be
        qualified or parameterized, such as `NS.MyType(T: type)`) from the
        interface or member path `X.Y.Z` (which is evaluated in the scope of
        `Class`), and avoid any ambiguity with in-class
        `impl fn Package.Interface(...)`.

    ```carbon
    class MyType {
      impl fn Core.Copy(self) -> Self;
      impl fn Core.AddWith(self, other: like OtherType) -> Self;
      impl fn Core.Eq.Equal(self, other: Self) -> bool;
    }

    impl fn MyType.(Core.Copy)(self) -> Self { ... }
    impl fn MyType.(Core.AddWith)(self, other: like OtherType) -> Self { ... }
    impl fn MyType.(Core.Eq.Equal)(self, other: Self) -> bool { ... }

    // Or equivalently:
    // impl MyType.(Self as Core.Copy) fn (self) -> Self { ... }
    // impl MyType.(Self as Core.AddWith)
    //     fn (self, other: like OtherType) -> Self { ... }
    // impl MyType.(Self as Core.Eq)
    //     fn Equal(self, other: Self) -> bool { ... }
    ```

-   A function forward-declared in an `impl` or as a `default fn` inside an
    `interface` can also be defined out-of-line using a standalone `fn`
    definition that explicitly specifies the function name (`.InterfaceName` for
    a primary function):

    ```carbon
    class MyType {
      impl fn Core.SubWith(self, other: OtherType) -> MyType;
    }

    fn MyType.(Self as Core.SubWith(OtherType)).SubWith(
        self, other: OtherType) -> MyType { ... }

    // For a file-scope `impl MyType as Core.SubWith fn (self, other: OtherType) -> MyType;`:
    fn (MyType as Core.SubWith(OtherType)).SubWith(
        self, other: OtherType) -> MyType { ... }

    // For a `default fn` in `interface Core.AddWith(U: type)`:
    fn Core.AddWith(U: type).AddWith(self, other: U) -> Result { ... }
    ```

### Updating `Op` interfaces

Existing single-function interfaces switch from `Op` to an unnamed primary
function when the interface name works well as a method name on an implementing
type:

-   Switch from `Op` to an unnamed primary function:
    -   Arithmetic interfaces: `Core.Negate`, `Core.AddWith`, `Core.SubWith`,
        `Core.MulWith`, `Core.DivWith`, `Core.ModWith`.
    -   Bitwise interfaces: `Core.BitComplement`, `Core.BitAndWith`,
        `Core.BitOrWith`, `Core.BitXorWith`, `Core.LeftShiftWith`,
        `Core.RightShiftWith`.
    -   Assignment and increment/decrement interfaces: `Core.AssignWith`,
        `Core.AddAssignWith`, `Core.SubAssignWith`, `Core.MulAssignWith`,
        `Core.DivAssignWith`, `Core.ModAssignWith`, `Core.BitAndAssignWith`,
        `Core.BitOrAssignWith`, `Core.BitXorAssignWith`,
        `Core.LeftShiftAssignWith`, `Core.RightShiftAssignWith`, `Core.Inc`,
        `Core.Dec`.
    -   Callable interface: `Core.Call`.
    -   Pointer dereference interface: `Core.Deref`.
    -   Member binding interfaces: `Core.BindToValue`, `Core.BindToRef`.
    -   Lifecycle and pattern matching interfaces: `Core.Destroy`, `Core.Copy`,
        `Core.Match`.
-   Keep an explicit method name (while supporting `impl fn` and abbreviated
    `impl` syntax by specifying the qualified method name):
    -   Conversion interfaces (`Core.As`, `Core.ImplicitAs`,
        `Core.ReferenceImplicitAs`) keep `fn Convert`, because `x.As()` is not a
        descriptive method name on an implementing type (abbreviated as
        `impl fn Core.As(Dest).Convert(self) -> Dest` or
        `impl fn Core.ImplicitAs.Convert(self) -> Dest`).
    -   Comparison interfaces (`Core.EqWith`, `Core.OrderedWith`) have multiple
        methods (`Equal` and `default fn NotEqual`; `Compare` and defaulted
        relational methods) with distinct names and are unchanged (abbreviated
        as `impl fn Core.Eq.Equal(self, other: Self) -> bool` and
        `impl fn Core.Ordered.Compare(self, other: Self) -> Core.Ordering`).

## Details

The normative design updates are integrated directly into the design
documentation in this pull request:

-   [`/docs/design/generics/overview.md`](/docs/design/generics/overview.md),
    [`/docs/design/generics/details.md`](/docs/design/generics/details.md),
    [`/docs/design/generics/terminology.md`](/docs/design/generics/terminology.md),
    and
    [`/docs/design/generics/appendix-rewrite-constraints.md`](/docs/design/generics/appendix-rewrite-constraints.md):
    Primary interface functions, abbreviated `impl` declarations, `impl fn` and
    `impl Self as`, interface argument and associated constant deduction,
    interface extension and anonymous `alias` rules, and out-of-line
    redeclaration matching.
-   [`/docs/design/functions.md`](/docs/design/functions.md): Function syntax,
    redeclaration matching for unnamed primary functions, and the `Call`
    interface.
-   [`/docs/design/expressions/member_access.md`](/docs/design/expressions/member_access.md),
    [`/docs/design/expressions/arithmetic.md`](/docs/design/expressions/arithmetic.md),
    [`/docs/design/expressions/bitwise.md`](/docs/design/expressions/bitwise.md),
    [`/docs/design/expressions/comparison_operators.md`](/docs/design/expressions/comparison_operators.md),
    [`/docs/design/expressions/indexing.md`](/docs/design/expressions/indexing.md),
    [`/docs/design/assignment.md`](/docs/design/assignment.md),
    [`/docs/design/classes.md`](/docs/design/classes.md),
    [`/docs/design/declaring_entities.md`](/docs/design/declaring_entities.md),
    [`/docs/design/pattern_matching.md`](/docs/design/pattern_matching.md),
    [`/docs/design/sum_types.md`](/docs/design/sum_types.md),
    [`/docs/design/values.md`](/docs/design/values.md), and
    [`/docs/design/README.md`](/docs/design/README.md): Updates to operator,
    callable, member-binding, and `Match`/`Copy` interfaces and `impl`
    declarations.

## Rationale

-   [Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Programmers implementing a single-function interface in a class (such as an
    overloaded operator, lifecycle operation, or `Core.Call`) can write `impl fn
    Core.AddWith(self, other: Other) -> ReturnType` without repeating operand
    types or first learning associated type rewrite syntax (`where .Result =
    ReturnType`). When an implementation needs multiple declarations or explicit
    associated constants not present in the signature, it transitions directly
    to `impl Self as ...` with a `where` clause or a braced `impl` body.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Replacing `Op` with a primary function that takes the interface's name gives
    methods meaningful names (`x.AddWith(y)`, `x.Negate()`, `f.Call(...)`) when
    extended into classes or used in generic code, and avoids collisions when a
    class extends multiple operator interfaces. Defining `impl fn X.Y.Z` as
    `impl Self as X.Y.Z fn` when `X.Y.Z` names an interface with a primary
    function (such as `impl fn Core.Add(...)`) and as `impl Self as X.Y fn Z`
    when `Z` is a function member of `X.Y` (such as `impl fn
    Core.Eq.Equal(...)`) gives class-body operator and lifecycle implementations
    the familiar structure of member function declarations while removing
    duplicate parameter and return types.
-   Balancing [Only one way](/docs/project/principles/one_way.md) and
    [Low context-sensitivity](/docs/project/principles/low_context_sensitivity.md):
    Replacing `impl as` with `impl fn` (and requiring `impl Self as` when `as`
    is written) ensures that `as` in an `impl` declaration always has a self
    type on the left and a facet type on the right, and avoids having both
    `impl as X fn (...)` and `impl fn X(...)` compete as shorthands in class
    scope. Using `impl fn Class.(X.Y.Z)(...)` for out-of-line definitions
    reuses Carbon's compound member access and `#5366` scope re-entry syntax to
    unambiguously separate the class from package-qualified interface names
    (`Core.Copy`, `Core.Eq.Equal`). We also keep the surface area minimal by
    leaving `interface` declarations braced, keeping compound member access
    (`x.(I.I)(y)`) and standalone out-of-line function definitions
    (`fn (T as I).I(...)`) explicit, and restricting the implicit function name
    to qualified member lookup so unqualified lookup is never shadowed.

## Alternatives considered

### Keep placeholder names like `Op` or repeat the interface name

We could keep the current design where every interface function has an explicit
identifier, either a placeholder like `Op` or a repetition of the interface
name.

Advantages:

-   Requires no new syntax or name lookup rules for unnamed functions.

Disadvantages:

-   `Op` works poorly with `extend impl` in a class: extending
    `Core.AddWith(Point)` into `Point` injects a method named `Op` into `Point`,
    colliding with any other extended operator interface and producing an
    uninformative `p.Op(q)` method name.
-   Repeating the interface name in both `interface` and `impl` declarations
    (`interface Negate { fn Negate... }`, `impl Self as Negate { fn Negate...
    }`) is verbose and causes the function name to shadow the interface name in
    unqualified lookup inside the interface body unless special lookup rules are
    introduced anyway.

### Keep `impl as` alongside or instead of `impl fn`

We considered keeping `impl as <Facet> fn (...)` (or omitting `as` in `impl
<Facet> fn (...)`) in class scope instead of replacing `impl as` with `impl fn`
and `impl Self as`.

Advantages:

-   Preserves `impl as <Facet> { ... }` for braced `impl` declarations in class
    and interface scope without writing `Self` before `as`.

Disadvantages:

-   Writing `impl as Core.MulWith(like f32) fn (self, scale: f32) -> Self`
    places an unnamed `fn (` mid-line and separates the operation name from its
    parameter list, making classes with many operator and lifecycle
    implementations harder to scan than
    `impl fn Core.MulWith(self, scale: like f32) -> Self`.
-   Keeping both `impl as X fn (...)` and `impl fn X(...)` would provide two
    competing shorthand spellings for the same in-class declaration. Requiring
    `Self` before `as` (`impl Self as ...`) while providing `impl fn` for
    single-function implementations makes `as` consistently binary (`<Type> as
    <Facet>`).

### Also abbreviate `interface` declarations

We considered also allowing `interface` declarations to omit `{ ... }` braces
(such as `interface Inc fn (ref self);`), combined with an inline return-type
binding syntax `-> (Result: type)` to declare an associated type on the
interface (such as `interface Deref fn (self) -> (Result: type);`).

Advantages:

-   Provides syntactic symmetry between single-function `interface` declarations
    and single-function `impl` declarations.
-   Makes simple single-function interfaces without defaulted associated types a
    single declaration without braces.

Disadvantages:

-   Single-function interfaces in Carbon very commonly require associated types
    with `default` values (such as `default let Result: type = Self;` across all
    arithmetic and bitwise operator interfaces). Supporting those in an
    abbreviated `interface` header would require introducing additional complex
    features (such as default values inside return-type patterns), whereas
    leaving them braced splits single-function interfaces into two styles.
-   `interface` declarations are written far less frequently than `impl`
    declarations, so the brevity benefit of omitting braces on `interface` does
    not justify the extra syntax and grammar rules.

### Omit the function name in compound member access

We considered allowing compound member access `x.(I)(y)` and `impl` member
access `T.impl(I)` to implicitly designate the primary function `I.I` when `I`
is a facet type with a primary function (or anonymous `alias`).

Advantages:

-   Avoids repeating the interface name in qualified calls such as
    `x.(Core.Negate.Negate)()` or `x.(Core.AddWith(U).AddWith)(y)`.

Disadvantages:

-   Overloads facet types in member access to mean either a facet type or an
    associated function depending on whether the interface happens to declare a
    primary function.
-   Most calls to primary functions use operator syntax (`-x`, `x + y`) or
    simple member access on an extended class or constrained generic parameter
    (`x.Negate()`, `x.AddWith(y)`), so explicit compound member access is
    infrequent and benefits from naming the member explicitly as
    `x.(Core.AddWith(U).AddWith)(y)` as proposed in
    [#4711](https://github.com/carbon-language/carbon-lang/issues/4711).

### Omit the function name in standalone out-of-line `fn` definitions

We considered allowing standalone out-of-line `fn` definitions of an `impl`'s
primary function to omit `.InterfaceName` after the parenthesized `impl` scope,
as in
`fn (MyType as Core.AddWith(OtherType))(self, other: OtherType) -> MyType`.

Advantages:

-   Avoids repeating `.AddWith` after `(MyType as Core.AddWith(OtherType))`.

Disadvantages:

-   Looks like a call to a parenthesized expression rather than a member
    function definition, and cannot be extended to out-of-line `default fn`
    definitions on parameterized interfaces (`fn Core.AddWith(U: type)(...)` is
    ambiguous with a function parameter list).
-   An out-of-line abbreviated `impl` definition
    (`impl fn MyType.(Core.AddWith)(...) { ... }` or
    `impl MyType.(Self as Core.AddWith) fn (...) { ... }`) already avoids
    repeating `.AddWith`, so a second shorthand on standalone `fn` definitions
    is unnecessary.

### Make the implicit function name visible to unqualified lookup

We considered making the implicit name of a primary function visible to
unqualified name lookup inside the `interface`, an extending `interface` or
`constraint`, or an `impl`.

Advantages:

-   Allows unqualified recursive calls or references to the primary function
    inside the `interface` or `impl`.

Disadvantages:

-   In `interface AddWith(U: type)`, if the primary function were visible to
    unqualified lookup as `AddWith`, any unqualified mention of `AddWith` inside
    the interface or an extending constraint (such as referring to
    `AddWith(Other)` in a constraint or `where` clause) would find the function
    instead of the interface. Restricting the implicit name to qualified member
    lookup (`self.AddWith(...)`, `Self.AddWith`) keeps unqualified `AddWith`
    unambiguous in all scopes.

### Call primary functions without naming the member

We considered allowing a value of a type constrained by a single-function
interface `I` to be called directly as `x(y)` or with a placeholder member
syntax such as `x._(y)`.

Advantages:

-   Avoids naming the interface at the call site when invoking the primary
    function.

Disadvantages:

-   Calling `x(y)` directly conflicts with the `Core.Call` interface when a type
    implements both `Core.Call` and another single-function interface, or when a
    generic parameter is constrained by multiple interfaces.
-   Using `x._(y)` fails to disambiguate when two extended interfaces in a
    constraint or class both have primary functions. Giving the primary function
    the qualified name of its interface (`x.I(y)` and `x.(I.I)(y)`) provides a
    clear name at the call site and disambiguates multiple interfaces.

### Deduce associated constants in braced `impl` definitions

We considered also deducing associated constants like `Result` from the primary
function signature inside a braced `impl` body:

```carbon
impl Point as Core.AddWith(Point) {
  fn (self, other: Point) -> Point { ... }
}
```

Advantages:

-   Allows omitting `where .Result = Point` even when `{ ... }` braces are
    written on the `impl`.

Disadvantages:

-   A braced `impl` body may contain multiple declarations, aliases, or forward
    declarations, making it less obvious that the facet type of the `impl` is
    determined by a declaration inside the braces. Consistent with
    [#731](/proposals/p000731-generics-details-2-adapters-associated-types-parameterized-interfaces.md#syntax-for-associated-constants),
    associated constant deduction is restricted to the single-declaration
    abbreviated form (`impl Point as Core.AddWith fn ...` and
    `impl fn Core.AddWith...`).

### Dedicated `op` keyword

Early discussions in
[#4711](https://github.com/carbon-language/carbon-lang/issues/4711) considered
using an `op` keyword in place of `fn` for operator interfaces and
implementations.

Advantages:

-   Visually distinguishes primary/operator functions from named `fn`
    declarations without omitting an identifier after `fn`.

Disadvantages:

-   Single-function interfaces are not limited to built-in operators (for
    example, `Call`, `Match`, `Copy`, `Destroy`, or user-defined single-action
    interfaces).
-   Reusing `fn` aligns with Carbon's existing function declaration and
    anonymous function (lambda) syntax from
    [#3848](/proposals/p003848-lambdas.md) without introducing a new declaration
    keyword.
