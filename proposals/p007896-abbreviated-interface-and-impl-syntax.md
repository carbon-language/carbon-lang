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
    -   [Abbreviated `impl` declarations and associated constant deduction](#abbreviated-impl-declarations-and-associated-constant-deduction)
    -   [Interface extension, anonymous aliases, and named constraints](#interface-extension-anonymous-aliases-and-named-constraints)
    -   [Out-of-line definitions](#out-of-line-definitions)
    -   [Updating `Op` interfaces](#updating-op-interfaces)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Future work](#future-work)
    -   [Deduce interface arguments from the function signature](#deduce-interface-arguments-from-the-function-signature)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep placeholder names like `Op` or repeat the interface name](#keep-placeholder-names-like-op-or-repeat-the-interface-name)
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
constraint:

```carbon
interface Negate {
  default let Result: type = Self;
  fn (self) -> Result;
}

interface AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}

class Point {
  var x: i32;
  var y: i32;

  impl as Negate fn (self) -> Self {
    return {.x = -self.x, .y = -self.y};
  }
  impl as Add fn (self, other: Self) -> Self {
    return {.x = self.x + other.x, .y = self.y + other.y};
  }
  impl as Eq fn Equal(self, other: Self) -> bool {
    return self.x == other.x and self.y == other.y;
  }
}
```

An unnamed `fn` declaration in an `interface` implicitly takes the name of the
enclosing `interface` for qualified member lookup (such as
`p.(AddWith(Point).AddWith)(q)`, `(Point as AddWith(Point)).AddWith`, or
`p.Negate()` and `p.AddWith(q)` in generic code or when the `impl` is extended).
In an abbreviated `impl`, the function may either omit its name (to implement
the interface's primary function) or specify a member function name (such as
`fn Equal` in `Eq` or `fn Convert` in `ImplicitAs`), and any associated
constants that appear in that function's signature are deduced from the `impl`
function's signature without a `where` clause.

This replaces placeholder method names like `Op` across operator, callable,
lifecycle, and other single-operation interfaces whose names work as method
names on implementing types.

## Problem

Many interfaces in Carbon represent a single operation: arithmetic and bitwise
operators (`AddWith`, `Negate`), assignment and compound assignment
(`AssignWith`, `AddAssignWith`, `Inc`, `Dec`), callables (`Call`), pointer
dereferencing (`Deref`), lifecycle operations (`Copy`, `Destroy`), and pattern
matching (`Match`). In the current design, declaring and implementing these
interfaces has two main problems:

1.  Placeholder or redundant function names: Because every associated function
    in an interface requires an explicit identifier, single-function interfaces
    either use a placeholder name like `Op` or repeat the interface's name
    (`interface EqualWith(U: type) { fn EqualWith... }`). When an interface is
    implemented with `extend impl as` in a class, a placeholder name like `Op`
    cannot be usefully extended into the class's scope without colliding with
    other interfaces using `Op` or exposing an unhelpful `x.Op()` method.
    Conversely, repeating the interface name in both the `interface` and `impl`
    declarations is verbose and causes the function name to shadow the interface
    name in unqualified lookup inside the interface body.
2.  Boilerplate in single-function `impl` declarations: Implementing a
    single-operation interface whose return type differs from the default
    requires both a `where .Result = ...` clause on the facet type and braces
    around a single `fn` definition:

    ```carbon
    // Current syntax:
    impl Meters as DivWith(Seconds) where .Result = MetersPerSecond {
      fn Op(self, other: Seconds) -> MetersPerSecond {
        return {.value = self.value / other.value};
      }
    }
    ```

    Here, the return type `MetersPerSecond` must be written twice, once in
    `where .Result = MetersPerSecond` and again in `-> MetersPerSecond`, and
    programmers writing an operator implementation must learn associated type
    rewrite constraints before returning a type other than `Self`.

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
    out-of-line as `impl Class.(as Interface)`.
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
interface AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}
```

This function is the _primary function_ of the interface:

-   It implicitly has the same name as the enclosing `interface` for qualified
    member lookup (such as `x.AddWith(y)`, `Self.AddWith(x, y)`,
    `x.(AddWith(U).AddWith)(y)`, `(T as AddWith(U)).AddWith(x, y)`, or
    `AddWith(U).AddWith(x, y)`).
-   Its implicit name is never found by unqualified name lookup in any scope,
    including inside the `interface`, an extending `interface`, `constraint`, or
    `class`, or an `impl`. Within those scopes, the unqualified identifier
    `AddWith` refers to the interface itself. To call or reference the primary
    function, code uses qualified member lookup (such as `self.AddWith(other)`
    in a class that extends the implementation,
    `self.(AddWith(U).AddWith)(other)` in a non-extending `impl`, or
    `Self.AddWith`).

An `impl` of an interface with a primary function may omit the function name on
at most one `fn` declaration to implement the primary function, or it may write
the interface's name explicitly (`fn AddWith(self, other: U) -> Self`).

### Abbreviated `impl` declarations and associated constant deduction

An `impl` declaration (including `extend impl` and `final impl`) may omit the
`{ ... }` braces and follow the facet type directly with a `fn` declaration or
definition:

-   Omitting the function name after `fn` (`impl ... as Facet fn (...)`)
    implements the primary function of the interface or named constraint.
-   Specifying a function name after `fn` (`impl ... as Facet fn Name(...)`)
    implements the associated function `Name` of the interface or named
    constraint (such as `fn Equal` in `EqWith`, `fn Compare` in `OrderedWith`,
    `fn Convert` in `As` and `ImplicitAs`, or the explicit name of a primary
    function).

```carbon
class CustomInt {
  var value: i32;

  impl as Add fn (self, other: Self) -> Self {
    return {.value = self.value + other.value};
  }

  impl as Inc fn (ref self) {
    ++self.value;
  }

  impl as Eq fn Equal(self, other: Self) -> bool {
    return self.value == other.value;
  }
}
```

Abbreviated `impl` syntax is allowed when all other requirements of the
interface or named constraint are satisfied by defaults, a `where` clause on
the `impl`, or deduction from the function's signature.

In an abbreviated `impl` declaration, any non-`final` associated constants
declared by the interface itself (including those with `default` values, such as
`Result` in `AddWith`) that appear in deducible positions of the implemented
function's signature in the interface are deduced by matching the `impl`'s `fn`
signature against the interface's `fn` signature:

-   In `impl Meters as DivWith(Seconds) fn (self, other: Seconds) ->
    MetersPerSecond`, matching `-> MetersPerSecond` against `-> Result` deduces
    `.Result = MetersPerSecond` without a `where .Result = MetersPerSecond`
    clause.
-   Omitting the `->` return type clause on the `impl`'s `fn` signature is
    treated as `-> ()` for deduction, so matching an omitted return clause
    against `-> Result` deduces `.Result = ()` (as in
    `impl Func(Arg) as Call(Arg) fn (self, arg: Arg)`). Return type deduction
    (`-> auto` or `=> expr`) cannot be used to deduce an associated constant.
-   Associated constants belonging to a required interface (such as
    `Bind(T).Result` in `BindToValue(T)`) are not defined by the `impl` of the
    extending interface; the `impl`'s `fn` signature must match the value
    established by the implementation of the required interface.
-   Deduced associated constants belong to the `impl`, so they cannot depend on
    generic parameters of the `fn` itself (though they may depend on `forall`
    parameters of the `impl` or enclosing generic parameters).
-   Associated constant deduction only occurs in the abbreviated `impl ... fn
    ...` form. When `{ ... }` braces are used on an `impl`, associated constants
    must be specified with a `where` constraint or use their defaults.
-   In a class-scope `impl` that specifies an explicit type before `as` other
    than `Self` (such as `impl f32 as MulWith(Vec2)` inside `class Vec2`),
    `Self` after `as` refers to the `impl`'s self type (`f32`) rather than the
    enclosing class (`Vec2`). To prevent confusion with the enclosing class,
    using `Self` in the facet type or abbreviated `fn` signature of a
    class-scope `impl` whose explicit self type is not `Self` is an error.

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
-   `extend impl as I` (or `extend final impl as I`) in `interface J`: Copies
    `I`'s primary function into `J` as `J`'s primary function (taking the
    qualified name `J`), and the generated blanket `impl` of `I` forwards `I`'s
    primary function to `J`'s primary function (`U.impl(J.J)`). An interface `J`
    cannot use `extend impl as` for multiple interfaces that have primary
    functions.

In addition, an `interface` or `constraint` may declare at most one _anonymous
`alias`_ by omitting the alias name before `=`:

```carbon
constraint Add {
  extend require impls AddWith(Self) where .Result = Self;
  alias = AddWith(Self).AddWith;
}
```

Like a primary function, an anonymous `alias` implicitly takes the name of the
enclosing `interface` or `constraint` for qualified member lookup only (such as
`x.Add(y)`, `x.(Add.Add)(y)`, or `(T as Add).Add(x, y)`), and is never found by
unqualified lookup. Implementing a named constraint uses the constraint's names
and supports abbreviated `impl` syntax when the constraint has a primary
function by way of an anonymous `alias`:

```carbon
impl T as Add fn (self, other: Self) -> Self { ... }
```

### Out-of-line definitions

An abbreviated `impl` ending in `;` defines the `impl` (fixing its associated
constants) and forward-declares its function for out-of-line definition, whereas
`impl as I;` (without `fn`) forward-declares the `impl` itself:

-   A forward-declared `impl` (`impl as AddWith(OtherType);`) can be defined
    out-of-line using an abbreviated `impl` definition, provided any deduced
    associated constants match the values established by the first declaration:

    ```carbon
    class MyType {
      impl as AddWith(OtherType);
    }

    impl MyType.(as AddWith(OtherType))
        fn (self, other: OtherType) -> MyType { ... }
    ```

-   A function forward-declared by an abbreviated `impl`
    (`impl as AddWith(OtherType) fn (...);`), inside a braced `impl`, or as a
    `default fn` inside an `interface` can be defined out-of-line using a
    standalone `fn` definition that explicitly specifies the function name
    (`.InterfaceName` for a primary function):

    ```carbon
    class MyType {
      impl as AddWith(OtherType)
          fn (self, other: OtherType) -> MyType;
    }

    fn MyType.(as AddWith(OtherType)).AddWith(
        self, other: OtherType) -> MyType { ... }

    // For a file-scope `impl MyType as SubWith(OtherType)`:
    fn (MyType as SubWith(OtherType)).SubWith(
        self, other: OtherType) -> MyType { ... }

    // For a `default fn` in `interface AddWith(U: type)`:
    fn AddWith(U: type).AddWith(self, other: U) -> Result { ... }
    ```

### Updating `Op` interfaces

Existing single-function interfaces switch from `Op` to an unnamed primary
function when the interface name works well as a method name on an implementing
type:

-   Switch from `Op` to an unnamed primary function:
    -   Arithmetic interfaces: `Negate`, `AddWith`, `SubWith`, `MulWith`,
        `DivWith`, `ModWith`.
    -   Bitwise interfaces: `BitComplement`, `BitAndWith`, `BitOrWith`,
        `BitXorWith`, `LeftShiftWith`, `RightShiftWith`.
    -   Assignment and increment/decrement interfaces: `AssignWith`,
        `AddAssignWith`, `SubAssignWith`, `MulAssignWith`, `DivAssignWith`,
        `ModAssignWith`, `BitAndAssignWith`, `BitOrAssignWith`,
        `BitXorAssignWith`, `LeftShiftAssignWith`, `RightShiftAssignWith`,
        `Inc`, `Dec`.
    -   Callable interface: `Call`.
    -   Pointer dereference interface: `Deref`.
    -   Member binding interfaces: `BindToValue`, `BindToRef`.
    -   Lifecycle and pattern matching interfaces: `Destroy`, `Copy`, `Match`.
-   Keep an explicit method name (while supporting abbreviated `impl` syntax by
    specifying that method name after `fn`):
    -   Conversion interfaces (`As`, `ImplicitAs`, `ReferenceImplicitAs`) keep
        `fn Convert`, because `x.As()` is not a descriptive method name on an
        implementing type (abbreviated as
        `impl as ImplicitAs(Dest) fn Convert(self) -> Dest`).
    -   Comparison interfaces (`EqWith`, `OrderedWith`) have multiple methods
        (`Equal` and `default fn NotEqual`; `Compare` and defaulted relational
        methods) with distinct names and are unchanged (abbreviated as
        `impl as Eq fn Equal(self, other: Self) -> bool` and
        `impl as Ordered fn Compare(self, other: Self) -> Ordering`).

## Details

The normative design updates are integrated directly into the design
documentation in this pull request:

-   [`/docs/design/generics/overview.md`](/docs/design/generics/overview.md) and
    [`/docs/design/generics/details.md`](/docs/design/generics/details.md):
    Primary interface functions, abbreviated `impl` declarations, associated
    constant deduction, interface extension and anonymous `alias` rules, and
    out-of-line redeclaration matching.
-   [`/docs/design/functions.md`](/docs/design/functions.md): Function syntax,
    redeclaration matching for unnamed primary functions, and the `Call`
    interface.
-   [`/docs/design/expressions/member_access.md`](/docs/design/expressions/member_access.md),
    [`/docs/design/expressions/arithmetic.md`](/docs/design/expressions/arithmetic.md),
    [`/docs/design/expressions/bitwise.md`](/docs/design/expressions/bitwise.md),
    [`/docs/design/expressions/comparison_operators.md`](/docs/design/expressions/comparison_operators.md),
    [`/docs/design/assignment.md`](/docs/design/assignment.md),
    [`/docs/design/pattern_matching.md`](/docs/design/pattern_matching.md),
    [`/docs/design/sum_types.md`](/docs/design/sum_types.md),
    [`/docs/design/values.md`](/docs/design/values.md), and
    [`/docs/design/README.md`](/docs/design/README.md): Updates to operator,
    callable, member-binding, and `Match`/`Copy` interfaces and their rewrites.

## Rationale

-   [Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Programmers implementing a single-function interface (such as an overloaded
    operator or `Call`) can write `impl ... fn (...) -> ReturnType` without
    first learning associated type rewrite syntax (`where .Result =
    ReturnType`). When an implementation needs multiple declarations or explicit
    associated constants not present in the signature, it transitions directly
    to a `where` clause or a braced `impl` body.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Replacing `Op` with a primary function that takes the interface's name gives
    methods meaningful names (`x.AddWith(y)`, `x.Negate()`, `f.Call(...)`) when
    extended into classes or used in generic code, and avoids collisions when a
    class extends multiple operator interfaces. Abbreviated `impl` syntax
    removes redundant braces and duplicate return type specifications across
    both primary-function interfaces and interfaces with a single required named
    function (`EqWith`, `OrderedWith`, `As`, `ImplicitAs`).
-   Balancing [Only one way](/docs/project/principles/one_way.md) and
    [Low context-sensitivity](/docs/project/principles/low_context_sensitivity.md):
    Allowing an omitted function name in `interface`/`impl` and omitting braces
    on a single-function `impl` adds a second spelling alongside braced `impl`
    blocks. We keep this surface area minimal by leaving `interface`
    declarations braced, keeping compound member access (`x.(I.I)(y)`) and
    standalone out-of-line function definitions (`fn (T as I).I(...)`) explicit,
    and restricting the implicit function name to qualified member lookup so
    unqualified lookup is never shadowed.

## Future work

### Deduce interface arguments from the function signature

When implementing a parameterized interface in an abbreviated `impl` (such as
`AddWith(U)`, `MulWith(like f32)`, `AssignWith(like StringView)`, `Call(Args)`,
or `ImplicitAs(Dest)`), the interface's type arguments are repeated in the `fn`
parameter list or return type:

```carbon
impl as MulWith(like f32) fn (self, scale: f32) -> Self { ... }
impl as AssignWith(like StringView) fn (ref self, sv: StringView);
impl as ImplicitAs(StringView) fn Convert(self) -> StringView { ... }
```

Future work could extend deduction in abbreviated `impl` declarations to deduce
omitted interface arguments from the `fn` signature, combined with allowing
`like` on parameter types as shorthand for `like` on the deduced interface
argument:

```carbon
impl as MulWith fn (self, scale: like f32) -> Self { ... }
impl as AssignWith fn (ref self, sv: like StringView);
impl as ImplicitAs fn Convert(self) -> StringView { ... }
```

Because a class may implement the same parameterized interface for multiple
argument types (such as `AssignWith(Self)` and `AssignWith(like StringView)`),
omitting the interface argument list means the `impl`'s identity is no longer
determined prior to `fn`. Designing this extension is left to future work so its
interaction with `impl` identity, redeclaration matching, and `like` parameter
syntax can be evaluated separately.

## Alternatives considered

### Keep placeholder names like `Op` or repeat the interface name

We could keep the current design where every interface function has an explicit
identifier, either a placeholder like `Op` or a repetition of the interface
name.

Advantages:

-   Requires no new syntax or name lookup rules for unnamed functions.

Disadvantages:

-   `Op` works poorly with `extend impl as` in a class: extending
    `AddWith(Point)` into `Point` injects a method named `Op` into `Point`,
    colliding with any other extended operator interface and producing an
    uninformative `p.Op(q)` method name.
-   Repeating the interface name in both `interface` and `impl` declarations
    (`interface Negate { fn Negate... }`, `impl as Negate { fn Negate... }`) is
    verbose and causes the function name to shadow the interface name in
    unqualified lookup inside the interface body unless special lookup rules are
    introduced anyway.

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
    `x.(Negate.Negate)()` or `x.(AddWith(U).AddWith)(y)`.

Disadvantages:

-   Overloads facet types in member access to mean either a facet type or an
    associated function depending on whether the interface happens to declare a
    primary function.
-   Most calls to primary functions use operator syntax (`-x`, `x + y`) or
    simple member access on an extended class or constrained generic parameter
    (`x.Negate()`, `x.AddWith(y)`), so explicit compound member access is
    infrequent and benefits from naming the member explicitly as
    `x.(AddWith(U).AddWith)(y)` as proposed in
    [#4711](https://github.com/carbon-language/carbon-lang/issues/4711).

### Omit the function name in standalone out-of-line `fn` definitions

We considered allowing standalone out-of-line `fn` definitions of an `impl`'s
primary function to omit `.InterfaceName` after the parenthesized `impl` scope,
as in `fn (MyType as AddWith(OtherType))(self, other: OtherType) -> MyType`.

Advantages:

-   Avoids repeating `.AddWith` after `(MyType as AddWith(OtherType))`.

Disadvantages:

-   Looks like a call to a parenthesized expression rather than a member
    function definition, and cannot be extended to out-of-line `default fn`
    definitions on parameterized interfaces (`fn AddWith(U: type)(...)` is
    ambiguous with a function parameter list).
-   An out-of-line abbreviated `impl` definition
    (`impl MyType.(as AddWith(OtherType)) fn (...) { ... }`) already avoids
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

-   Calling `x(y)` directly conflicts with the `Call` interface when a type
    implements both `Call` and another single-function interface, or when a
    generic parameter is constrained by multiple interfaces.
-   Using `x._(y)` fails to disambiguate when two extended interfaces in a
    constraint or class both have primary functions. Giving the primary function
    the qualified name of its interface (`x.I(y)` and `x.(I.I)(y)`) provides a
    clear name at the call site and disambiguates multiple interfaces.

### Deduce associated constants in braced `impl` definitions

We considered also deducing associated constants like `Result` from the primary
function signature inside a braced `impl` body:

```carbon
impl Point as AddWith(Point) {
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
    abbreviated form (`impl Point as AddWith(Point) fn ...`).

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
    example, `Call`, `Match`, `Copy`, or user-defined single-action interfaces).
-   Reusing `fn` with an omitted name aligns with Carbon's existing syntax for
    anonymous functions (lambdas) from [#3848](/proposals/p003848-lambdas.md)
    without introducing a new declaration keyword.
