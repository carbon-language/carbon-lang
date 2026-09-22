# Qualified names and member access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Simple member access](#simple-member-access)
    -   [Compound member access](#compound-member-access)
    -   [`impl` member access](#impl-member-access)
    -   [Restrictions](#restrictions)
-   [Name lookup](#name-lookup)
    -   [Package and namespace members](#package-and-namespace-members)
    -   [Types and extended types](#types-and-extended-types)
    -   [`extend`](#extend)
    -   [Values](#values)
        -   [Facet binding](#facet-binding)
        -   [Compile-time bindings](#compile-time-bindings)
    -   [Lookup ambiguity](#lookup-ambiguity)
-   [Tuple indexing](#tuple-indexing)
-   [`impl` lookup](#impl-lookup)
    -   [The `Self` type](#the-self-type)
    -   [Query lookup](#query-lookup)
    -   [Members of facets that are associated with different interfaces](#members-of-facets-that-are-associated-with-different-interfaces)
    -   [Note on the relationship between simple and compound member access](#note-on-the-relationship-between-simple-and-compound-member-access)
    -   [`impl` member access example](#impl-member-access-example)
        -   [Callables for member functions](#callables-for-member-functions)
-   [Instance binding](#instance-binding)
    -   [Member binding interfaces](#member-binding-interfaces)
    -   [Inheritance and other implicit conversions](#inheritance-and-other-implicit-conversions)
    -   [Data fields](#data-fields)
    -   [Generic type of a class member](#generic-type-of-a-class-member)
        -   [Methods](#methods)
        -   [Fields](#fields)
    -   [C++ pointer-to-member values](#c-pointer-to-member-values)
    -   [C++ operator overloading](#c-operator-overloading)
-   [Non-instance members](#non-instance-members)
    -   [Overloading](#overloading)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _qualified name_ is a [word](../lexical_conventions/words.md) that is preceded
by a period or a rightward arrow. The name is found within a contextually
determined entity:

-   In a member access expression, this is the entity preceding the period.
-   In a pointer member access expression, this is the entity pointed to by the
    pointer preceding the rightward arrow.
-   For a designator in a struct literal, the name is introduced as a member of
    the struct type.

A _member access expression_ allows a member of a value, type, interface,
namespace, and so on to be accessed by specifying a qualified name for the
member.

A member access expression is a _simple_ member access expression of the form:

-   _member-access-expression_ ::= _expression_ `.` _word_
-   _member-access-expression_ ::= _expression_ `->` _word_
-   _member-access-expression_ ::= _expression_ `.` _integer-literal_
-   _member-access-expression_ ::= _expression_ `->` _integer-literal_

a _compound_ member access of the form:

-   _member-access-expression_ ::= _expression_ `.` `(` _expression_ `)`
-   _member-access-expression_ ::= _expression_ `->` `(` _expression_ `)`

or an _`impl`_ member access of the form:

-   _member-access-expression_ ::= _expression_ `.` `impl` `(` _expression_ `)`
-   _member-access-expression_ ::= _expression_ `->` `impl` `(` _expression_ `)`

The _member name_ is the _word_, _integer-literal_, or the parenthesized
_expression_ in the member access expression. Compound and `impl` member
accesses allow specifying a qualified member name.

For example:

```carbon
namespace Widgets;

interface Widgets.Widget {
  fn Grow(ref self, factor: f64);
}

class Widgets.Cog {
  var size: i32;
  fn Make(size: i32) -> Self;
  extend impl as Widgets.Widget;
}

fn Widgets.GrowSomeCogs() {
  var cog1: Cog = Cog.Make(1);
  var cog2: Cog = typeof(cog1).Make(2);
  var cog_pointer: Cog* = &cog2;
  let cog1_size: i32 = cog1.size;
  cog1.Grow(1.5);
  cog2.(Cog.Grow)(cog1_size as f64);
  cog1.(Widget.Grow)(1.1);
  cog2.(Widgets.Cog.impl(Widgets.Widget.Grow))(1.9);
  cog_pointer->Grow(0.75);
  cog_pointer->(Widget.Grow)(1.2);
}
```

Note that `.` is used consistently for member access, whether it is applied to
an object, type, or namespace. This is in contrast to C++, which uses `::` for
types and namespaces.

Pointer member access expressions are those using a `->` instead of a `.` and
their semantics are exactly what would result from first dereferencing the
expression preceding the `->` and then forming a member access expression using
a `.`. For example, `x->y` and `x->(y)` are interpreted as `(*x).y` and
`(*x).(y)` respectively. More details on this syntax and semantics can be found
in the [pointers](/docs/design/values.md#pointers) design. The rest of this
document describes the semantics using `.` alone for simplicity.

A member access expression (simple, compound, or `impl`) performs some subset of
these three steps in order:

1.  [Name lookup](#name-lookup)
2.  [`impl` lookup](#impl-lookup)
3.  [Instance binding](#instance-binding)

Each step is performed at most once per member access expression. Which steps
are performed depends on the form of the member access expression:

| Member access expression         | Name lookup | `impl` lookup | Instance binding |
| -------------------------------- | ----------- | ------------- | ---------------- |
| Simple member access `a.b`       | ✅ Always   | Sometimes     | Sometimes        |
| Compound member access `a.(m)`   | ❌ Never    | Sometimes     | ✅ Always        |
| `impl` member access `a.impl(m)` | ❌ Never    | ✅ Always     | ❌ Never         |

### Simple member access

For a simple member access`a.b` where `b` is an integer literal, then `a` must
be a tuple, and `a.b` is resolved using [tuple indexing](#tuple-indexing).
Otherwise, `a.b` depends on what kind of entity `a` is:

-   If `a` is a
    [namespace or package](/docs/design/code_and_name_organization/README.md),
    only name lookup for `b` is performed.
-   If `a` names a [non-type facet](/docs/design/generics/terminology.md#facet),
    then `b` is [looked up](#name-lookup) in the type of `a` (which by
    definition is a facet type such as an interface). If the lookup finds an
    associated entity, then [`impl` lookup](#impl-lookup) is performed using `a
    as type` as the `Self` type. This lookup commonly can be satisfied by the
    facet `a` directly.
-   If `a` names a
    [facet type](/docs/design/generics/terminology.md#facet-type), then `a.b`
    performs [name lookup](#name-lookup) for `b` in `a`. No `impl` lookup is
    performed.
-   If `a` names another type (including any [class](/docs/design/classes.md)),
    then `a.b` performs [name lookup](#name-lookup) for `b` in `a`. If the
    result of lookup is an associated entity, then [`impl` lookup](#impl-lookup)
    is performed using `a` as the `Self` type.
-   Otherwise, `a.b` is performed in two steps:
    -   `m` is set to the result of evaluating
        [`typeof(a).b`](type_operators.md#typeof)
        -   `typeof(a)` will always be a facet type or other type, resolved
            using one of the above rules.
        -   As a result of the rewrite, `b` will be looked up in `typeof(a)`,
            and possibly [`impl` lookup](#impl-lookup) will be performed, with
            no further rewrites
    -   [Instance binding](#instance-binding) is performed to bind `a` to `m`.
        This is done unconditionally. No other simple member access cases
        perform instance binding.

If it is not known which case `a` is in, as can happen in generic code,
`a.b` is ill-formed.

### Compound member access

Compound member access `a.(m)` performs up to two steps:

-   If `m` is an associated entity, perform [`impl` lookup](#impl-lookup) with
    the `Self` type set to the type of `a`. This must succeed and be valid.
-   [Instance binding](#instance-binding) is performed.

### `impl` member access

To get `impl` lookup without instance binding, for example to to get the method
`F` in the implementation of the interface `I` for the class `C`, write
`C.impl(I.F)`. This always performs [`impl` lookup](#impl-lookup) with the
`Self` type equal to `C` and nothing else. It is invalid unless `C` is known to
implement `I` (this check is delayed until the expression is no longer template
dependent). As a result, the left argument must always be a type or facet.

### Restrictions

If the first operand is a package or namespace name, only name lookup is
supported, and so the expression must be a simple member access expression.

If the first operand is a type, extended type, or facet, it must be a
compile-time constant. This disallows member access into a type except during
compile-time, see leads issue
[#1293](https://github.com/carbon-language/carbon-lang/issues/1293).

## Name lookup

Only simple member access performs name lookup. Name lookup is always performed
on a namespace, package, type, or extended type.

### Package and namespace members

The member name must be a _word_ that names a member of that package or
namespace, and the result is the package or namespace member with that name.

An expression that names a package or namespace can only be used as the first
operand of a member access or as the target of an `alias` declaration.

```carbon
namespace MyNamespace;
fn MyNamespace.MyFunction() {}

// ✅ OK, can alias a namespace.
alias MyNS = MyNamespace;
fn CallMyFunction() { MyNS.MyFunction(); }

// ❌ Error: a namespace is not a value.
let MyNS2: auto = MyNamespace;

fn CallMyFunction2() {
  // ❌ Error: cannot perform compound member access into a namespace.
  MyNamespace.(MyNamespace.MyFunction)();
}
```

The first operand may also be the keyword `package`, as in `package.Foo`, to
name the `Foo` member of the current package. This can be used to disambiguate
between different `Foo` definitions, as in:

```carbon
// This defines `package.Foo`
class Foo {}
class Bar {
  // This defines `Bar.Foo`, or equivalently `package.Bar.Foo`.
  class Foo {}
  fn F() {
    // ✅ OK, `x` has type `Foo` from the outer scope.
    var x: package.Foo = {};

    // ❌ Error: ambiguous;
    // `Foo` could mean `package.Foo` or `Bar.Foo`.
    var y: Foo = {};
  }
}
```

### Types and extended types

Like the previous case, types (including
[facet types](/docs/design/generics/terminology.md#facet-type)) have member
names, and lookup searches those names. For example:

-   `i32.Least` finds the member constant `Least` of the type `i32`.
-   `Add.Op` finds the member function `Op` of the interface `Add`. Because a
    facet type is a type, this is a special case of the previous bullet.

A search for a name within an extended type searches for the name in its
[type component](/docs/design/values.md#extended-types). Note that this means
that the extended type of an expression never affects simple member access into
that expression, except through its type component.

### `extend`

An `extend` declaration declares that the enclosing scope extends the scope
associated with any target entity named in the declaration. That is to say name
lookups into the enclosing scope will also look into the scopes which are
nominated by the `extend` declaration. The `extend` declaration requires that
the target scopes named in an `extend` declaration are complete, or if the
target is a generic parameter, requires the type of the parameter to be
complete.

To `extend` an entity `Y` with another `Z` means that name lookups into `Y` will
also look into `Z`. Immediately after the `extend` operation, members of `Z`
should also be found when doing name lookup into `Y`, both from outside and from
inside the definition of `Y`. In order to be able to perform lookups into `Z`,
we require that `extend` operations only target scopes that are complete.

This requirement functions recursively. Given an interface `B` that extends
another interface `A`: By naming `A` in an extend declaration, we require `A` is
complete. This provides that its entire definition is known, and thus its
`extend` relationship to `B`. The `extend` relationship there also provides that
`B` is complete.

If the target scope of an `extend` declaration is a generic parameter, its type
must be complete where the `extend` declaration is written, as name lookups into
the extended scope will look into the type of the generic parameter.

```carbon
interface I {
  fn F();
}

class C(T: I) {
  extend base: T;
  // `F` names `T.F` here, found in `I`.
  fn G() { F(); }
}
```

As any generic parameter in the enclosing scope is replaced by a more specific
value, extended scopes that depend on a generic parameter must remain complete.
This includes forming a specific for the extended scope involving the parameter
in order to surface any monomorphization errors in the resulting specific.

In the next example, the `extend` declaration in interface `A(N)` names a
symbolic facet type which can produce monomorphization errors when a negative
value is provided for `N`. When a more specific value for the target `B(N)` is
provided, we require the specific value to be complete as well by forming the
specific. A diagnostic error would be produced while checking `C(-1)` for
completeness, as it requires `A(-1)` to be complete, which requires
`B(array(i32, -1))` to be complete, and that contains an invalid type.

```carbon
interface B(T: type) {}

interface A(N: i32) {
  // Requires `B(N)` to be complete.
  extend require impls B(array(i32, N)) {}
}

class C(N: i32) {
  // Requires `A(N)` to be complete, which requires `B(N)` to be complete.
  extend impl as A(N);
}

fn F() {
  // Requires `C(-1)` to be complete, which requires `A(-1)` to be complete, which requires `B(array(i32, -1))` to be complete.
  var c: C(-1);
}
```

These rules prohibit an `extend` declaration from naming its enclosing scope,
since by being part of the definition of that scope, it is implied that the
enclosing scope is not complete. This seems reasonable as all names available
inside the enclosing interface or named constraint are already available or
would conflict with the ones that are.

> **Alternative considered:**
> [Not requiring the target scope to be complete immediately](/proposals/p006395-type-completeness-in-extend.md#alternatives-considered).

### Values

If the first operand `a` is not a type, extended type, facet, package, or
namespace, it does not have member names, and `a.b` looks up `b` in `typeof(a)`.

```carbon
interface Printable {
  fn Print(self);
}

impl i32 as Printable;

class Point {
  var x: i32;
  var y: i32;
  // Extending impl injects the name `Print` into
  // class `Point`.
  extend impl as Printable;
}

fn PrintPointTwice() {
  var p: Point = {.x = 0, .y = 0};

  // ✅ OK, `x` found in type of `p`, namely `Point`.
  p.x = 1;
  // ✅ OK, `y` found in the type `Point`.
  p.(Point.y) = 1;

  // ✅ OK, `Print` found in type of `p`, namely `Point`.
  p.Print();
  // ✅ OK, `Print` found in the type `Printable`, and
  // `Printable.Print` found in the type of `p`.
  p.(Printable.Print)();
}
```

#### Facet binding

As a specific case of the previous rule for values, if `a` names a non-type
facet, such as `Avatar as Cowboy`, then `b` is looked up in the type of `a`,
which in this case would be `Cowboy`. This can be used to disambiguate when a
type implements multiple interfaces with the same member name, as in this
example:

```carbon
interface Cowboy {
  fn Draw(self);
}

interface Renderable {
  fn Draw(self);
}

class Avatar {
  extend impl Avatar as Cowboy;
  extend impl Avatar as Renderable;
}
```

Simple member access `(Avatar as Cowboy).Draw` finds the `Cowboy.Draw`
implementation for `Avatar`, ignoring `Renderable.Draw`.

This also means that a search for members of a symbolic facet binding `T: C`
treats the facet binding as an
[archetype](/docs/design/generics/terminology.md#archetype), and finds members
of the facet `T` of facet type `C`.

For example:

```carbon
interface Printable {
  fn Print(self);
}

fn GenericPrint[T: Printable](a: T) {
  // ✅ OK, type of `a` is the facet binding `T`;
  // `Print` found in the facet `T as Printable`.
  a.Print();
}
```

**Note:** If lookup is performed into a type that involves a template binding,
the lookup will be performed both in the context of the template definition and
in the context of the template instantiation, as described in
[the "compile-time bindings" section](#compile-time-bindings). The results of
these lookups are [combined](#lookup-ambiguity).

#### Compile-time bindings

If the value or type of the first operand depends on a checked or template
generic parameter, or in fact any
[compile-time binding](/docs/design/generics/terminology.md#bindings), the
lookup is performed from a context where the value of that binding is unknown.
Evaluation of an expression involving the binding may still succeed, but will
result in a symbolic constant involving that binding.

```carbon
class GenericWrapper(T: type) {
  var field: T;
}
fn F[T: type](x: GenericWrapper(T)) -> T {
  // ✅ OK, finds `GenericWrapper(T).field`.
  return x.field;
}

interface Renderable {
  fn Draw(self);
}
fn DrawChecked[T: Renderable](c: T) {
  // `Draw` resolves to `(T as Renderable).Draw` or
  // `T.impl(Renderable.Draw)`.
  c.Draw();
}

class Cowboy { fn Draw(self); }
impl Cowboy as Renderable { fn Draw(self); }

fn CallsDrawChecked(c: Cowboy) {
  // ✅ Calls member of `impl Cowboy as Renderable`.
  DrawChecked(c);
  // In contrast to this which calls member of `Cowboy`:
  c.Draw();
}
```

If the value or type depends on any template bindings, the lookup is redone from
a context where the values of those bindings are known, but where the values of
any checked bindings are still unknown. The lookup results from these two
contexts are [combined](#lookup-ambiguity).

```carbon
fn DrawTemplate[template T: type](c: T) {
  // `Draw` not found in `type`, looked up in the
  // actual deduced value of `T`.
  c.Draw();
}

fn CallsDrawTemplate(c: Cowboy) {
  // ✅ Calls member of `Cowboy`:
  DrawTemplate(c);
  // Same behavior as:
  c.Draw();
}
```

Since we have decided to forbid specialization of class templates, see
[proposal #2200: Template generics](https://github.com/carbon-language/carbon-lang/pull/2200),
the compiler can assume the body of a templated class will be the same for all
argument values:

```carbon
class TemplateWrapper(template T: type) {
  var field: T;
}
fn G[template T: type](x: TemplateWrapper(T)) -> T {
  // ✅ Allowed, finds `TemplateWrapper(T).field`.
  return x.field;
}
```

In addition, the lookup will be performed again when `T` is known. This allows
cases where the lookup only succeeds for specific values of `T`:

```carbon
class HasField {
  var field: i32;
}
class DerivingWrapper(template T: type) {
  extend base: T;
}
fn H[template T: type](x: DerivingWrapper(T)) -> i32 {
  // ✅ Allowed, but no name `field` found in template
  // definition of `DerivingWrapper`.
  return x.field;
}
fn CallH(a: DerivingWrapper(HasField),
         b: DerivingWrapper(i32)) {
  // ✅ Member `field` in base class found in instantiation.
  var x: i32 = H(a);
  // ❌ Error, no member `field` in type of `b`.
  var y: i32 = H(b);
}
```

**Note:** All lookups are done from a context where the values of any checked
bindings that are in scope are unknown. Unlike for a template binding, the
actual value of a checked binding never affects the result of name lookup.

### Lookup ambiguity

Multiple lookups can be performed when resolving a member access expression with
a [template binding](#compile-time-bindings). We resolve this the same way as
when looking in multiple interfaces that are
[combined with `&`](/docs/design/generics/details.md#combining-interfaces-by-anding-facet-types):

-   If more than one distinct member is found, after performing
    [`impl` lookup](#impl-lookup) if necessary, the lookup is ambiguous, and the
    program is invalid.
-   If no members are found, the program is invalid.
-   Otherwise, the result of combining the lookup results is the unique member
    that was found.

```carbon
interface Renderable {
  fn Draw(self);
}

fn DrawTemplate2[template T: Renderable](c: T) {
  // Member lookup finds `(T as Renderable).Draw` and the
  // `Draw` member of the actual deduced value of `T`, if any.
  c.Draw();
}

class Cowboy { fn Draw(self); }
impl Cowboy as Renderable { fn Draw(self); }

class Pig { }
impl Pig as Renderable {
  fn Draw(self);
}

class RoundWidget {
  impl as Renderable {
    fn Draw(self);
  }
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  fn Draw(self) {}
  impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn FlyTemplate[template T: type](c: T) {
  c.Fly();
}

fn Draw(c: Cowboy, p: Pig, r: RoundWidget, s: SquareWidget) {
  // ❌ Error: ambiguous. `Cowboy.Draw` and
  // `(Cowboy as Renderable).Draw` are different.
  DrawTemplate2(c);

  // ✅ OK, lookup in type `Pig` finds nothing, so uses
  // lookup in facet type `Pig as Renderable`.
  DrawTemplate2(p);

  // ✅ OK, lookup in type `RoundWidget` and lookup in facet
  // type `RoundWidget as Renderable` find the same entity.
  DrawTemplate2(r);

  // ✅ OK, lookup in type `SquareWidget` and lookup in facet
  // type `SquareWidget as Renderable` find the same entity.
  DrawTemplate2(s);

  // ❌ Error: `Fly` method not found in `Pig` or
  // `Pig as type`.
  FlyTemplate(p);
}
```

## Tuple indexing

Tuple types have member names that are *integer-literal*s, not *word*s.

Each positional element of a tuple is considered to have a name that is the
corresponding decimal integer: `0`, `1`, and so on. The spelling of the
_integer-literal_ is required to exactly match one of those names, and the
result of name lookup is an instance member that refers to the
corresponding element of the tuple.

```carbon
// ✅ `a == 42`.
let a: i32 = (41, 42, 43).1;
// ❌ Error: no tuple element named `0x1`.
let b: i32 = (1, 2, 3).0x1;
// ❌ Error: no tuple element named `2`.
let c: i32 = (1, 2).2;

var t: (i32, i32, i32) = (1, 2, 3);
let p: (i32, i32, i32)* = &t;
// ✅ `m == 3`.
let m: i32 = p->2;
```

In a compound member access whose second operand is of integer or integer
literal type, the first operand is required to be of tuple type or to extend a
tuple type, otherwise member access fails. The second operand is required to
be a non-negative template constant that is less than the number of tuple
elements, and the result is an instance member that refers to the corresponding
positional element of the tuple.

```carbon
// ✅ `d == 43`.
let d: i32 = (41, 42, 43).(1 + 1);
// ✅ `e == 2`.
let template e: i32 = (1, 2, 3).(0x1);
// ❌ Error: no tuple element with index 4.
let f: i32 = (1, 2).(2 * 2);

// ✅ `n == 3`.
let n: i32 = p->(e);
```

For now, we don't have the extended type machinery to get the compile-time
value of arguments, which we would need to define accessing integer members
using the [customized binding machinery](#member-binding-interfaces). The
intent is that `x.1` means the same thing as `x.(1)` for any `x` whose type
defines what that means.

## `impl` lookup

`impl` lookup maps an associated entity of an interface to the corresponding
member of the relevant `impl`. It is performed when member access names an
associated entity of an interface (except when the member was found by a search
of a facet type scope in a simple member access expression), or when
[calling an associated function of an interface](/docs/design/generics/details.md#associated-functions).

The `impl` query consists of a `Self` type and the interface that the associated
entity is a member of.

### The `Self` type

In this example,

```carbon
interface I {
  fn F();
  fn M(self);
}

class C {
  extend impl as I {
    fn F();   // C.impl(I.F)
    fn M(self);
  }
}

fn G[T: I](x: T, y: C) { ... }
```

using the rules from [the overview section](#overview), a member access in the
body of the function `G` would perform `impl` lookup with a `Self` type as
follows:

-   For simple member access into a non-type facet `T`, for example `T.F` or
    `T.M`, the `Self` type is `T as type`.
-   For simple member access into a class `C`, for example `C.F` or `C.M`, the
    `Self` type is `C`.
-   Simple member access into `x` or `y` will use `typeof(x)` and `typeof(y)`
    respectively, to get `T as type` or `C` as the `Self` types.
-   Compound member access `x.(I.F)` or `y.(I.F)` will use `typeof(x)` and
    `typeof(y)` respectively as in the previous case.
-   Compound member access into types, as in `C.(I.F)` or `I.(I.F)`, will use
    `typeof(C)` or `typeof(I)` as the `Self` type, which in both cases is
    `type`. This will generally fail, since `type` doesn't implement `I`.
-   `impl` member access `C.impl(I.F)` will use the first operand `C` as the
    `Self` type.
-   Associated function call `I.M(y)` will deduce the `Self` type from the
    supplied arguments, in this case `Self` is deduced to be `typeof(y)` or `C`.

### Query lookup

Once the `Self` type `T` and interface `I` is determined, the appropriate `impl
T as I` implementation is located. The program is invalid if no such `impl`
exists. When `T` or `I` depends on a checked binding, a suitable constraint must
be specified to ensure that such an `impl` will exist. When `T` or `I` depends
on a template binding, this check is deferred until the value for the template
binding is known.

Once the `impl` is located, the associated entity is replaced with the
corresponding member of the `impl`. Using the definitions from the last example,
`impl` lookup for `y.F` looks for and finds the implementation `C as I` and uses
its `F` member, which is `C.impl(I.F)`. For
[`impl` member access expressions](#impl-member-access), this is the result. For
[compound member access](#compound-member-access), the result will be the second
argument to [instance binding](#instance-binding). For
[simple member access](#simple-member-access), the result is either used
directly or passed to instance binding, depending on which case it is.

Further examples:

```carbon
interface Addable {
  // #1
  fn Add(self, other: Self) -> Self;
  // #2
  default fn Sum[Seq: Iterable where .ValueType = Self](seq: Seq) -> Self {
    // ...
  }
  alias AliasForSum = Sum;
}

class Integer {
  extend impl as Addable {
    // #3
    fn Add(self, other: Self) -> Self;
    // #4, generated from default implementation for #2.
    // fn Sum[...](...);
  }

  alias AliasForAdd = Addable.Add;
}
```

-   For `Integer.Sum`, name lookup resolves the name `Sum` to \#2, which
    is an associated entity. `impl` lookup then locates the
    `impl Integer as Addable`, and determines that the member access refers to
    \#4.
-   For `i.Add(j)` where `i: Integer`, `typeof(i).Add` resolves the name `Add`
    to \#1 and performs `impl` lookup to locate `impl Integer as Addable`,
    determining that `typeof(i).Add` refers to \#3. Finally,
    [instance binding](#instance-binding) binds `i` to \#3.
-   `Integer.AliasForAdd` finds \#3, the `Add` member of the facet
    `Integer as Addable`, not \#1, the interface member `Addable.Add`.
-   `i.AliasForAdd`, with `i: Integer`, evaluates
    `typeof(i).AliasForAdd` to find \#3, the `Add`
    member of the facet `Integer as Addable`, and performs
    [instance binding](#instance-binding).
-   `Addable.AliasForSum` finds \#2, the member in the interface `Addable`, and
    does not perform `impl` lookup.

**Note:** When an interface member is added to a class by an alias, `impl`
lookup is not performed as part of handling the alias, but will happen when
naming the interface member as a member of the class.

```carbon
interface Renderable {
  // #5
  fn Draw(self);
}

class RoundWidget {
  impl as Renderable {
    // #6
    fn Draw(self);
  }
  // `Draw` names #5, the member of the `Renderable` interface.
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  // #7
  fn Draw(self) {}
  impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn DrawWidget(r: RoundWidget, s: SquareWidget) {
  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #5, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #6.
  // The outer member access then forms a bound member function that
  // calls #6 with `self == r`, as described in "Instance binding".
  r.(RoundWidget.Draw)();

  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #7.
  // The outer member access then forms a bound member function that
  // calls #7 with `self == s`.
  s.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #7.
  // The outer member access fails because we can't call
  // #7, `Draw(self: SquareWidget)`, on a `RoundWidget` object `r`.
  r.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #5, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #6.
  // The outer member access fails because we can't call
  // #6, `Draw(self: RoundWidget)`, on a `SquareWidget` object `s`.
  s.(RoundWidget.Draw)();
}

base class WidgetBase {
  // ✅ OK, even though `WidgetBase` does not implement `Renderable`.
  alias Draw = Renderable.Draw;

  fn DrawAll[T: Renderable](v: Vector(T)) {
    for (w: T in v) {
      // ✅ OK. Unqualified lookup for `Draw` finds alias `WidgetBase.Draw`
      // to `Renderable.Draw`, which does not perform `impl` lookup yet.
      // Then the compound member access expression performs `impl` lookup
      // into `impl T as Renderable`, since `T` is known to implement
      // `Renderable`. Finally, the member function is bound to `w` as
      // described in "Instance binding".
      w.(Draw)();

      // ❌ Error: `Self.Draw` performs `impl` lookup with `WidgetBase` as
      // `Self`, which fails because `WidgetBase` does not implement
      // `Renderable`.
      w.(Self.Draw)();
    }
  }
}

class TriangleWidget {
  extend base: WidgetBase;
  impl as Renderable;
}
fn DrawTriangle(t: TriangleWidget) {
  // ✅ OK: name `Draw` resolves to `Draw` member of `WidgetBase`, which
  // is `Renderable.Draw`. Then impl lookup with `Self` as
  // `typeof(t) == TriangleWidget` replaces that with `Draw` member of
  // `impl TriangleWidget as Renderable`.
  t.Draw();
}
```

### Members of facets that are associated with different interfaces

Performing `impl` lookup when accessing a member of a non-type facet also
supports members of facets that are associated entities of different interfaces,
as in this example:

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
  // members by way of `x` or `T` works and uses the
  // implementation of `I` by `T`.
  T.I_F();
  x.I_M();
}
```

This works because simple member access in facets still performs `impl` lookup,
and is not restricted to only the members of the facet itself.

### Note on the relationship between simple and compound member access

For simple member access on normal values, `a.b` will often be equivalent to
`a.(typeof(a).b)`, but it is not defined that way to ensure at most one `impl`
lookup occurs. The concern is in the case where name lookup for `b` finds an
associated entity, `typeof(a).b` will perform `impl` lookup, which could
possibly find another associated entity. In that case, the compound member
access `a.(...)` would perform a second `impl` lookup that we do not want.

### `impl` member access example

```carbon
fn AddTwoIntegers(a: Integer, b: Integer) -> Integer {
  // Since `Addable.Add` is an associated entity of `Addable`, `Self`
  // is set to `typeof(a)`, and so uses `Integer as Addable`.
  return a.(Addable.Add)(b);
  //      ^ impl lookup and instance binding here
  // Impl lookup transforms this into #3:
  //   return a.((Integer as Addable).Add)(b);
  // or equivalently:
  //   return a.(Integer.impl(Addable.Add))(b);
  // which no longer requires impl lookup.

  // ❌ By the same logic, in this example, `Self` is set to the
  // type of `Integer`, and so uses `type as Addable`, which
  // isn't implemented.
  return Integer.(Addable.Add)(...);
}

fn SumIntegers(v: Vector(Integer)) -> Integer {
  // `Integer.impl(Addable.Sum)` performs `impl` lookup with `Self`
  // equal to `Integer` (`Integer as Addable`), and no instance binding.
  return Integer.impl(Addable.Sum)(v);
  //         ^ impl lookup but no instance binding here
  // Impl lookup transforms this into #4:
  //   ((Integer as Addable).Sum)(v);
  // which no longer requires impl lookup.

  // ❌ Error: `typeof(Integer) == type` does not implement `Addable`.
  Integer.(Addable.Sum)(v);

  var a: Integer;
  // ❌ Error: `Addable.Sum` is not an instance member, so instance
  // binding to `a` fails.
  a.(Addable.Sum)(v);
}
```

#### Callables for member functions

Using `a.impl(m)` and `a.(m)`, we can produce callables for methods and member
functions from interfaces, with the option of binding or not binding `self` for
associated methods:

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
  // or:
  typeof(c).impl(I.F)();

  // impl lookup of `I.M` for `C` taking a `C` parameter for `self`: `C.impl(I.M)`.
  // This may be called with `c` passed in for `self` using:
  C.impl(I.M)(c);
  // or:
  c.(C.impl(I.M))();

  // impl lookup of `I.M` for `C` where the `self` parameter is bound to `c`:
  // `c.(I.M)`
  c.(I.M)();

  // Equivalent to `c.(I.M)()`:
  I.M(c);
}
```

## Instance binding

_Instance binding_ associates an expression with a particular object or value
instance. For example, this is the value bound to `self` when calling a method
or the member of `self` extracted when accessing a field.

[Simple member access](#simple-member-access) `x.y` performs instance binding
when the name `y` is looked up in the type of `x` instead of `x` itself.
[`impl` member access](#impl-member-access) `x.impl(y)` never performs instance
binding. [Compound member access](#compound-member-access) `x.(y)` always
performs instance binding (after performing [`impl` lookup](#impl-lookup) if `y`
is an associated entity).

When instance binding is performed, it is an error if `y` is already bound to an
instance or is a [non-instance member](#non-instance-members) that does not
implement the member binding interfaces. For example:

```carbon
interface DebugPrint {
  // instance member
  fn Print(self);
}
impl i32 as DebugPrint;
impl type as DebugPrint;

fn Debug() {
  var i: i32 = 1;

  // Prints `1` using `(i32 as DebugPrint).Print` bound to `i`.
  i.(DebugPrint.Print)();

  // Prints `i32` using `(type as DebugPrint).Print` bound to `i32`.
  i32.(DebugPrint.Print)();

  // ❌ This is an error since `i32.(DebugPrint.Print)` is already
  // bound, and may not be bound again to `i`.
  i.(i32.(DebugPrint.Print))();
}
```

To get the `M` member of interface `I` for a type `T`, use `T.impl(I.M)` or
`(T as I).M`, as these do not perform instance binding on `T`, in contrast to
`T.(I.M)`.

Instance binding is performed using the implementation of either
[the `BindToValue` or `BindToRef` member binding interface](#member-binding-interfaces)
by `typeof(a)`. The compiler provides `final` builtin implementations to provide
default instance binding behavior.

For compiler-provided builtin implementations of instance binding, the result of
instance binding depends on what instance member `M` was found:

-   For a field member of a struct type or tuple type, `x` is converted to a
    struct or tuple extended type by
    [extended type decomposition](/docs/design/values.md#extended-type-conversions),
    and the `.f` element of the result of that conversion becomes the result of
    `x.f`. All other elements are
    [discarded](/docs/design/values.md#extended-type-conversions).
-   For a field member in class `C`, `x` is required to be of type `C` or of a
    type derived from `C`. The result is the corresponding subobject within `x`.
    If `x` is an
    [initializing expression](/docs/design/values.md#initializing-expressions),
    then a
    [temporary is materialized](/docs/design/values.md#temporary-materialization)
    for `x`. The result of `x.y` has the same
    [expression category](/docs/design/values.md#expression-categories) as the
    possibly materialized `x`.

    ```carbon
    class Size {
      var width: i32;
      var height: i32;
    }

    var dims: Size = {.width = 1, .height = 2};
    // `dims.width` denotes the field `width` of the object `dims`.
    Print(dims.width);
    // `dims` is a reference expression, so `dims.height` is a
    // reference expression.
    dims.height = 3;

    fn GetSize() -> Size;
    // `GetSize()` returns an initializing expression, which is
    // materialized as a temporary on member access, so
    // `GetSize().width` is an ephemeral reference expression.
    Print(GetSize().width);
    ```

-   For a method, the result is a _bound method_, which is a value `F` such that
    a function call `F(args)` behaves the same as a call to `M(args)` with the
    `self` parameter initialized by `x`.

    ```carbon
    class Blob {
      fn Mutate(ref self, n: i32);
    }
    fn F(p: Blob*) {
      // ✅ OK, forms bound method `((*p).Mutate)` and calls it.
      // This calls `Blob.Mutate` with `self` initialized by `*p`
      // and `n` initialized by `5`.
      (*p).Mutate(5);

      // ✅ OK, same as above.
      let bound_m: auto = (*p).Mutate;
      bound_m(5);
    }
    ```

### Member binding interfaces

Instance binding in `x.(y)` or `x.y` is defined in terms of rewrites to invoking
an interface method, like other operators. There are two interfaces used,
depending on whether `x` is a value expression or a reference expression:

```carbon
// This determines the type of the result of member binding. It is
// a separate interface shared by `BindToValue` and `BindToRef` to
// ensure they produce the same result type. We don't want the
// type of an expression to depend on the expression category
// of the arguments.
interface Bind(T: type) {
  let Result: type;
}

// For a value expression `x` with type `T` and an expression
// `y` of type `U`, `x.(y)` is `y.((U as BindToValue(T)).Op)(x)`
interface BindToValue(T: type) {
  extend require impls Bind(T);
  fn Op(self, x: T) -> Result;
}

// For a reference expression `x` with type `T` and an expression
// `y` of type `U`, `x.(y)` is
// `y.((U as BindToRef(T)).Op)(ref x)`
interface BindToRef(T: type) {
  extend require impls Bind(T);
  fn Op(self, ref p: T) -> ref Result;
}
```

> **QUESTION:** Should these interfaces use `extend final impl as Bind(T)`
> instead of `extend require impls Bind(T)`, per
> [proposal #5337](/proposals/p005337-interface-extension-and-final-impl-update.md)?

To use instance members of a class, we go through this step of _member
binding_. Consider a class `C`:

```carbon
class C {
  fn F(self) -> i32 { return self.x + 5; }
  fn Static() -> i32 { return 2; }
  var x: i32;
}
```

Each member of `C` with a distinct name has a corresponding type (like
`__TypeOf_C_F`) and value of that type (like `__C_F`). For each instance method,
there is another type that [adapts](/docs/design/classes.md#adapters) `C` and
represents the type of binding that method with either a `C` value or variable:

```carbon
class __TypeOf_C_F {}
let __C_F: __TypeOf_C_F = {};
class __Binding_C_F {
  adapt C;
}
```

This is the type that results from instance binding an instance of `C` with
`C.F`. It defines the bound method value and bound method type of
[proposal #2875](/proposals/p002875-functions-function-types-and-function-calls.md#bound-methods).
For example,

```carbon
let v: C = {.x = 3};
Assert(v.F() == 8);
var r: C = {.x = 4};
Assert(r.F() == 9);
```

is interpreted as:

```carbon
let v: C = {.x = 3};
Assert((v as __Binding_C_F).(Call(()).Op)() == 8);
var r: C = {.x = 4};
Assert((r as __Binding_C_F).(Call(()).Op)() == 9);
```

How does this arise?

1.  First the simple member access evaluates `typeof(a).F`, which is `C.F` (`__C_F`
    with type `__TypeOf_C_F`).
2.  It then looks at the expression to the left of the `.`:
    -   If it is a reference expression, the "member binding to reference"
        (`BindToRef`) operator is applied.
    -   If it is a value expression, the "member binding to value"
        (`BindToValue`) operator is applied.
3.  The result of the member binding has a type that implements the call
    interface.

These member binding operations are implemented by the compiler as `final`
builtin implementations for the types of the instance class members:

```carbon
final impl __TypeOf_C_F as Bind(C) {
  where Result = __Binding_C_F;
}

final impl __TypeOf_C_F as BindToValue(C) {
  fn Op(unused self, x: C) -> __Binding_C_F {
    return x as __Binding_C_F;
  }
}

// Note that the return type has to match, since
// it is an associated type in the `Bind(C)` interface
// that both `BindToValue(C)` and `BindToRef(C)` extend.
final impl __TypeOf_C_F as BindToRef(C) {
  fn Op(unused self, p: ref C) -> ref __Binding_C_F {
    return p as __Binding_C_F;
  }
}
```

Those implementations are how we get from `__C_F` with type `__TypeOf_C_F` to
`v as __Binding_C_F` or `r as __Binding_C_F`, conceptually following these
steps:

```carbon
// `v` is a value and so uses `BindToValue`
v.F() == v.(C.F)()
      == v.(__C_F)()
      == __C_F.((__TypeOf_C_F as BindToValue(C)).Op)(v)()
      == (v as __Binding_C_F)()

// `r` is a reference expression and so uses `BindToRef`
r.F() == r.(C.F)()
      == r.(__C_F)()
      == __C_F.((__TypeOf_C_F as BindToRef(C)).Op)(ref r)()
      == (r as __Binding_C_F)()
```

However, to avoid recursive application of these same rules, we need to avoid
expressing this in terms of evaluating `__C_F.(`...`)`. Instead the third step
uses an intrinsic compiler primitive, as in:

```carbon
// `v` is a value and so uses `BindToValue`
v.F() == v.(C.F)()
      == v.(__C_F)()
      == inlined_method_call_compiler_intrinsic(
              <function body (__TypeOf_C_F as BindToValue(C)).Op overload 0>,
              __C_F, (v))()
      == (v as __Binding_C_F)()

// `r` is a reference expression and so uses `BindToRef`
r.F() == r.(C.F)()
      == r.(__C_F)()
      == inlined_method_call_compiler_intrinsic(
              <function body (__TypeOf_C_F as BindToRef(C)).Op overload 0>,
              __C_F, (ref r))()
      == (r as __Binding_C_F)()
```

At this point we have resolved the member binding, and are left with an
expression of type `__Binding_C_F` followed by `()`. In the first case, that
expression is a value expression. In the second case, it is a reference
expression.

The last ingredient is the implementation of the call interfaces for these bound
types:

```carbon
// Member binding with `C.F` produces something with type
// `__Binding_C_F` whether it is a value or reference
// expression. Since `C.F` takes `self: Self` it can be
// used in both cases.
impl __Binding_C_F as Call(()) where .Result = i32 {
  fn Op(self) -> i32 {
    // Calls `(self as C).(C.F)()`, but without triggering
    // member binding again.
    return inlined_method_call_compiler_intrinsic(
        <function body C.F overload 0>, self as C, ());
  }
}

// `C.Static` is a non-instance member function, so `__TypeOf_C_Static`
// implements the call interface directly (allowing `C.Static()` to work),
// and does not implement `BindToValue` or `BindToRef`.
impl __TypeOf_C_Static as Call(()) where .Result = i32 {
  fn Op(unused self) -> i32 {
    return inlined_call_compiler_intrinsic(
               <function body C.Static overload 0>, ());
  }
}
```

Going back to `v.F()` and `r.F()`, after member binding the next step is to
resolve the call. As described in
[proposal #2875](https://github.com/carbon-language/carbon-lang/pull/2875), this
call is rewritten to an invocation of the `Op` method of the `Call(())`
interface, using the implementations just defined. Note:

-   Passing `r as __Binding_C_F` to the `self` parameter of `Call(()).Op`
    converts the reference expression to a value.
-   The `Call` interface is special. We don't rewrite calls to `Call(__).Op` to
    avoid infinite recursion.

```carbon
v.F() == (v as __Binding_C_F)()
      == (v as __Binding_C_F).((__Binding_C_F as Call(())).Op)()
      == inlined_method_call_compiler_intrinsic(
            <function body (__Binding_C_F as Call(())).Op overload 0>,
            v as __Binding_C_F, ());
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>,
             (v as __Binding_C_F) as C, ())
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>, v, ())

r.F() == (r as __Binding_C_F)()
      == (r as __Binding_C_F).((__Binding_C_F as Call(())).Op)()
      == inlined_method_call_compiler_intrinsic(
            <function body (__Binding_C_F as Call(())).Op overload 0>,
            r as __Binding_C_F <as value expression>, ());
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>,
             r <as value expression>, ())
```

> **Note:** This rewrite results in compiler intrinsics for calling. This is to
> show that no more rewrites are applied.

### Inheritance and other implicit conversions

Now consider methods of a base class:

```carbon
base class B {
  fn F(self);
  virtual fn V(self);
}

class D {
  extend base: B;
  impl fn V(self);
}

var d: D = {};
d.(B.F)();
d.(B.V)();
```

To allow this to work, we need the implementation of the member binding
interfaces to allow implicit conversions:

```carbon
final impl [T: ImplicitAs(B)] __TypeOf_B_F as Bind(T) {
  where Result = __Binding_B_F;
}


final impl [T: ImplicitAs(B)] __TypeOf_B_F as BindToValue(T) {
  fn Op(self, x: T) -> __Binding_B_F {
    return (x as B) as __Binding_B_F;
  }
}

final impl [T: type where .Self* impls ImplicitAs(B*)]
    __TypeOf_B_F as BindToRef(T) {
  fn Op(self, ref p: T) -> ref __Binding_B_F {
    return *((&p as B*) as __Binding_B_F*);
  }
}
```

This matches the expected semantics of method calls, even for methods of final
classes.

Note that the implementation of the member binding interfaces is where the
`Self` type of a method is used. If that type is different from the class it is
being defined in, as considered in
[#1345](https://github.com/carbon-language/carbon-lang/issues/1345), that will
be reflected in the member binding implementations.

```carbon
class C {
  // Note: not `self: Self` or `self: C`!
  fn G(self: Different);
}

let c: C = {};
// `c.G()` is only allowed if there is an implicit
// conversion from `C` to `Different`.

let d: Different = {};
// Allowed:
d.(C.G)();
```

results in an implementation using `Different` instead of `C`:

```carbon
final impl [T: ImplicitAs(Different)] __TypeOf_C_G as Bind(T) {
  where Result = __Binding_C_G;
}

// `C.G` will only member bind to values that can implicitly convert
// to type `Different`.
final impl [T: ImplicitAs(Different)] __TypeOf_C_G as BindToValue(T);
```

### Data fields

The same `BindToValue` and `BindToRef` operations are also used to define access
to the data fields in an object.

For example, given a class with a data member `m` with type `i32`:

```carbon
class C {
  var m: i32;
}
```

we want the usual operations to work, with `x.m` equivalent to `x.(C.m)`:

```carbon
let v: C = {.m = 4};
var x: C = {.m = 3};
x.m += 5;
Assert(x.(C.m) == v.m + v.(C.m));
```

To accomplish this we will, as before, associate an empty (stateless or
zero-sized) type with the `m` member of `C`, that just exists to support the
member binding operation. However, this time the result type of member binding
is simply `i32`, the type of the variable, instead of a new, dedicated type.

```carbon
class __TypeOf_C_m {}
let __C_m: __TypeOf_C_m = {};

final impl __TypeOf_C_m as Bind(C) {
  where Result = i32;
}

final impl __TypeOf_C_m as BindToValue(C) {
  fn Op(self, x: C) -> i32 {
    // Effectively performs `x.m`, but without triggering member binding again.
    return value_compiler_intrinsic(x, __OffsetOf_C_m, i32);
  }
}

final impl __TypeOf_C_m as BindToRef(C) {
  fn Op(self, ref p: C) -> ref i32 {
    // Effectively performs `p.m`, but without triggering member binding again,
    // by doing something like `*(((&p as byte*) + __OffsetOf_C_m) as i32*)`
    return *offset_compiler_intrinsic(&p, __OffsetOf_C_m, i32);
  }
}
```

These definitions give us the desired semantics:

```carbon
// For value `v` with type `T` and `y` of type `U`,
// `v.(y)` is `y.((U as BindToValue(T)).Op)(v)`
v.m == v.(C.m)
    == v.(__C_m)
    == v.(__C_m as (__TypeOf_C_m as BindToValue(C)))
    == __C_m.((__TypeOf_C_m as BindToValue(C)).Op)(v)
    == value_compiler_intrinsic(v, __OffsetOf_C_m, i32)

// For reference expression `var x: T` and `y` of type `U`,
// `x.(y)` is `y.(U as BindToRef(T)).Op(ref x)`
x.m == x.(C.m)
    == x.(__C_m)
    == __C_m.((__TypeOf_C_m as BindToRef(C)).Op)(ref x)
    == *offset_compiler_intrinsic(&x, __OffsetOf_C_m, i32)
// Note that this requires `x` to be a reference expression,
// so `&x` is valid, and produces a reference expression,
// since it is the result of dereferencing a pointer.
```

The fields of [tuple types](/docs/design/tuples.md) and
[struct types](/docs/design/classes.md#struct-types) operate the same way.

```carbon
let t_let: (i32, i32) = (3, 6);
Assert(t_let.(((i32, i32) as type).0) == 3);

var t_var: (i32, i32) = (4, 8);
Assert(t_var.(((i32, i32) as type).1) == 8);
t_var.(((i32, i32) as type).1) = 9;
Assert(t_var.1 == 9);

let s_let: {.x: i32, .y: i32} = {.x = 5, .y = 10};
Assert(s_let.({.x: i32, .y: i32}.x) == 5);

var s_var: {.x: i32, .y: i32} = {.x = 6, .y = 12};
Assert(s_var.({.x: i32, .y: i32}.y) == 12);
s_var.({.x: i32, .y: i32}.y) = 13;
Assert(s_var.y == 13);
```

For example, `{.x: i32, .y: i32}.x` is a value `__Struct_x_i32_y_i32_Field_x`,
analogous to `__C_m`, of a type `__TypeOf_Struct_x_i32_y_i32_Field_x` (that is
zero-sized / has no state), analogous to `__TypeOf_C_m`, that implements the
member binding interfaces for any type that implicitly converts to
`{.x: i32, .y: i32}`.

Note that for tuples, the `as type` is needed since `(i32, i32)` on its own is a
tuple, not a type. In particular `(i32, i32)` is not the type of `t_let` or
`t_var`. `(i32, i32).0` is just `i32`, and isn't the name of the first element
of an `(i32, i32)` tuple.

### Generic type of a class member

Given the above, we can write a constraint on a symbolic parameter to match the
names of an unbound class member. There are two cases: methods and fields.

#### Methods

For value methods, the receiver object may be passed by value. To be able to
call the method, we must include a restriction that the result of `BindToValue`
implements `Call(())`:

```carbon
// `m` can be any method object that implements `Call(())` once bound.
fn CallMethod
    [T: type, M: BindToValue(T) where .Result impls Call(())]
    (x: T, m: M) -> auto {
  // `x.(m)` is rewritten to a call to `BindToValue(T).Op`. The
  // constraint on `M` ensures the result implements `Call(())`.
  return x.(m)();
}
```

This works with any value method. This also works with inheritance and virtual
methods, using
[the support for implicit conversions of self](#inheritance-and-other-implicit-conversions).

```carbon
base class X {
  virtual fn V(self) -> i32 { return 1; }
  fn B(self) -> i32 { return 0; }
}
class Y {
  extend base: X;
  impl fn V(self) -> i32 { return 2; }
}
class Z {
  extend base: X;
  impl fn V(self) -> i32 { return 3; }
}

var (x: X, y: Y, z: Z);

// Respects inheritance
Assert(CallMethod(x, X.B) == 0);
Assert(CallMethod(y, X.B) == 0);
Assert(CallMethod(z, X.B) == 0);

// Respects method overriding
Assert(CallMethod(x, X.V) == 1);
Assert(CallMethod(y, X.V) == 2);
Assert(CallMethod(z, X.V) == 3);
```

#### Fields

Fields can be accessed, given the type of the field:

```carbon
fn GetField
    [T: type, F: BindToValue(T) where .Result = i32]
    (x: T, f: F) -> i32 {
  // `x.(f)` is rewritten to `f.((F as BindToValue(T)).Op)(x)`,
  // and `(F as BindToValue(T)).Op` is a method on `f` with
  // return type `i32` by the constraint on `F`.
  return x.(f);
}

fn SetField
    [T: type, F: BindToRef(T) where .Result = i32]
    (ref x: T, f: F, y: i32) {
  // `x.(f)` which becomes:
  //   `f.((F as BindToRef(T)).Op)(ref x)`.
  // The constraint `F` says the return of
  // `(F as BindToRef(T)).Op` is an `i32` reference
  // which may then be assigned.
  x.(f) = y;
}

class C {
  var m: i32;
  var n: i32;
}
var c: C = {.m = 5, .n = 6};
Assert(GetField(c, C.m) == 5);
Assert(GetField(c, C.n) == 6);
SetField(&c, C.m, 42);
SetField(&c, C.n, 12);
Assert(GetField(c, C.m) == 42);
Assert(GetField(c, C.n) == 12);
```

### C++ pointer-to-member values

[C++ pointer-to-member](https://en.cppreference.com/cpp/language/pointer)
values are usable from Carbon once bound to an instance:

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

In
[the generic type of a class member section](#generic-type-of-a-class-member),
the names of members, such as `X.B`, `X.V`, and `C.n`, refer to zero-sized /
stateless objects where all the offset information is encoded in the type.
However, the definitions of `CallMethod`, `SetField`, and `GetField` do not
depend on that fact and are usable with objects, such as C++
pointers-to-members, that include the offset information in the runtime object
state. Member binding implementations are defined for them so that they may be
used with Carbon's `.(`...`)` and `->(`...`)` operators.

For example, C++ code can call the above Carbon functions:

```cpp
struct C {
  int F() const { return m + 1; }
  int m;
};

int main() {
  // pointer to data member `m` of class C
  int C::* p = &C::m;
  C c = {2};
  assert(c.*p == 2);
  assert(Carbon::GetField(c, p) == 2);
  Carbon::SetField(&c, p, 4);
  assert(c.m == 4);
  // pointer to method `F` of class C
  int (C::*q)() const = &C::F;
  assert(Carbon::CallMethod(&c, q) == 5);
}
```

### C++ operator overloading

C++ does not support customizing the behavior of `x.y`. It does support
customizing the behavior of `operator*` and `operator->` which is frequently
used to support smart pointers and iterators. There is, however, nothing
restricting the implementations of those two operators to be consistent, so that
`(*x).y` and `x->y` are the same.

Carbon instead only has a single interface for customizing dereference,
corresponding to `operator*` not `operator->`. All uses of `x->y` are rewritten
to use `(*x).y` instead. This may cause some friction when porting C++ code
where those operators are not consistent. If the C++ code is just missing the
definition of `operator*` corresponding to an `operator->`, a workaround is just
to define `operator*`.

Other cases of divergence between those operators should be rare, since that is
both surprising to users and for the common case of iterators, violates the C++
requirements. If necessary, we can in the future introduce a specific construct
just for C++ interop that invokes the C++ arrow operator, such as
`CppArrowOperator(x)`, that returns a pointer.

## Non-instance members

Non-instance members of types (including classes and interfaces) do not
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

  class Nested {}

  extend impl as I;
}

fn InvalidInstanceAccess(x: C) {
  // Invalid: non-instance members do not implement binding interfaces:
  // ❌ x.F();
  // ❌ x.G();
  // ❌ x.s = 1;
  // ❌ var n: x.Nested;
}

fn ValidAccess(x: C) {
  // Instead, these should be written:
  C.F();  // ✅
  C.impl(I.F)();  // ✅
  typeof(x).F();  // ✅

  C.G();  // ✅
  typeof(x).G();  // ✅

  C.s = 1;  // ✅
  typeof(x).s = 1;  // ✅

  var n1: C.Nested = {};  // ✅
  var n2: typeof(x).Nested = {};  // ✅
}
```

We require that the caller distinguish whether they are performing instance
binding, which means that changing a method to a non-instance member function
requires updating callers.

### Overloading

Nothing about the second operand is used to decide whether to perform instance
binding or whether to use the value or type of the left operand for `impl`
lookup. The only fact about the right operand that is used is whether it names
an associated entity. This is to support overloading between instance and
non-instance members with the same name, once that is added to the language.
Instead, instance binding is controlled by the calling syntax used.

## Precedence and associativity

Member access expressions associate left-to-right:

```carbon
class Inner {
  fn F(self);
}
class A {
  var B: Inner;
}
interface B {
  fn F(self);
}
impl A as B;

fn Use(a: A) {
  // Calls member `F` of field `a.B`.
  (a.B).F();
  // Calls member `F` of interface `B`, as implemented by type `A`.
  a.(B.F)();
  // Same as `(a.B).F()`.
  a.B.F();
}
```

Member access has [lower precedence](README.md#precedence) than primary
expressions (literals, unqualified names, and expressions in parentheses, as in
[C++](https://cppreference.com/cpp/language/expressions#Primary_expressions)),
and higher precedence than all other expression forms.

```carbon
// ✅ OK, `*` has lower precedence than `.`. Same as `(A.B)*`.
var p: A.B*;
// ✅ OK, `1 + (X.Y)` not `(1 + X).Y`.
var n: i32 = 1 + X.Y;
```

## Alternatives considered

-   [Separate syntax for static versus dynamic access, such as `::` versus `.`](/proposals/p000989-member-access-expressions.md#separate-syntax-for-static-versus-dynamic-access)
-   [Use a different lookup rule for names in templates](/proposals/p000989-member-access-expressions.md#use-a-different-lookup-rule-in-templates)
-   [Meaning of `Type.Interface`](/proposals/p000989-member-access-expressions.md#meaning-of-typeinterface)
-   [Swap the member binding interface parameters](/proposals/p003720-member-binding-operators.md#swap-the-member-binding-interface-parameters)
-   [Member binding to references produces a value that wraps a pointer](/proposals/p003720-member-binding-operators.md#member-binding-to-references-produces-a-value-that-wraps-a-pointer)
-   [Separate interface for compile-time member binding instead of type member binding](/proposals/p003720-member-binding-operators.md#separate-interface-for-compile-time-member-binding-instead-of-type-member-binding)
-   [Non-instance members are idempotent under member binding](/proposals/p003720-member-binding-operators.md#non-instance-members-are-idempotent-under-member-binding)
-   [Separate `Result` types for `BindToValue` and `BindToRef`](/proposals/p003720-member-binding-operators.md#separate-result-types-for-bindtovalue-and-bindtoref)
-   [`BindToValue` is a subtype of `BindToRef`](/proposals/p003720-member-binding-operators.md#bindtovalue-is-a-subtype-of-bindtoref)
-   [Directly rewrite all calls to interface member functions to method call intrinsics](/proposals/p003720-member-binding-operators.md#directly-rewrite-all-calls-to-interface-member-functions-to-method-call-intrinsics)
-   [Different way to distinguish whether instance binding occurs](/proposals/p007697-updates-to-member-access.md#different-way-to-distinguish-whether-instance-binding-occurs)
-   [Non-instance members could implement the binding interfaces](/proposals/p007697-updates-to-member-access.md#non-instance-members-could-implement-the-binding-interfaces)
-   [Other member access operators](/proposals/p007697-updates-to-member-access.md#other-member-access-operators)
-   [Bind interfaces only used for compound member access](/proposals/p007697-updates-to-member-access.md#bind-interfaces-only-used-for-compound-member-access)

## References

-   Proposal
    [#989: member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   [Question for leads: constrained template name lookup](https://github.com/carbon-language/carbon-lang/issues/949)
-   Proposal
    [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
-   Proposal
    [#2550: Simplified package declaration for the `Main` package](https://github.com/carbon-language/carbon-lang/pull/2550)
-   Proposal
    [#3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720)
-   Proposal
    [#5434: `ref` parameters, arguments, returns and `val` returns](https://github.com/carbon-language/carbon-lang/pull/5434)
-   Proposal
    [#6395: Type completeness in extend](https://github.com/carbon-language/carbon-lang/pull/6395)
-   Proposal
    [#7697: Updates to member access](https://github.com/carbon-language/carbon-lang/pull/7697)
