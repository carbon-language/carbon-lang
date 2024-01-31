# Qualified names and member access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Member resolution](#member-resolution)
    -   [Package and namespace members](#package-and-namespace-members)
    -   [Types and facets](#types-and-facets)
    -   [Values](#values)
    -   [Facet binding](#facet-binding)
        -   [Compile-time bindings](#compile-time-bindings)
        -   [Lookup ambiguity](#lookup-ambiguity)
-   [`impl` lookup](#impl-lookup)
    -   [`impl` lookup for simple member access](#impl-lookup-for-simple-member-access)
    -   [`impl` lookup for compound member access](#impl-lookup-for-compound-member-access)
-   [Instance binding](#instance-binding)
-   [Non-instance members](#non-instance-members)
-   [Non-vacuous member access restriction](#non-vacuous-member-access-restriction)
-   [Tuple indexing](#tuple-indexing)
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

A member access expression is either a _simple_ member access expression of the
form:

-   _member-access-expression_ ::= _expression_ `.` _word_
-   _member-access-expression_ ::= _expression_ `->` _word_

or a _compound_ member access of the form:

-   _member-access-expression_ ::= _expression_ `.` `(` _expression_ `)`
-   _member-access-expression_ ::= _expression_ `->` `(` _expression_ `)`

Compound member accesses allow specifying a qualified member name.

For example:

```carbon
namespace Widgets;

interface Widgets.Widget {
  fn Grow[addr self: Self*](factor: f64);
}

class Widgets.Cog {
  var size: i32;
  fn Make(size: i32) -> Self;
  extend impl as Widgets.Widget;
}

fn Widgets.GrowSomeCogs() {
  var cog1: Cog = Cog.Make(1);
  var cog2: Cog = cog1.Make(2);
  var cog_pointer: Cog* = &cog2;
  let cog1_size: i32 = cog1.size;
  cog1.Grow(1.5);
  cog2.(Cog.Grow)(cog1_size as f64);
  cog1.(Widget.Grow)(1.1);
  cog2.(Widgets.Cog.(Widgets.Widget.Grow))(1.9);
  cog_pointer->Grow(0.75);
  cog_pointer->(Widget.Grow)(1.2);
}
```

Pointer member access expressions are those using a `->` instead of a `.` and
their semantics are exactly what would result from first dereferencing the
expression preceding the `->` and then forming a member access expression using
a `.`. For example, a simple pointer member access expression _expression_ `->`
_word_ becomes `(` `*` _expression_ `)` `.` _word_. More details on this syntax
and semantics can be found in the [pointers](/docs/design/values.md#pointers)
design. The rest of this document describes the semantics using `.` alone for
simplicity.

A member access expression is processed using the following steps:

-   First, the word or parenthesized expression to the right of the `.` is
    [resolved](#member-resolution) to a specific member entity, called `M` in
    this document.
-   Then, if necessary, [`impl` lookup](#impl-lookup) is performed to map from a
    member of an interface to a member of the relevant `impl`, potentially
    updating `M`.
-   Then, if necessary, [instance binding](#instance-binding) is performed to
    locate the member subobject corresponding to a field name or to build a
    bound method object, producing the result of the member access expression.
-   If [instance binding is not performed](#non-instance-members), the result is
    `M`.

## Member resolution

The process of _member resolution_ determines which member `M` a member access
expression is referring to.

For a simple member access, the second operand is a word. If the first operand
is a type, facet, package, or namespace, a search for the word is performed in
the first operand. Otherwise, a search for the word is performed in the type of
the first operand. In either case, the search must succeed. In the latter case,
if the result is an instance member, then [instance binding](#instance-binding)
is performed on the first operand.

For a compound member access, the second operand is evaluated as a compile-time
constant to determine the member being accessed. The evaluation is required to
succeed and to result in a member of a type, interface, or non-type facet. If
the result is an instance member, then [instance binding](#instance-binding) is
always performed on the first operand.

### Package and namespace members

If the first operand is a package or namespace name, the expression must be a
simple member access expression. The _word_ must name a member of that package
or namespace, and the result is the package or namespace member with that name.

An expression that names a package or namespace can only be used as the first
operand of a member access or as the target of an `alias` declaration.

```
namespace MyNamespace;
fn MyNamespace.MyFunction() {}

// ✅ OK, can alias a namespace.
alias MyNS = MyNamespace;
fn CallMyFunction() { MyNS.MyFunction(); }

// ❌ Error: a namespace is not a value.
let MyNS2:! auto = MyNamespace;

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

### Types and facets

If the first operand is a type or facet, it must be a compile-time constant.
This disallows member access into a type except during compile-time, see leads
issue [#1293](https://github.com/carbon-language/carbon-lang/issues/1293).

Like the previous case, types (including
[facet types](/docs/design/generics/terminology.md#facet-type)) have member
names, and lookup searches those names. For example:

-   `i32.Least` finds the member constant `Least` of the type `i32`.
-   `Add.Op` finds the member function `Op` of the interface `Add`. Because a
    facet type is a type, this is a special case of the previous bullet.

Unlike the previous case, both simple and compound member access is allowed.

Non-type facets, such as `T as Cowboy`, also have members. Specifically, the
members of the `impl` or `impl`s that form the implementation of `T as Cowboy`.
Being part of the `impl` rather than the interface, no further
[`impl` lookup](#impl-lookup) is needed.

```carbon
interface Cowboy {
  fn Draw[self: Self]();
}

interface Renderable {
  fn Draw[self: Self]();
}

class Avatar {
  extend impl Avatar as Cowboy;
  extend impl Avatar as Renderable;
}
```

Simple member access `(Avatar as Cowboy).Draw` finds the `Cowboy.Draw`
implementation for `Avatar`, ignoring `Renderable.Draw`.

### Values

If the first operand is not a type, package, namespace, or facet it does not
have member names, and a search is performed into the type of the first operand
instead.

```carbon
interface Printable {
  fn Print[self: Self]();
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

### Facet binding

A search for members of a facet binding `T:! C` treats the facet binding as an
[archetype](/docs/design/generics/terminology.md#archetype), and finds members
of the facet `T` of facet type `C`.

For example:

```
interface Printable {
  fn Print[self: Self]();
}

fn GenericPrint[T:! Printable](a: T) {
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
result in a symbolic value involving that binding.

```carbon
class GenericWrapper(T:! type) {
  var field: T;
}
fn F[T:! type](x: GenericWrapper(T)) -> T {
  // ✅ OK, finds `GenericWrapper(T).field`.
  return x.field;
}

interface Renderable {
  fn Draw[self: Self]();
}
fn DrawChecked[T:! Renderable](c: T) {
  // `Draw` resolves to `(T as Renderable).Draw` or
  // `T.(Renderable.Draw)`.
  c.Draw();
}

class Cowboy { fn Draw[self: Self](); }
impl Cowboy as Renderable { fn Draw[self: Self](); }

fn CallsDrawChecked(c: Cowboy) {
  // ✅ Calls member of `impl Cowboy as Renderable`.
  DrawChecked(c);
  // In contrast to this which calls member of `Cowboy`:
  c.Draw();
}
```

If the value or type depends on any template bindings, the lookup is redone from
a context where the values of those bindings are known, but where the values of
any symbolic bindings are still unknown. The lookup results from these two
contexts are [combined](#lookup-ambiguity).

```carbon
fn DrawTemplate[template T:! type](c: T) {
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
class TemplateWrapper(template T:! type) {
  var field: T;
}
fn G[template T:! type](x: TemplateWrapper(T)) -> T {
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
class DerivingWrapper(template T:! type) {
  extend base: T;
}
fn H[template T:! type](x: DerivingWrapper(T)) -> i32 {
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

**Note:** All lookups are done from a context where the values of any symbolic
bindings that are in scope are unknown. Unlike for a template binding, the
actual value of a symbolic binding never affects the result of member
resolution.

#### Lookup ambiguity

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
  fn Draw[self: Self]();
}

fn DrawTemplate2[template T:! Renderable](c: T) {
  // Member lookup finds `(T as Renderable).Draw` and the
  // `Draw` member of the actual deduced value of `T`, if any.
  c.Draw();
}

class Cowboy { fn Draw[self: Self](); }
impl Cowboy as Renderable { fn Draw[self: Self](); }

class Pig { }
impl Pig as Renderable {
  fn Draw[self: Self]();
}

class RoundWidget {
  impl as Renderable {
    fn Draw[self: Self]();
  }
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  fn Draw[self: Self]() {}
  impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn FlyTemplate[template T:! type](c: T) {
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

## `impl` lookup

`impl` lookup maps a member of an interface to the corresponding member of the
relevant `impl`. It is performed when member access names an interface member,
except when the member was found by a search of a facet type scope in a simple
member access expression.

### `impl` lookup for simple member access

For a simple member access `a.b` where `b` names a member of an interface `I`:

-   If the interface member was found by searching a
    non-[facet-type](/docs/design/generics/terminology.md#facet-type) scope `T`,
    for example a class or an adapter, then `impl` lookup is performed for
    `T as I`.
    -   In the case where the member was found in a base class of the class that
        was searched, `T` is the derived class that was searched, not the base
        class in which the name was declared.
    -   More generally, if the member was found in something the type extends,
        such as a facet type or mixin, `T` is the type that was initially
        searched, not what it extended.
-   Otherwise, `impl` lookup is not performed.

The appropriate `impl T as I` implementation is located. The program is invalid
if no such `impl` exists. When `T` or `I` depends on a symbolic binding, a
suitable constraint must be specified to ensure that such an `impl` will exist.
When `T` or `I` depends on a template binding, this check is deferred until the
value for the template binding is known.

`M` is replaced by the member of the `impl` that corresponds to `M`.

[Instance binding](#instance-binding) may also apply if the member is an
instance member.

For example:

```carbon
interface Addable {
  // #1
  fn Add[self: Self](other: Self) -> Self;
  // #2
  default fn Sum[Seq:! Iterable where .ValueType = Self](seq: Seq) -> Self {
    // ...
  }
  alias AliasForSum = Sum;
}

class Integer {
  extend impl as Addable {
    // #3
    fn Add[self: Self](other: Self) -> Self;
    // #4, generated from default implementation for #2.
    // fn Sum[...](...);
  }

  alias AliasForAdd = Addable.Add;
}
```

-   For `Integer.Sum`, member resolution resolves the name `Sum` to \#2, which
    is not an instance member. `impl` lookup then locates the
    `impl Integer as Addable`, and determines that the member access refers to
    \#4.
-   For `i.Add(j)` where `i: Integer`, member resolution resolves the name `Add`
    to \#1, which is an instance member. `impl` lookup then locates the
    `impl Integer as Addable`, and determines that the member access refers to
    \#3. Finally, instance binding will be performed as described later.
-   `Integer.AliasForAdd` finds \#3, the `Add` member of the facet type
    `Integer as Addable`, not \#1, the interface member `Addable.Add`.
-   `i.AliasForAdd`, with `i: Integer`, finds \#3, the `Add` member of the facet
    type `Integer as Addable`, and performs
    [instance binding](#instance-binding) since the member is an instance
    member.
-   `Addable.AliasForSum` finds \#2, the member in the interface `Addable`, and
    does not perform `impl` lookup.

**Note:** When an interface member is added to a class by an alias, `impl`
lookup is not performed as part of handling the alias, but will happen when
naming the interface member as a member of the class.

```carbon
interface Renderable {
  // #5
  fn Draw[self: Self]();
}

class RoundWidget {
  impl as Renderable {
    // #6
    fn Draw[self: Self]();
  }
  // `Draw` names #5, the member of the `Renderable` interface.
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  // #7
  fn Draw[self: Self]() {}
  impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn DrawWidget(r: RoundWidget, s: SquareWidget) {
  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #5, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #6.
  // The outer member access then forms a bound member function that
  // calls #6 on `r`, as described in "Instance binding".
  r.(RoundWidget.Draw)();

  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #7.
  // The outer member access then forms a bound member function that
  // calls #7 on `s`.
  s.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #7.
  // The outer member access fails because we can't call
  // #7, `Draw[self: SquareWidget]()`, on a `RoundWidget` object `r`.
  r.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #5, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #6.
  // The outer member access fails because we can't call
  // #6, `Draw[self: RoundWidget]()`, on a `SquareWidget` object `s`.
  s.(RoundWidget.Draw)();
}

base class WidgetBase {
  // ✅ OK, even though `WidgetBase` does not implement `Renderable`.
  alias Draw = Renderable.Draw;

  fn DrawAll[T:! Renderable](v: Vector(T)) {
    for (var w: T in v) {
      // ✅ OK. Unqualified lookup for `Draw` finds alias `WidgetBase.Draw`
      // to `Renderable.Draw`, which does not perform `impl` lookup yet.
      // Then the compound member access expression performs `impl` lookup
      // into `impl T as Renderable`, since `T` is known to implement
      // `Renderable`. Finally, the member function is bound to `w` as
      // described in "Instance binding".
      w.(Draw)();

      // ❌ Error: `Self.Draw` performs `impl` lookup, which fails
      // because `WidgetBase` does not implement `Renderable`.
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
  // is `Renderable.Draw`. Then impl lookup replaces that with `Draw`
  // member of `impl TriangleWidget as Renderable`.
  t.Draw();
}
```

### `impl` lookup for compound member access

For a compound member access `a.(b)` where `b` names a member of an interface
`I`, `impl` lookup is performed for `T as I`, where:

-   If `b` is an instance member, `T` is the type of `a`. In this case,
    [instance binding](#instance-binding) is always performed.
-   Otherwise, `a` is implicitly converted to `I`, and `T` is the result of
    symbolically evaluating the converted expression. In this case,
    [instance binding](#instance-binding) is never performed.

For example:

```carbon
fn AddTwoIntegers(a: Integer, b: Integer) -> Integer {
  // Since `Addable.Add` is an instance member of `Addable`, `T`
  // is set to the type of `a`, and so uses `Integer as Addable`.
  return a.(Addable.Add)(b);
  //      ^ impl lookup and instance binding here
  // Impl lookup transforms this into #3:
  //   return a.((Integer as Addable).Add)(b);
  // which no longer requires impl lookup.

  // ❌ By the same logic, in this example, `T` is set to the
  // type of `Integer`, and so uses `type as Addable`, which
  // isn't implemented.
  return Integer.(Addable.Add)(...);
}

fn SumIntegers(v: Vector(Integer)) -> Integer {
  // Since `Addable.Sum` is a  non-instance member of `Addable`,
  // `Integer` is implicitly converted to `Addable`, and so uses
  // `Integer as Addable`.
  Integer.(Addable.Sum)(v);
  //     ^ impl lookup but no instance binding here
  // Impl lookup transforms this into #4:
  //   ((Integer as Addable).Sum)(v);
  // which no longer requires impl lookup.

  var a: Integer;
  // ❌ This is an error since `a` does not implicitly convert to
  // a type.
  a.(Addable.Sum)(v);
}
```

## Instance binding

Next, _instance binding_ may be performed. This associates an expression with a
particular object instance. For example, this is the value bound to `self` when
calling a method.

For the simple member access syntax `x.y`, if `x` is an entity that has member
names, such as a namespace or a type, then `y` is looked up within `x`, and
instance binding is not performed. Otherwise, `y` is looked up within the type
of `x` and instance binding is performed if an instance member is found.

If instance binding is performed:

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
    `self` parameter initialized by a corresponding recipient argument:

    -   If the method declares its `self` parameter with `addr`, the recipient
        argument is `&x`.
    -   Otherwise, the recipient argument is `x`.

    ```carbon
    class Blob {
      fn Mutate[addr self: Self*](n: i32);
    }
    fn F(p: Blob*) {
      // ✅ OK, forms bound method `((*p).M)` and calls it.
      // This calls `Blob.Mutate` with `self` initialized by `&(*p)`
      // and `n` initialized by `5`.
      (*p).Mutate(5);

      // ✅ OK, same as above.
      let bound_m: auto = (*p).Mutate;
      bound_m(5);
    }
    ```

The compound member access syntax `x.(Y)`, where `Y` names an instance member,
always performs instance binding. It is an error if `Y` is already bound to an
instance member. For example:

```carbon
interface DebugPrint {
  // instance member
  fn Print[self:Self]();
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

To get the `M` member of interface `I` for a type `T`, use `(T as I).M`, as this
doesn't attempt to perform instance binding on `T`, in contrast to `T.(I.M)`.

## Non-instance members

If instance binding is not performed, the result is the member `M` determined by
member resolution and `impl` lookup. Evaluating the member access expression
evaluates the first argument and discards the result.

An expression that names an instance member, but for which instance binding is
not performed, can only be used as the second operand of a compound member
access or as the target of an `alias` declaration.

```carbon
class C {
  fn StaticMethod();
  var field: i32;
  class Nested {}
}
fn CallStaticMethod(c: C) {
  // ✅ OK, calls `C.StaticMethod`.
  C.StaticMethod();

  // ✅ OK, evaluates expression `c`, discards the result, then
  // calls `C.StaticMethod`.
  c.StaticMethod();

  // ❌ Error: name of instance member `C.field` can only be used in
  // a member access or alias.
  C.field = 1;
  // ✅ OK, instance binding is performed by outer member access,
  // same as `c.field = 1;`
  c.(C.field) = 1;

  // ✅ OK
  let T:! type = C.Nested;
  // ❌ Error: value of `:!` binding is not compile-time because it
  // refers to local variable `c`.
  let U:! type = c.Nested;
}
```

## Non-vacuous member access restriction

The first operand of a member access expression must be used in some way: a
compound member access must result in `impl` lookup, instance binding, or both.
In a simple member access, this always holds, because the first operand is
always used for lookup.

```
interface Printable {
  fn Print[self: Self]();
}
impl i32 as Printable;

fn MemberAccess(n: i32) {
  // ✅ OK: `(i32 as Printable).Print` is the `Print` member of the
  // `i32 as Printable` facet corresponding to the `Printable.Print`
  // interface member.
  // `n.((i32 as Printable).Print)` is that member function bound to `n`.
  n.((i32 as Printable).Print)();

  // ✅ Same as above, `n.(Printable.Print)` is effectively interpreted
  // as `n.((T as Printable).Print)()`, where `T` is the type of `n`.
  // Performs impl lookup and then instance binding.
  n.(Printable.Print)();
}

interface Factory {
  fn Make() -> Self;
}
impl i32 as Factory;

// ✅ OK, member `Make` of interface `Factory`.
alias X1 = Factory.Make;
// ❌ Error, compound access without impl lookup or instance binding.
alias X2 = Factory.(Factory.Make);
// ✅ OK, member `Make` of `impl i32 as Factory`.
alias X3 = (i32 as Factory).Make;
// ❌ Error, compound access without impl lookup or instance binding.
alias X4 = i32.((i32 as Factory).Make);
```

## Tuple indexing

A tuple indexing expression is of the form:

-   _expression_ `.` _integer-literal_
-   _expression_ `->` _integer-literal_

The _expression_ is required to be of tuple type.

Each positional element of the tuple is considered to have a name that is the
corresponding decimal integer: `0`, `1`, and so on. The spelling of the
_integer-literal_ is required to exactly match one of those names, and the
result is the corresponding element of the tuple.

```
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

In a compound member access of the form:

-   _expression_ `.` `(` _expression_ `)`
-   _expression_ `->` `(` _expression_ `)`

in which the first _expression_ is a tuple and the second _expression_ is of
integer or integer literal type, the second _expression_ is required to be a
non-negative template constant that is less than the number of tuple elements,
and the result is the corresponding positional element of the tuple.

```
// ✅ `d == 43`.
let d: i32 = (41, 42, 43).(1 + 1);
// ✅ `e == 2`.
let template e:! i32 = (1, 2, 3).(0x1);
// ❌ Error: no tuple element with index 4.
let f: i32 = (1, 2).(2 * 2);

// ✅ `n == 3`.
let n: i32 = p->(e);
```

## Precedence and associativity

Member access expressions associate left-to-right:

```
class A {
  class B {
    fn F();
  }
}
interface B {
  fn F();
}
impl A as B;

fn Use(a: A) {
  // Calls member `F` of class `A.B`.
  (a.B).F();
  // Calls member `F` of interface `B`, as implemented by type `A`.
  a.(B.F)();
  // Same as `(a.B).F()`.
  a.B.F();
}
```

Member access has lower precedence than primary expressions, and higher
precedence than all other expression forms.

```
// ✅ OK, `*` has lower precedence than `.`. Same as `(A.B)*`.
var p: A.B*;
// ✅ OK, `1 + (X.Y)` not `(1 + X).Y`.
var n: i32 = 1 + X.Y;
```

## Alternatives considered

-   [Separate syntax for static versus dynamic access, such as `::` versus `.`](/proposals/p0989.md#separate-syntax-for-static-versus-dynamic-access)
-   [Use a different lookup rule for names in templates](/proposals/p0989.md#use-a-different-lookup-rule-in-templates)
-   [Meaning of `Type.Interface`](/proposals/p0989.md#meaning-of-typeinterface)

## References

-   Proposal
    [#989: member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   [Question for leads: constrained template name lookup](https://github.com/carbon-language/carbon-lang/issues/949)
-   Proposal
    [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
-   Proposal
    [#2550: Simplified package declaration for the `Main` package](https://github.com/carbon-language/carbon-lang/pull/2550)
