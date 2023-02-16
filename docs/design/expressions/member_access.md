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
    -   [Lookup within values](#lookup-within-values)
        -   [Templates and generics](#templates-and-generics)
        -   [Lookup ambiguity](#lookup-ambiguity)
-   [`impl` lookup](#impl-lookup)
-   [Instance binding](#instance-binding)
-   [Non-instance members](#non-instance-members)
-   [Non-vacuous member access restriction](#non-vacuous-member-access-restriction)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _qualified name_ is a [word](../lexical_conventions/words.md) that is preceded
by a period. The name is found within a contextually determined entity:

-   In a member access expression, this is the entity preceding the period.
-   For a designator in a struct literal, the name is introduced as a member of
    the struct type.

A _member access expression_ allows a member of a value, type, interface,
namespace, and so on to be accessed by specifying a qualified name for the
member.

A member access expression is either a _simple_ member access expression of the
form:

-   _member-access-expression_ ::= _expression_ `.` _word_

or a _compound_ member access of the form:

-   _member-access-expression_ ::= _expression_ `.` `(` _expression_ `)`

Compound member accesses allow specifying a qualified member name.

For example:

```carbon
package Widgets api;
interface Widget {
  fn Grow[addr self: Self*](factor: f64);
}
class Cog {
  var size: i32;
  fn Make(size: i32) -> Self;
  impl as Widgets.Widget;
}

fn GrowSomeCogs() {
  var cog1: Cog = Cog.Make(1);
  var cog2: Cog = cog1.Make(2);
  let cog1_size: i32 = cog1.size;
  cog1.Grow(1.5);
  cog2.(Cog.Grow)(cog1_size as f64);
  cog1.(Widget.Grow)(1.1);
  cog2.(Widgets.Cog.(Widgets.Widget.Grow))(1.9);
}
```

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

### Lookup within values

When the first operand is not a package or namespace name, there are three
remaining cases we wish to support:

-   The first operand is a value, and lookup should consider members of the
    value's type.
-   The first operand is a type, and lookup should consider members of that
    type. For example, `i32.Least` should find the member constant `Least` of
    the type `i32`.
-   The first operand is a type-of-type, and lookup should consider members of
    that type-of-type. For example, `Addable.Add` should find the member
    function `Add` of the interface `Addable`. Because a type-of-type is a type,
    this is a special case of the previous bullet.

Note that because a type is a value, and a type-of-type is a type, these cases
are overlapping and not entirely separable.

If any of the above lookups ever looks for members of a type parameter, it
should consider members of the type-of-type, treating the type parameter as an
archetype.

**Note:** If lookup is performed into a type that involves a template parameter,
the lookup will be performed both in the context of the template definition and
in the context of the template instantiation, as described in
[templates and generics](#templates-and-generics).

For a simple member access, the word is looked up in the following types:

-   If the first operand can be evaluated and evaluates to a type, that type.
-   If the type of the first operand can be evaluated, that type.
-   If the type of the first operand is a generic type parameter, and the type
    of that generic type parameter can be evaluated, that type-of-type.

The results of these lookups are [combined](#lookup-ambiguity).

For a compound member access, the second operand is evaluated as a constant to
determine the member being accessed. The evaluation is required to succeed and
to result in a member of a type or interface.

For example:

```
interface Printable {
  fn Print[self: Self]();
}
external impl i32 as Printable;
class Point {
  var x: i32;
  var y: i32;
  // Internal impl injects the name `Print` into class `Point`.
  impl as Printable;
}

fn PrintPointTwice() {
  var p: Point = {.x = 0, .y = 0};

  // ✅ OK, `x` found in type of `p`, namely `Point`.
  p.x = 1;
  // ✅ OK, `y` found in the type `Point`.
  p.(Point.y) = 1;

  // ✅ OK, `Print` found in type of `p`, namely `Point`.
  p.Print();
  // ✅ OK, `Print` found in the type `Printable`.
  p.(Printable.Print)();
}
fn GenericPrint[T:! Printable](a: T) {
  // ✅ OK, type of `a` is the type parameter `T`;
  // `Print` found in the type of `T`, namely `Printable`.
  a.Print();
}
fn CallGenericPrint(p: Point) {
  GenericPrint(p);
}
```

#### Templates and generics

If the value or type of the first operand depends on a template or generic
parameter, the lookup is performed from a context where the value of that
parameter is unknown. Evaluation of an expression involving the parameter may
still succeed, but will result in a symbolic value involving that parameter.

```
class GenericWrapper(T:! type) {
  var field: T;
}
fn F[T:! type](x: GenericWrapper(T)) -> T {
  // ✅ OK, finds `GenericWrapper(T).field`.
  return x.field;
}

class TemplateWrapper(template T:! type) {
  var field: T;
}
fn G[template T:! type](x: TemplateWrapper(T)) -> T {
  // 🤷 Not yet decided.
  return x.field;
}
```

> **TODO:** The behavior of `G` above is not yet fully decided. If class
> templates can be specialized, then we cannot know the members of
> `TemplateWrapper(T)` without knowing `T`, so this first lookup will find
> nothing. In any case, as described below, the lookup will be performed again
> when `T` is known.

If the value or type depends on any template parameters, the lookup is redone
from a context where the values of those parameters are known, but where the
values of any generic parameters are still unknown. The lookup results from
these two contexts are [combined](#lookup-ambiguity).

**Note:** All lookups are done from a context where the values of any generic
parameters that are in scope are unknown. Unlike for a template parameter, the
actual value of a generic parameter never affects the result of member
resolution.

```carbon
class Cowboy { fn Draw[self: Self](); }
interface Renderable {
  fn Draw[self: Self]();
}
external impl Cowboy as Renderable { fn Draw[self: Self](); }
fn DrawDirect(c: Cowboy) { c.Draw(); }
fn DrawGeneric[T:! Renderable](c: T) { c.Draw(); }
fn DrawTemplate[template T:! Renderable](c: T) { c.Draw(); }

fn Draw(c: Cowboy) {
  // ✅ Calls member of `Cowboy`.
  DrawDirect(c);
  // ✅ Calls member of `impl Cowboy as Renderable`.
  DrawGeneric(c);
  // ❌ Error: ambiguous.
  DrawTemplate(c);
}

class RoundWidget {
  external impl as Renderable {
    fn Draw[self: Self]();
  }
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  fn Draw[self: Self]() {}
  external impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn DrawWidget(r: RoundWidget, s: SquareWidget) {
  // ✅ OK, lookup in type and lookup in type-of-type find the same entity.
  DrawTemplate(r);

  // ✅ OK, lookup in type and lookup in type-of-type find the same entity.
  DrawTemplate(s);

  // ✅ OK, found in type.
  r.Draw();
  s.Draw();
}
```

#### Lookup ambiguity

Multiple lookups can be performed when resolving a member access expression. If
more than one member is found, after performing [`impl` lookup](#impl-lookup) if
necessary, the lookup is ambiguous, and the program is invalid. Similarly, if no
members are found, the program is invalid. Otherwise, the result of combining
the lookup results is the unique member that was found.

## `impl` lookup

When the second operand of a member access expression resolves to a member of an
interface `I`, and the first operand is a value other than a type-of-type,
_`impl` lookup_ is performed to map the member of the interface to the
corresponding member of the relevant `impl`. The member of the `impl` replaces
the member of the interface in all further processing of the member access
expression.

```carbon
interface Addable {
  // #1
  fn Add[self: Self](other: Self) -> Self;
  // #2
  default fn Sum[Seq:! Iterable where .ValueType = Self](seq: Seq) -> Self {
    // ...
  }
}

class Integer {
  impl as Addable {
    // #3
    fn Add[self: Self](other: Self) -> Self;
    // #4, generated from default implementation for #2.
    // fn Sum[...](...);
  }
}

fn SumIntegers(v: Vector(Integer)) -> Integer {
  // Member resolution resolves the name `Sum` to #2.
  // `impl` lookup then locates the `impl Integer as Addable`,
  // and determines that the member access refers to #4,
  // which is then called.
  return Integer.Sum(v);
}

fn AddTwoIntegers(a: Integer, b: Integer) -> Integer {
  // Member resolution resolves the name `Add` to #1.
  // `impl` lookup then locates the `impl Integer as Addable`,
  // and determines that the member access refers to #3.
  // Finally, instance binding will be performed as described later.
  // This can be written more verbosely and explicitly as any of:
  // -   `return a.(Integer.Add)(b);`
  // -   `return a.(Addable.Add)(b);`
  // -   `return a.(Integer.(Addable.Add))(b);`
  return a.Add(b);
}
```

The type `T` that is expected to implement `I` depends on the first operand of
the member access expression, `V`:

-   If `V` can be evaluated and evaluates to a type, then `T` is `V`.
    ```carbon
    // `V` is `Integer`. `T` is `V`, which is `Integer`.
    // Alias refers to #2.
    alias AddIntegers = Integer.Add;
    ```
-   Otherwise, `T` is the type of `V`.
    ```carbon
    let a: Integer = {};
    // `V` is `a`. `T` is the type of `V`, which is `Integer`.
    // `a.Add` refers to #2.
    let twice_a: Integer = a.Add(a);
    ```

The appropriate `impl T as I` implementation is located. The program is invalid
if no such `impl` exists. When `T` or `I` depends on a generic parameter, a
suitable constraint must be specified to ensure that such an `impl` will exist.
When `T` or `I` depends on a template parameter, this check is deferred until
the argument for the template parameter is known.

`M` is replaced by the member of the `impl` that corresponds to `M`.

```carbon
interface I {
  // #1
  default fn F[self: Self]() {}
  let N:! i32;
}
class C {
  impl as I where .N = 5 {
    // #2
    fn F[self: C]() {}
  }
}

// `V` is `I` and `M` is `I.F`. Because `V` is a type-of-type,
// `impl` lookup is not performed, and the alias binds to #1.
alias A1 = I.F;

// `V` is `C` and `M` is `I.F`. Because `V` is a type, `impl`
// lookup is performed with `T` being `C`, and the alias binds to #2.
alias A2 = C.F;

let c: C = {};

// `V` is `c` and `M` is `I.N`. Because `V` is a non-type, `impl`
// lookup is performed with `T` being the type of `c`, namely `C`, and
// `M` becomes the associated constant from `impl C as I`.
// The value of `Z` is 5.
let Z: i32 = c.N;
```

[Instance binding](#instance-binding) may also apply if the member is an
instance member.

```carbon
var c: C;
// `V` is `c` and `M` is `I.F`. Because `V` is not a type, `T` is the
// type of `c`, which is `C`. `impl` lookup is performed, and `M` is
// replaced with #2. Then instance binding is performed.
c.F();
```

**Note:** When an interface member is added to a class by an alias, `impl`
lookup is not performed as part of handling the alias, but will happen when
naming the interface member as a member of the class.

```carbon
interface Renderable {
  // #1
  fn Draw[self: Self]();
}

class RoundWidget {
  external impl as Renderable {
    // #2
    fn Draw[self: Self]();
  }
  // `Draw` names the member of the `Renderable` interface.
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  // #3
  fn Draw[self: Self]() {}
  external impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn DrawWidget(r: RoundWidget, s: SquareWidget) {
  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #1, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #2.
  // The outer member access then forms a bound member function that
  // calls #2 on `r`, as described in "Instance binding".
  r.(RoundWidget.Draw)();

  // ✅ OK: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #3.
  // The outer member access then forms a bound member function that
  // calls #3 on `s`.
  s.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `SquareWidget`, #3.
  // The outer member access fails because we can't call
  // #3, `Draw[self: SquareWidget]()`, on a `RoundWidget` object `r`.
  r.(SquareWidget.Draw)();

  // ❌ Error: In the inner member access, the name `Draw` resolves to the
  // member `Draw` of `Renderable`, #1, which `impl` lookup replaces with
  // the member `Draw` of `impl RoundWidget as Renderable`, #2.
  // The outer member access fails because we can't call
  // #2, `Draw[self: RoundWidget]()`, on a `SquareWidget` object `s`.
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

class TriangleWidget extends WidgetBase {
  external impl as Renderable;
}
fn DrawTriangle(t: TriangleWidget) {
  // ✅ OK: name `Draw` resolves to `Draw` member of `WidgetBase`, which
  // is `Renderable.Draw`. Then impl lookup replaces that with `Draw`
  // member of `impl TriangleWidget as Renderable`.
  t.Draw();
}
```

## Instance binding

If member resolution and `impl` lookup produce a member `M` that is an instance
member -- that is, a field or a method -- and the first operand `V` of `.` is a
value other than a type, then _instance binding_ is performed, as follows:

-   For a field member in class `C`, `V` is required to be of type `C` or of a
    type derived from `C`. The result is the corresponding subobject within `V`.
    The result is an lvalue if `V` is an lvalue.

    ```carbon
    var dims: auto = {.width = 1, .height = 2};
    // `dims.width` denotes the field `width` of the object `dims`.
    Print(dims.width);
    // `dims` is an lvalue, so `dims.height` is an lvalue.
    dims.height = 3;
    ```

-   For a method, the result is a _bound method_, which is a value `F` such that
    a function call `F(args)` behaves the same as a call to `M(args)` with the
    `self` parameter initialized by a corresponding recipient argument:

    -   If the method declares its `self` parameter with `addr`, the recipient
        argument is `&V`.
    -   Otherwise, the recipient argument is `V`.

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

## Non-instance members

If instance binding is not performed, the result is the member `M` determined by
member resolution and `impl` lookup. Evaluating the member access expression
evaluates `V` and discards the result.

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

  // ✅ OK, evaluates expression `c` then calls `C.StaticMethod`.
  c.StaticMethod();

  // ❌ Error: name of instance member `C.field` can only be used in a
  // member access or alias.
  C.field = 1;
  // ✅ OK, instance binding is performed by outer member access,
  // same as `c.field = 1;`
  c.(C.field) = 1;

  // ✅ OK
  let T:! type = C.Nested;
  // ❌ Error: value of `:!` binding is not constant because it
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
external impl i32 as Printable {
  fn Print[self: Self]();
}
fn MemberAccess(n: i32) {
  // ✅ OK: `Printable.Print` is the interface member.
  // `i32.(Printable.Print)` is the corresponding member of the `impl`.
  // `n.(i32.(Printable.Print))` is a bound member function naming that member.
  n.(i32.(Printable.Print))();

  // ✅ Same as above, `n.(Printable.Print)` is effectively interpreted as
  // `n.(T.(Printable.Print))()`, where `T` is the type of `n`,
  // because `n` does not evaluate to a type. Performs impl lookup
  // and then instance binding.
  n.(Printable.Print)();
}

// ✅ OK, member `Print` of interface `Printable`.
alias X1 = Printable.Print;
// ❌ Error, compound access doesn't perform impl lookup or instance binding.
alias X2 = Printable.(Printable.Print);
// ✅ OK, member `Print` of `impl i32 as Printable`.
alias X3 = i32.(Printable.Print);
// ❌ Error, compound access doesn't perform impl lookup or instance binding.
alias X4 = i32.(i32.(Printable.Print));
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
external impl A as B;

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
