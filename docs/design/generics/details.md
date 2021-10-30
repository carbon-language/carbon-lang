# Generics: Details

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Interfaces](#interfaces)
-   [Implementing interfaces](#implementing-interfaces)
    -   [Facet type](#facet-type)
    -   [Implementing multiple interfaces](#implementing-multiple-interfaces)
    -   [External impl](#external-impl)
    -   [Qualified member names](#qualified-member-names)
-   [Generics](#generics)
    -   [Model](#model)
-   [Interfaces recap](#interfaces-recap)
-   [Type-of-types and facet types](#type-of-types-and-facet-types)
-   [Named constraints](#named-constraints)
    -   [Subtyping between type-of-types](#subtyping-between-type-of-types)
-   [Combining interfaces by anding type-of-types](#combining-interfaces-by-anding-type-of-types)
-   [Interface requiring other interfaces](#interface-requiring-other-interfaces)
    -   [Interface extension](#interface-extension)
        -   [`extends` and `impl` with named constraints](#extends-and-impl-with-named-constraints)
        -   [Diamond dependency issue](#diamond-dependency-issue)
    -   [Use case: overload resolution](#use-case-overload-resolution)
-   [Type compatibility](#type-compatibility)
-   [Adapting types](#adapting-types)
    -   [Adapter compatibility](#adapter-compatibility)
    -   [Extending adapter](#extending-adapter)
    -   [Use case: Using independent libraries together](#use-case-using-independent-libraries-together)
    -   [Adapter with stricter invariants](#adapter-with-stricter-invariants)
    -   [Application: Defining an impl for use by other types](#application-defining-an-impl-for-use-by-other-types)
-   [Associated constants](#associated-constants)
    -   [Associated class functions](#associated-class-functions)
-   [Associated types](#associated-types)
    -   [Model](#model-1)
-   [Parameterized interfaces](#parameterized-interfaces)
    -   [Impl lookup](#impl-lookup)
    -   [Parameterized named constraints](#parameterized-named-constraints)
-   [Where constraints](#where-constraints)
    -   [Constraint use cases](#constraint-use-cases)
        -   [Set an associated constant to a specific value](#set-an-associated-constant-to-a-specific-value)
        -   [Same type constraints](#same-type-constraints)
            -   [Set an associated type to a specific value](#set-an-associated-type-to-a-specific-value)
            -   [Equal generic types](#equal-generic-types)
                -   [Satisfying both type-of-types](#satisfying-both-type-of-types)
        -   [Type bound for associated type](#type-bound-for-associated-type)
            -   [Type bounds on associated types in declarations](#type-bounds-on-associated-types-in-declarations)
            -   [Type bounds on associated types in interfaces](#type-bounds-on-associated-types-in-interfaces)
        -   [Combining constraints](#combining-constraints)
        -   [Recursive constraints](#recursive-constraints)
        -   [Parameterized type implements interface](#parameterized-type-implements-interface)
        -   [Another type implements parameterized interface](#another-type-implements-parameterized-interface)
    -   [Implied constraints](#implied-constraints)
        -   [Must be legal type argument constraints](#must-be-legal-type-argument-constraints)
    -   [Open question: referencing names in the interface being defined](#open-question-referencing-names-in-the-interface-being-defined)
    -   [Manual type equality](#manual-type-equality)
        -   [`observe` declarations](#observe-declarations)
-   [Other constraints as type-of-types](#other-constraints-as-type-of-types)
    -   [Is a derived class](#is-a-derived-class)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Same implementation restriction](#same-implementation-restriction)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other impls](#example-creating-an-impl-out-of-other-impls)
    -   [Type facet of another type](#type-facet-of-another-type)
    -   [Sized types and type-of-types](#sized-types-and-type-of-types)
        -   [Model](#model-2)
    -   [`TypeId`](#typeid)
-   [Future work](#future-work)
    -   [Conditional conformance](#conditional-conformance)
    -   [Parameterized impls](#parameterized-impls)
        -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
    -   [Dynamic types](#dynamic-types)
        -   [Runtime type parameters](#runtime-type-parameters)
        -   [Runtime type fields](#runtime-type-fields)
    -   [Abstract return types](#abstract-return-types)
    -   [Interface defaults](#interface-defaults)
    -   [Evolution](#evolution)
    -   [Testing](#testing)
    -   [Operator overloading](#operator-overloading)
    -   [Impls with state](#impls-with-state)
    -   [Generic associated types and higher-ranked types](#generic-associated-types-and-higher-ranked-types)
        -   [Generic associated types](#generic-associated-types)
        -   [Higher-ranked types](#higher-ranked-types)
    -   [Field requirements](#field-requirements)
    -   [Generic type specialization](#generic-type-specialization)
    -   [Bridge for C++ customization points](#bridge-for-c-customization-points)
    -   [Variadic arguments](#variadic-arguments)
    -   [Range constraints on generic integers](#range-constraints-on-generic-integers)
-   [References](#references)

<!-- tocstop -->

## Overview

This document goes into the details of the design of generic type parameters.

Imagine we want to write a function parameterized by a type argument. Maybe our
function is `PrintToStdout` and let's say we want to operate on values that have
a type for which we have an implementation of the `ConvertibleToString`
interface. The `ConvertibleToString` interface has a `ToString` method returning
a string. To do this, we give the `PrintToStdout` function two parameters: one
is the value to print, let's call that `val`, the other is the type of that
value, let's call that `T`. The type of `val` is `T`, what is the type of `T`?
Well, since we want to let `T` be any type implementing the
`ConvertibleToString` interface, we express that in the "interfaces are
type-of-types" model by saying the type of `T` is `ConvertibleToString`.

Since we can figure out `T` from the type of `val`, we don't need the caller to
pass in `T` explicitly, so it can be an
[deduced argument](terminology.md#deduced-parameter) (also see
[deduced argument](overview.md#deduced-parameters) in the Generics overview
doc). Basically, the user passes in a value for `val`, and the type of `val`
determines `T`. `T` still gets passed into the function though, and it plays an
important role -- it defines the implementation of the interface. We can think
of the interface as defining a struct type whose members are function pointers,
and an implementation of an interface as a value of that struct with actual
function pointer values. So an implementation is a table of function pointers
(one per function defined in the interface) that gets passed into a function as
the type argument. For more on this, see [the model section](#model) below.

In addition to function pointer members, interfaces can include any constants
that belong to a type. For example, the
[type's size](#sized-types-and-type-of-types) (represented by an integer
constant member of the type) could be a member of an interface and its
implementation. There are a few cases why we would include another interface
implementation as a member:

-   [associated types](#associated-types)
-   [type parameters](#parameterized-interfaces)
-   [interface requirements](#interface-requiring-other-interfaces)

The function can decide whether that type argument is passed in
[statically](terminology.md#static-dispatch-witness-table) (basically generating
a separate function body for every different type passed in) by using the
"generic argument" syntax (`:!`, see [the generics section](#generics) below) or
[dynamically](terminology.md#dynamic-dispatch-witness-table) using the regular
argument syntax (just a colon, `:`, see
[the runtime type parameters section](#runtime-type-parameters) below). Either
way, the interface contains enough information to
[type and definition check](terminology.md#complete-definition-checking) the
function body -- you can only call functions defined in the interface in the
function body. Contrast this with making the type a template argument, where you
could just use `Type` instead of an interface and it will work as long as the
function is only called with types that allow the definition of the function to
compile. The interface bound has other benefits:

-   allows the compiler to deliver clearer error messages,
-   documents expectations, and
-   expresses that a type has certain semantics beyond what is captured in its
    member function names and signatures.

The last piece of the puzzle is how the caller of the function can produce a
value with the right type. Let's say the user has a value of type `Song`, and of
course songs have all sorts of functionality. If we want a `Song` to be printed
using the `PrintToStdout` function, it needs to implement the
`ConvertibleToString` interface. Note that we _don't_ say that `Song` is of type
`ConvertibleToString` but instead that it has a "facet type". This means there
is another type, called `Song as ConvertibleToString`, with the following
properties:

-   `Song as ConvertibleToString` has the same _data representation_ as `Song`.
-   `Song as ConvertibleToString` is an implementation of the interface
    `ConvertibleToString`. The functions of `Song as ConvertibleToString` are
    just implementations of the names and signatures defined in the
    `ConvertibleToString` interface, like `ToString`, and not the functions
    defined on `Song` values.
-   Carbon will implicitly convert values from type `Song` to type
    `Song as ConvertibleToString` when calling a function that can only accept
    types of type `ConvertibleToString`.
-   In the normal case where the implementation of `ConvertibleToString` for
    `Song` is not defined as `external`, every member of
    `Song as ConvertibleToString` is also a member of `Song`. This includes
    members of `ConvertibleToString` that are not explicitly named in the `impl`
    definition but have defaults.
-   You may access the `ToString` function for a `Song` value `w` by writing a
    _qualified_ function call, like `w.(ConvertibleToString.ToString)()`. The
    same effect may be achieved by casting, as in
    `(w as (Song as ConvertibleToString)).ToString()`. This qualified syntax is
    available whether or not the implementation is defined as `external`.
-   If other interfaces are implemented for `Song`, they are also implemented
    for `Song as ConvertibleToString`. The only thing that changes when casting
    a `Song` `w` to `Song as ConvertibleToString` are the names that are
    accessible without using the qualification syntax. A
    `Song as ConvertibleToString` value can likewise be cast to a `Song`; a
    `Song` acts just like another facet type for these purposes.

We define these facet types (alternatively, interface implementations) either
with the type, with the interface, or somewhere else where Carbon can be
guaranteed to see when needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

If `Song` doesn't implement an interface or we would like to use a different
implementation of that interface, we can define another type that also has the
same data representation as `Song` that has whatever different interface
implementations we want. However, Carbon won't implicitly convert to that other
type, the user will have to explicitly cast to that type in order to select
those alternate implementations. For more on this, see
[the adapting type section](#adapting-types) below.

## Interfaces

An [interface](terminology.md#interface), defines an API that a given type can
implement. For example, an interface capturing a linear-algebra vector API might
have two methods:

```
interface Vector {
  // Here "Self" means "the type implementing this interface".
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: Double) -> Self;
}
```

The syntax here is to match
[how the same members would be defined in a type](/docs/design/classes.md#methods).
Each declaration in the interface defines an
[associated entity](terminology.md#associated-entity). In this example, `Vector`
has two associated methods, `Add` and `Scale`.

An interface defines a type-of-type, that is a type whose values are types. The
values of an interface are specifically
[facet types](terminology.md#facet-type), by which we mean types that are
declared as specifically implementing **exactly** this interface, and which
provide definitions for all the functions (and other members) declared in the
interface.

## Implementing interfaces

Carbon interfaces are ["nominal"](terminology.md#nominal-interfaces), which
means that types explicitly describe how they implement interfaces. An
["impl"](terminology.md#impls-implementations-of-interfaces) defines how one
interface is implemented for a type. Every associated entity is given a
definition. Different types satisfying `Vector` can have different definitions
for `Add` and `Scale`, so we say their definitions are _associated_ with what
type is implementing `Vector`. The `impl` defines what is associated with the
type for that interface.

Impls may be defined inline inside the type definition:

```
class Point {
  var x: Double;
  var y: Double;
  impl as Vector {
    // In this scope, "Self" is an alias for "Point".
    fn Add[me: Self](b: Self) -> Self {
      return {.x = a.x + b.x, .y = a.y + b.y};
    }
    fn Scale[me: Self](v: Double) -> Self {
      return {.x = a.x * v, .y = a.y * v};
    }
  }
}
```

Interfaces that are implemented inline contribute to the type's API:

```
var p1: Point = {.x = 1.0, .y = 2.0};
var p2: Point = {.x = 2.0, .y = 4.0};
Assert(p1.Scale(2.0) == p2);
Assert(p1.Add(p1) == p2);
```

**Comparison with other languages:** Rust defines implementations lexically
outside of the `class` definition. This Carbon approach means that a type's API
is described by declarations inside the `class` definition and doesn't change
afterwards.

**References:** This interface implementation syntax was accepted in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553). In
particular, see
[the alternatives considered](/proposals/p0553.md#interface-implementation-syntax).

### Facet type

The `impl` definition defines a [facet type](terminology.md#facet-type):
`Point as Vector`. While the API of `Point` includes the two fields `x` and `y`
along with the `Add` and `Scale` methods, the API of `Point as Vector` _only_
has the `Add` and `Scale` methods of the `Vector` interface. The facet type
`Point as Vector` is [compatible](terminology.md#compatible-types) with `Point`,
meaning their data representations are the same, so we allow you to convert
between the two freely:

```
var a: Point = {.x = 1.0, .y = 2.0};
// `a` has `Add` and `Scale` methods:
a.Add(a.Scale(2.0));

// Cast from Point implicitly
var b: Point as Vector = a;
// `b` has `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

// Will also implicitly convert when calling functions:
fn F(c: Point as Vector, d: Point) {
  d.Add(c.Scale(2.0));
}
F(a, b);

// Explicit casts
var z: Point as Vector = (a as (Point as Vector)).Scale(3.0);
z.Add(b);
var w: Point = z as Point;
```

These [conversions](terminology.md#subtyping-and-casting) change which names are
exposed in the type's API, but as much as possible we don't want the meaning of
any given name to change. Instead we want these casts to simply change the
subset of names that are visible.

**Note:** In general the above is written assuming that casts are written
"`a as T`" where `a` is a value and `T` is the type to cast to. When we write
`Point as Vector`, the value `Point` is a type, and `Vector` is a type of a
type, or a "type-of-type".

**Note:** A type may implement any number of different interfaces, but may
provide at most one implementation of any single interface. This makes the act
of selecting an implementation of an interface for a type unambiguous throughout
the whole program, so for example `Point as Vector` is well defined.

We don't expect users to ordinarily name facet types explicitly in source code.
Instead, values are implicitly converted to a facet type as part of calling a
generic function, as described in the [Generics](#generics) section.

### Implementing multiple interfaces

To implement more than one interface when defining a type, simply include an
`impl` block per interface.

```
class Point {
  var x: Double;
  var y: Double;
  impl as Vector {
    fn Add[me: Self](b: Self) -> Self { ... }
    fn Scale[me: Self](v: Double) -> Self { ... }
  }
  impl as Drawable {
    fn Draw[me: Self]() { ... }
  }
}
```

In this case, all the functions `Add`, `Scale`, and `Draw` end up a part of the
API for `Point`. This means you can't implement two interfaces that have a name
in common (unless you use an `external impl` for one or both, as described
below).

```
class GameBoard {
  impl as Drawable {
    fn Draw[me: Self]() { ... }
  }
  impl as EndOfGame {
    // ❌ Error: `GameBoard` has two methods named
    // `Draw` with the same signature.
    fn Draw[me: Self]() { ... }
    fn Winner[me: Self](player: i32) { ... }
  }
}
```

**Open question:** Should we have some syntax for the case where you want both
names to be given the same implementation? It seems like that might be a common
case, but we won't really know if this is an important case until we get more
experience.

```
class Player {
  var name: String;
  impl as Icon {
    fn Name[me: Self]() -> String { return this.name; }
    // ...
  }
  impl as GameUnit {
    // Possible syntax for defining `GameUnit.Name` as
    // the same as `Icon.Name`:
    alias Name = Icon.Name;
    // ...
  }
}
```

### External impl

Interfaces may also be implemented for a type externally, by using the
`external impl` construct which takes the name of an existing type:

```
class Point2 {
  var x: Double;
  var y: Double;
}

external impl Point2 as Vector {
  // In this scope, "Self" is an alias for "Point2".
  fn Add[me: Self](b: Self) -> Self {
    return {.x = a.x + b.x, .y = a.y + b.y};
  }
  fn Scale[me: Self](v: Double) -> Self {
    return {.x = a.x * v, .y = a.y * v};
  }
}
```

**References:** The external interface implementation syntax was decided in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553). In
particular, see
[the alternatives considered](/proposals/p0553.md#interface-implementation-syntax).

The `external impl` statement is allowed to be defined in a different library
from `Point2`, restricted by [the coherence/orphan rules](#impl-lookup) that
ensure that the implementation of an interface won't change based on imports. In
particular, the `external impl` statement is allowed in the library defining the
interface (`Vector` in this case) in addition to the library that defines the
type (`Point2` here). This (at least partially) addresses
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

We don't want the API of `Point2` to change based on what is imported though. So
the `external impl` statement does not add the interface's methods to the type.
It would be particularly bad if two different libraries implemented interfaces
with conflicting names both affected the API of a single type. The result is you
can find all the names of direct (unqualified) members of a type in the
definition of that type. The only thing that may be in another library is an
`impl` of an interface.

On the other hand, if we convert to the facet type, those methods do become
visible:

```
var a: Point2 = {.x = 1.0, .y = 2.0};
// `a` does *not* have `Add` and `Scale` methods:
// ❌ Error: a.Add(a.Scale(2.0));

// Convert from Point2 implicitly
var b: Point2 as Vector = a;
// `b` does have `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

fn F(c: Point2 as Vector) {
  // Can call `Add` and `Scale` on `c` even though we can't on `a`.
  c.Add(c.Scale(2.0));
}
F(a);
```

You might intentionally use `external impl` to implement an interface for a type
to avoid cluttering the API of that type, for example to avoid a name collision.
A syntax for reusing method implementations allows us to do this selectively
when needed:

```
class Point3 {
  var x: Double;
  var y: Double;
  fn Add[me: Self](b: Self) -> Self {
    return {.x = a.x + b.x, .y = a.y + b.y};
  }
}

external impl Point3 as Vector {
  alias Add = Point3.Add;  // Syntax TBD
  fn Scale[me: Self](v: Double) -> Self {
    return {.x = a.x * v, .y = a.y * v};
  }
}
```

With this definition, `Point3` includes `Add` in its API but not `Scale`, while
`Point3 as Vector` includes both. This maintains the property that you can
determine the API of a type by looking at its definition. In this case, the
`external impl` may be defined lexically inside the scope of the class.

```
class Point3 {
  var x: Double;
  var y: Double;
  fn Add[me: Self](b: Self) -> Self {
    return {.x = a.x + b.x, .y = a.y + b.y};
  }
  // Type before `as` is optional and defaults to the current class.
  external impl as Vector {
    alias Add = Point3.Add;  // Syntax TBD
    fn Scale[me: Self](v: Double) -> Self {
      return {.x = a.x * v, .y = a.y * v};
    }
  }
}

// OR:

class Point3 {
  var x: Double;
  var y: Double;
  external impl as Vector {
    fn Add[me: Self](b: Self) -> Self {
      return {.x = a.x + b.x, .y = a.y + b.y};
    }
    fn Scale[me: Self](v: Double) -> Self {
      return {.x = a.x * v, .y = a.y * v};
    }
  }
  alias Add = Vector.Add;  // Syntax TBD
}
```

Being defined lexically inside the class means that implementation is available
to other members defined in the class. For example, it would allow implementing
another interface or method that requires this interface to be implemented.

**Open question:** Do implementations need to be defined lexically inside the
class to get access to private members, or is it sufficient to be defined in the
same library as the class?

**Rejected alternative:** We could allow types to have different APIs in
different files based on explicit configuration in that file. For example, we
could support a declaration that a given interface or a given method of an
interface is "in scope" for a particular type in this file. With that
declaration, the method could be called unqualified. This avoids most concerns
arising from name collisions between interfaces. It has a few downsides though:

-   It increases variability between files, since the same type will have
    different APIs depending on these declarations. This makes it harder to
    copy-paste code between files.
-   It makes reading code harder, since you have to search the file for these
    declarations that affect name lookup.

**Comparison with other languages:** Both Rust and Swift support external
implementation.
[Swift's syntax](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID277)
does this as an "extension" of the original type. In Rust, all implementations
are external as in
[this example](https://doc.rust-lang.org/rust-by-example/trait.html). Unlike
Swift and Rust, we don't allow a type's API to be modified outside its
definition. So in Carbon a type's API is consistent no matter what is imported,
unlike Swift and Rust.

### Qualified member names

Given a value of type `Point2` and an interface `Vector` implemented for that
type, you can access the methods from that interface using the member's
_qualified name_, whether or not the implementation is done externally with an
`external impl` statement:

```
var p1: Point2 = {.x = 1.0, .y = 2.0};
var p2: Point2 = {.x = 2.0, .y = 4.0};
Assert(p1.(Vector.Scale)(2.0) == p2);
Assert(p1.(Vector.Add)(p1) == p2);
```

Note that the name in the parens is looked up in the containing scope, not in
the names of members of `Point2`. So if there was another interface `Drawable`
with method `Draw` defined in the `Plot` package also implemented for `Point2`,
as in:

```
package Plot;
import Points;

interface Drawable {
  fn Draw[me: Self]();
}

external impl Points.Point2 as Drawable { ... }
```

You could access `Draw` with a qualified name:

```
import Plot;
import Points;

var p: Points.Point2 = {.x = 1.0, .y = 2.0};
p.(Plot.Drawable.Draw)();
```

**Comparison with other languages:** This is intended to be analogous to, in
C++, adding `ClassName::` in front of a member name to disambiguate, such as
[names defined in both a parent and child class](https://stackoverflow.com/questions/357307/how-to-call-a-parent-class-function-from-derived-class-function).

## Generics

Now let us write a function that can accept values of any type that has
implemented the `Vector` interface:

```
fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: Double) -> T {
  return a.Add(b).Scale(s);
}
var v: Point = AddAndScaleGeneric(a, w, 2.5);
```

Here `T` is a type whose type is `Vector`. The `:!` syntax means that `T` is a
_[generic parameter](terminology.md#generic-versus-template-parameters)_, that
is it must be known to the caller but we will only use the information present
in the signature of the function to typecheck the body of `AddAndScaleGeneric`'s
definition. In this case, we know that any value of type `T` implements the
`Vector` interface and so has an `Add` and a `Scale` method.

When we call `AddAndScaleGeneric`, we need to determine the value of `T` to use
when passed values with type `Point`. Since `T` has type `Vector`, the compiler
simply sets `T` to `Point as Vector`. This
[cast](terminology.md#subtyping-and-casting)
[erases](terminology.md#type-erasure) all of the API of `Point` and substitutes
the api of `Vector`, without changing anything about the data representation. It
acts like we called this non-generic function, found by setting `T` to
`Point as Vector`:

```
fn AddAndScaleForPointAsVector(
      a: Point as Vector, b: Point as Vector, s: Double)
      -> Point as Vector {
  return a.Add(b).Scale(s);
}
// May still be called with Point arguments, due to implicit conversions.
// Similarly the return value can be implicitly converted to a Point.
var v2: Point = AddAndScaleForPointAsVector(a, w, 2.5);
```

Since `Point` implements `Vector` inline, `Point` also has definitions for `Add`
and `Scale`:

```
fn AddAndScaleForPoint(a: Point, b: Point, s: Double) -> Point {
  return a.Add(b).Scale(s);
}

AddAndScaleForPoint(a, w, 2.5);
```

However, for another type implementing `Vector` but out-of-line using an
`external impl` statement, such as `Point2`, the situation is different:

```
fn AddAndScaleForPoint2(a: Point2, b: Point2, s: Double) -> Point2 {
  // ❌ ERROR: `Point2` doesn't have `Add` or `Scale` methods.
  return a.Add(b).Scale(s);
}
```

Even though `Point2` doesn't have `Add` and `Scale` methods, it still implements
`Vector` and so can still call `AddAndScaleGeneric`:

```
var a2: Point2 = {.x = 1.0, .y = 2.0};
var w2: Point2 = {.x = 3.0, .y = 4.0};
var v3: Point2 = AddAndScaleGeneric(a, w, 2.5);
```

**References:** The `:!` syntax was accepted in
[proposal #676](https://github.com/carbon-language/carbon-lang/pull/676).

### Model

The underlying model here is interfaces are
[type-of-types](terminology.md#type-of-type), in particular, the type of
[facet types](terminology.md#facet-type):

-   [Interfaces](#interfaces) are types of
    [witness tables](terminology.md#witness-tables)
-   Facet types (defined by [Impls](#implementing-interfaces)) are
    [witness table](terminology.md#witness-tables) values
-   The compiler rewrites functions with an implicit type argument
    (`fn Foo[InterfaceName:! T](...)`) to have an actual argument with type
    determined by the interface, and supplied at the callsite using a value
    determined by the impl.

For the example above, [the Vector interface](#interfaces) could be thought of
defining a witness table type like:

```
class Vector {
  // Self is the representation type, which is only
  // known at compile time.
  var Self:! Type;
  // `fnty` is **placeholder** syntax for a "function type",
  // so `Add` is a function that takes two `Self` parameters
  // and returns a value of type `Self`.
  var Add: fnty(a: Self, b: Self) -> Self;
  var Scale: fnty(a: Self, v: Double) -> Self;
}
```

The [impl of Vector for Point](#implementing-interfaces) would be a value of
this type:

```
var VectorForPoint: Vector  = {
    .Self = Point,
    // `lambda` is **placeholder** syntax for defining a
    // function value.
    .Add = lambda(a: Point, b: Point) -> Point {
      return {.x = a.x + b.x, .y = a.y + b.y};
    },
    .Scale = lambda(a: Point, v: Double) -> Point {
      return {.x = a.x * v, .y = a.y * v};
    },
};
```

Finally we can define a generic function and call it, like
[`AddAndScaleGeneric` from the "Generics" section](#generics) by making the
witness table an explicit argument to the function:

```
fn AddAndScaleGeneric
    (t:! Vector, a: t.Self, b: t.Self, s: Double) -> t.Self {
  return t.Scale(t.Add(a, b), s);
}
// Point implements Vector.
var v: Point = AddAndScaleGeneric(VectorForPoint, a, w, 2.5);
```

The rule is that generic arguments (declared using `:!`) are passed at compile
time, so the actual value of the `t` argument here can be used to generate the
code for `AddAndScaleGeneric`. So `AddAndScaleGeneric` is using a
[static-dispatch witness table](terminology.md#static-dispatch-witness-table).

## Interfaces recap

Interfaces have a name and a definition.

The definition of an interface consists of a set of declarations. Each
declaration defines a requirement for any `impl` that is in turn a capability
that consumers of that `impl` can rely on. Typically those declarations also
have names, useful for both saying how the `impl` satisfies the requirement and
accessing the capability.

Interfaces are ["nominal"](terminology.md#nominal-interfaces), which means their
name is significant. So two interfaces with the same body definition but
different names are different, just like two classes with the same definition
but different names are considered different types. For example, lets say we
define another interface, say `LegoFish`, with the same `Add` and `Scale` method
signatures. Implementing `Vector` would not imply an implementation of
`LegoFish`, because the `impl` definition explicitly refers to the name
`Vector`.

An interface's name may be used in a few different contexts:

-   to define [an `impl` for a type](#implementing-interfaces),
-   as a namespace name in [a qualified name](#qualified-member-names), and
-   as a [type-of-type](terminology.md#type-of-type) for
    [a generic type parameter](#generics).

While interfaces are examples of type-of-types, type-of-types are a more general
concept, for which interfaces are a building block.

## Type-of-types and facet types

A [type-of-type](terminology.md#type-of-type) consists of a set of requirements
and a set of names. Requirements are typically a set of interfaces that a type
must satisfy (though other kinds of requirements are added below). The names are
aliases for qualified names in those interfaces.

An interface is one particularly simple example of a type-of-type. For example,
`Vector` as a type-of-type has a set of requirements consisting of the single
interface `Vector`. Its set of names consists of `Add` and `Scale` which are
aliases for the corresponding qualified names inside `Vector` as a namespace.

The requirements determine which types may be converted to a given type-of-type.
The result of converting a type `T` to a type-of-type `I` (written `T as I`) is
called a facet type, you might say a facet type `F` is the `I` facet of `T` if
`F` is `T as I`. The API of `F` is determined by the set of names in the
type-of-type.

This general structure of type-of-types holds not just for interfaces, but
others described in the rest of this document.

## Named constraints

If the interfaces discussed above are the building blocks for type-of-types,
[generic named constraints](terminology.md#named-constraints) describe how they
may be composed together. Unlike interfaces which are nominal, the name of a
named constraint is not a part of its value. Two different named constraints
with the same definition are equivalent even if they have different names. This
is because types don't explicitly specify which named constraints they
implement, types automatically implement any named constraints they can satisfy.

A named constraint definition can contain interface requirements using `impl`
declarations and names using `alias` declarations. Note that this allows us to
declare the aspects of a type-of-type directly.

```
constraint VectorLegoFish {
  // Interface implementation requirements
  impl as Vector;
  impl as LegoFish;
  // Names
  alias Scale = Vector.Scale;
  alias VAdd = Vector.Add;
  alias LFAdd = LegoFish.Add;
}
```

We don't expect developers to directly define many named constraints, but other
constructs we do expect them to use will be defined in terms of them. For
example, we can define the Carbon builtin `Type` as:

```
constraint Type { }
```

That is, `Type` is the type-of-type with no requirements (so matches every
type), and defines no names.

```
fn Identity[T:! Type](x: T) -> T {
  // Can accept values of any type. But, since we no nothing about the
  // type, we don't know about any operations on `x` inside this function.
  return x;
}

var i: i32 = Identity(3);
var s: String = Identity("string");
```

**Aside:** We can define `auto` as syntactic sugar for `(template _:! Type)`.
This definition allows you to use `auto` as the type for a local variable whose
type can be statically determined by the compiler. It also allows you to use
`auto` as the type of a function parameter, to mean "accepts a value of any
type, and this function will be instantiated separately for every different
type." This is consistent with the
[use of `auto` in the C++20 Abbreviated function template feature](https://en.cppreference.com/w/cpp/language/function_template#Abbreviated_function_template).

In general, the declarations in `constraint` definition match a subset of the
declarations in an `interface`. Named constraints used with generics, as opposed
to templates, should only include required interfaces and aliases to named
members of those interfaces.

To declare a named constraint that includes other declarations for use with
template parameters, use the `template` keyword before `constraint`. Method,
associated type, and associated function requirements may only be declared
inside a `template constraint`. Note that a generic constraint matches all
facets of a type if it matches any, but a template constraint can depend on the
specific names of members used in a particular facet.

There is an analogy between declarations used in a `constraint` and in an
`interface` definition. If an `interface` `I` has (non-`alias`) declarations
`X`, `Y`, and `Z`, like so:

```
interface I {
  X;
  Y;
  Z;
}
```

Then a type implementing `I` would have `impl as I` with definitions for `X`,
`Y`, and `Z`, as in:

```
class ImplementsI {
  // ...
  impl as I {
    X { ... }
    Y { ... }
    Z { ... }
  }
}
```

But the corresponding `constraint` or `template constraint`, `S`:

```
// or template constraint S {
constraint S {
  X;
  Y;
  Z;
}
```

would match any type with definitions for `X`, `Y`, and `Z` directly:

```
class ImplementsS {
  // ...
  X { ... }
  Y { ... }
  Z { ... }
}
```

**TODO:** Move the `template constraint` and `auto` content to the template
design document, once it exists.

### Subtyping between type-of-types

There is a subtyping relationship between type-of-types that allows you to call
one generic function from another as long as you are calling a function with a
subset of your requirements.

Given a generic type `T` with type-of-type `I1`, it may be
[implicitly converted](../expressions/implicit_conversions.md) to a type-of-type
`I2`, resulting in `T as I2`, as long as the requirements of `I1` are a superset
of the requirements of `I2`. Further, given a value `x` of type `T`, it can be
implicitly converted to `T as I2`. For example:

```
interface Printable { fn Print[me: Self](); }
interface Renderable { fn Draw[me: Self](); }

constraint PrintAndRender {
  impl as Printable;
  impl as Renderable;
}
constraint JustPrint {
  impl as Printable;
}

fn PrintIt[T2:! JustPrint](x2: T2) {
  x2.(Printable.Print)();
}
fn PrintDrawPrint[T1:! PrintAndRender](x1: T1) {
  // x1 implements `Printable` and `Renderable`.
  x1.(Printable.Print)();
  x1.(Renderable.Draw)();
  // Can call `PrintIt` since `T1` satisfies `JustPrint` since
  // it implements `Printable` (in addition to `Renderable`).
  // This calls `PrintIt` with `T2 == T1 as JustPrint` and
  // `x2 == x1 as T2`.
  PrintIt(x1);
}
```

## Combining interfaces by anding type-of-types

In order to support functions that require more than one interface to be
implemented, we provide a combination operator on type-of-types, written `&`.
This operator gives the type-of-type with the union of all the requirements and
the union of the names minus any conflicts.

```
interface Printable {
  fn Print[me: Self]();
}
interface Renderable {
  fn Center[me: Self]() -> (i32, i32);
  fn Draw[me: Self]();
}

// `Printable & Renderable` is syntactic sugar for this type-of-type:
constraint {
  impl as Printable;
  impl as Renderable;
  alias Print = Printable.Print;
  alias Center = Renderable.Center;
  alias Draw = Renderable.Draw;
}

fn PrintThenDraw[T:! Printable & Renderable](x: T) {
  // Can use methods of `Printable` or `Renderable` on `x` here.
  x.Print();  // Same as `x.(Printable.Print)();`.
  x.Draw();  // Same as `x.(Renderable.Draw)();`.
}

class Sprite {
  // ...
  impl as Printable {
    fn Print[me: Self]() { ... }
  }
  impl as Renderable {
    fn Center[me: Self]() -> (i32, i32) { ... }
    fn Draw[me: Self]() { ... }
  }
}

var s: Sprite = ...;
PrintThenDraw(s);
```

Any conflicting names between the two types are replaced with a name that is an
error to use.

```
interface Renderable {
  fn Center[me: Self]() -> (i32, i32);
  fn Draw[me: Self]();
}
interface EndOfGame {
  fn Draw[me: Self]();
  fn Winner[me: Self](player: i32);
}
// `Renderable & EndOfGame` is syntactic sugar for this type-of-type:
constraint {
  impl as Renderable;
  impl as EndOfGame;
  alias Center = Renderable.Center;
  // Open question: `forbidden`, `invalid`, or something else?
  forbidden Draw
    message "Ambiguous, use either `(Renderable.Draw)` or `(EndOfGame.Draw)`.";
  alias Winner = EndOfGame.Winner;
}
```

Conflicts can be resolved at the call site using
[the qualified name syntax](#qualified-member-names), or by defining a named
constraint explicitly and renaming the methods:

```
constraint RenderableAndEndOfGame {
  impl as Renderable;
  impl as EndOfGame;
  alias Center = Renderable.Center;
  alias RenderableDraw = Renderable.Draw;
  alias TieGame = EndOfGame.Draw;
  alias Winner = EndOfGame.Winner;
}

fn RenderTieGame[T:! RenderableAndEndOfGame](x: T) {
  // Calls Renderable.Draw()
  x.RenderableDraw();
  // Calls EndOfGame.Draw()
  x.TieGame();
}
```

Reserving the name when there is a conflict is part of resolving what happens
when you combine more than two type-of-types. If `x` is forbidden in `A`, it is
forbidden in `A & B`, whether or not `B` defines the name `x`. This makes `&`
associative and commutative, and so it is well defined on sets of interfaces, or
other type-of-types, independent of order.

Note that we do _not_ consider two type-of-types using the same name to mean the
same thing to be a conflict. For example, combining a type-of-type with itself
gives itself, `MyTypeOfType & MyTypeOfType == MyTypeOfType`. Also, given two
[interface extensions](#interface-extension) of a common base interface, the sum
should not conflict on any names in the common base.

**Rejected alternative:** Instead of using `&` as the combining operator, we
considered using `+`,
[like Rust](https://rust-lang.github.io/rfcs/0087-trait-bounds-with-plus.html).
See [#531](https://github.com/carbon-language/carbon-lang/issues/531) for the
discussion.

**Future work:** We may want to define another operator on type-of-types for
adding requirements to a type-of-type without affecting the names, and so avoid
the possibility of name conflicts. Note this means the operation is not
commutative. If we call this operator `[&]`, then `A [&] B` has the names of `A`
and `B [&] A` has the names of `B`.

```
// `Printable [&] Renderable` is syntactic sugar for this type-of-type:
constraint {
  impl as Printable;
  impl as Renderable;
  alias Print = Printable.Print;
}

// `Renderable [&] EndOfGame` is syntactic sugar for this type-of-type:
constraint {
  impl as Renderable;
  impl as EndOfGame;
  alias Center = Renderable.Center;
  alias Draw = Renderable.Draw;
}
```

Note that all three expressions `A & B`, `A [&] B`, and `B [&] A` have the same
requirements, and so you would be able to switch a function declaration between
them without affecting callers.

Nothing in this design depends on the `[&]` operator, and having both `&` and
`[&]` might be confusing for users, so it makes sense to postpone implementing
`[&]` until we have a demonstrated need. The `[&]` operator seems most useful
for adding requirements for interfaces used for
[operator overloading](#operator-overloading), where merely implementing the
interface is enough to be able to use the operator to access the functionality.

**Alternatives considered:** See
[Carbon: Access to interface methods](https://docs.google.com/document/d/1u_i_s31OMI_apPur7WmVxcYq6MUXsG3oCiKwH893GRI/edit?usp=sharing&resourcekey=0-0lzSNebBMtUBi4lStL825g).

**Comparison with other languages:** This `&` operation on interfaces works very
similarly to Rust's `+` operation, with the main difference being how you
[qualify names when there is a conflict](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html).

## Interface requiring other interfaces

Some interfaces will depend on other interfaces being implemented for the same
type. For example, in C++,
[the `Container` concept](https://en.cppreference.com/w/cpp/named_req/Container#Other_requirements)
requires all containers to also satisfy the requirements of
`DefaultConstructible`, `CopyConstructible`, `EqualityComparable`, and
`Swappable`. This is already a capability for
[type-of-types in general](#type-of-types-and-facet-types). For consistency we
will use the same semantics and syntax as we do for
[named constraints](#named-constraints):

```
interface Equatable { fn Equals[me: Self](that: Self) -> bool; }

interface Iterable {
  fn Advance[addr me: Self*]() -> bool;
  impl as Equatable;
}

def DoAdvanceAndEquals[T:! Iterable](x: T) {
  // `x` has type `T` that implements `Iterable`, and so has `Advance`.
  x.Advance();
  // `Iterable` requires an implementation of `Equatable`,
  // so `T` also implements `Equatable`.
  x.(Equatable.Equals)(x);
}

class Iota {
  impl as Iterable { fn Advance[me: Self]() { ... } }
  impl as Equatable { fn Equals[me: Self](that: Self) -> bool { ... } }
}
var x: Iota;
DoAdvanceAndEquals(x);
```

Like with named constraints, an interface implementation requirement doesn't by
itself add any names to the interface, but again those can be added with `alias`
declarations:

```
interface Hashable {
  fn Hash[me: Self]() -> u64;
  impl as Equatable;
  alias Equals = Equatable.Equals;
}

def DoHashAndEquals[T:! Hashable](x: T) {
  // Now both `Hash` and `Equals` are available directly:
  x.Hash();
  x.Equals(x);
}
```

**Comparison with other languages:**
[This feature is called "Supertraits" in Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait).

### Interface extension

When implementing an interface, we should allow implementing the aliased names
as well. In the case of `Hashable` above, this includes all the members of
`Equatable`, obviating the need to implement `Equatable` itself:

```
class Song {
  impl as Hashable {
    fn Hash[me: Self]() -> u64 { ... }
    fn Equals[me: Self](that: Self) -> bool { ... }
  }
}
var y: Song;
DoHashAndEquals(y);
```

This allows us to say that `Hashable`
["extends"](terminology.md#extending-an-interface) `Equatable`, with some
benefits:

-   This allows `Equatable` to be an implementation detail of `Hashable`.
-   This allows types implementing `Hashable` to implement all of its API in one
    place.
-   This reduces the boilerplate for types implementing `Hashable`.

We expect this concept to be common enough to warrant dedicated syntax:

```
interface Equatable { fn Equals[me: Self](that: Self) -> bool; }

interface Hashable {
  extends Equatable;
  fn Hash[me: Self]() -> u64;
}
// is equivalent to the definition of Hashable from before:
// interface Hashable {
//   impl as Equatable;
//   alias Equals = Equatable.Equals;
//   fn Hash[me: Self]() -> u64;
// }
```

No names in `Hashable` are allowed to conflict with names in `Equatable` (unless
those names are marked as `upcoming` or `deprecated` as in
[evolution future work](#evolution)). Hopefully this won't be a problem in
practice, since interface extension is a very closely coupled relationship, but
this may be something we will have to revisit in the future.

Examples:

-   The C++
    [Boost.Graph library](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/)
    [graph concepts](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/graph_concepts.html#fig:graph-concepts)
    has many refining relationships between concepts.
    [Carbon generics use case: graph library](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?usp=sharing&resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA)
    shows how those concepts might be translated into Carbon interfaces.
-   The [C++ concepts](https://en.cppreference.com/w/cpp/named_req) for
    containers, iterators, and concurrency include many requirement
    relationships.
-   Swift protocols, such as
    [Collection](https://developer.apple.com/documentation/swift/collection).

To write an interface extending multiple interfaces, use multiple `extends`
declarations. For example, the
[`BinaryInteger` protocol in Swift](https://developer.apple.com/documentation/swift/binaryinteger)
inherits from `CustomStringConvertible`, `Hashable`, `Numeric`, and `Stridable`.
The [`SetAlgeba` protocol](https://swiftdoc.org/v5.1/protocol/setalgebra/)
extends `Equatable` and `ExpressibleByArrayLiteral`, which would be declared in
Carbon:

```
interface SetAlgebra {
  extends Equatable;
  extends ExpressibleByArrayLiteral;
}
```

**Alternative considered:** The `extends` declarations are in the body of the
`interface` definition instead of the header so we can use
[associated types (defined below)](#associated-types) also defined in the body
in parameters or constraints of the interface being extended.

```
// A type can implement `ConvertibleTo` many times, using
// different values of `T`.
interface ConvertibleTo(T:! Type) { ... }

// A type can only implement `PreferredConversion` once.
interface PreferredConversion {
  let AssociatedType:! Type;
  extends ConvertibleTo(AssociatedType);
}
```

#### `extends` and `impl` with named constraints

The `extends` declaration makes sense with the same meaning inside a
[`constraint`](#named-constraints) definition, and so is also supported.

```
interface Media {
  fn Play[me: Self]();
}
interface Job {
  fn Run[me: Self]();
}

constraint Combined {
  extends Media;
  extends Job;
}
```

This definition of `Combined` is equivalent to requiring both the `Media` and
`Job` interfaces being implemented, and aliases their methods.

```
// Equivalent
constraint Combined {
  impl as Media;
  alias Play = Media.Play;
  impl as Job;
  alias Run = Job.Run;
}
```

Notice how `Combined` has aliases for all the methods in the interfaces it
requires. That condition is sufficient to allow a type to `impl` the named
constraint:

```
class Song {
  impl as Combined {
    fn Play[me: Self]() { ... }
    fn Run[me: Self]() { ... }
  }
}
```

This is equivalent to implementing the required interfaces directly:

```
class Song {
  impl as Media {
    fn Play[me: Self]() { ... }
  }
  impl as Job {
    fn Run[me: Self]() { ... }
  }
}
```

This is just like you get an implementation of `Equatable` by implementing
`Hashable` when `Hashable` extends `Equatable`. This provides a tool useful for
[evolution](#evolution).

Conversely, an `interface` can extend a `constraint`:

```
interface MovieCodec {
  extends Combined;

  fn Load[addr me: Self*](filename: String);
}
```

This gives `MovieCodec` the same requirements and names as `Combined`, and so is
equivalent to:

```
interface MovieCodec {
  impl as Media;
  alias Play = Media.Play;
  impl as Job;
  alias Run = Job.Run;

  fn Load[addr me: Self*](filename: String);
}
```

#### Diamond dependency issue

Consider this set of interfaces, simplified from
[this example generic graph library doc](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA#):

```
interface Graph {
  fn Source[addr me: Self*](e: EdgeDescriptor) -> VertexDescriptor;
  fn Target[addr me: Self*](e: EdgeDescriptor) -> VertexDescriptor;
}

interface IncidenceGraph {
  extends Graph;
  fn OutEdges[addr me: Self*](u: VertexDescriptor)
    -> (EdgeIterator, EdgeIterator);
}

interface EdgeListGraph {
  extends Graph;
  fn Edges[addr me: Self*]() -> (EdgeIterator, EdgeIterator);
}
```

We need to specify what happens when a graph type implements both
`IncidenceGraph` and `EdgeListGraph`, since both interfaces extend the `Graph`
interface.

```
class MyEdgeListIncidenceGraph {
  impl as IncidenceGraph { ... }
  impl as EdgeListGraph { ... }
}
```

The rule is that we need one definition of each method of `Graph`. Each method
though could be defined in the `impl` block of `IncidenceGraph`,
`EdgeListGraph`, or `Graph`. These would all be valid:

-   `IncidenceGraph` implements all methods of `Graph`, `EdgeListGraph`
    implements none of them.

    ```
    class MyEdgeListIncidenceGraph {
      impl as IncidenceGraph {
        fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn OutEdges[addr me: Self*](u: VertexDescriptor)
            -> (EdgeIterator, EdgeIterator) { ... }
      }
      impl as EdgeListGraph {
        fn Edges[addr me: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
      }
    }
    ```

-   `IncidenceGraph` and `EdgeListGraph` implement all methods of `Graph`
    between them, but with no overlap.

    ```
    class MyEdgeListIncidenceGraph {
      impl as IncidenceGraph {
        fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn OutEdges[addr me: Self*](u: VertexDescriptor)
            -> (EdgeIterator, EdgeIterator) { ... }
      }
      impl as EdgeListGraph {
        fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Edges[addr me: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
      }
    }
    ```

-   Explicitly implementing `Graph`.

    ```
    class MyEdgeListIncidenceGraph {
      impl as Graph {
        fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
      }
      impl as IncidenceGraph { ... }
      impl as EdgeListGraph { ... }
    }
    ```

-   Implementing `Graph` externally.

    ```
    class MyEdgeListIncidenceGraph {
      impl as IncidenceGraph { ... }
      impl as EdgeListGraph { ... }
    }
    external impl as Graph {
      fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
      fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    }
    ```

This last point means that there are situations where we can only detect a
missing method definition by the end of the file. This doesn't delay other
aspects of semantic checking, which will just assume that these methods will
eventually be provided.

### Use case: overload resolution

Implementing an extended interface is an example of a more specific match for
[lookup resolution](#lookup-resolution-and-specialization). For example, this
could be used to provide different implementations of an algorithm depending on
the capabilities of the iterator being passed in:

```
interface ForwardIntIterator {
  fn Advance[addr me: Self*]();
  fn Get[me: Self]() -> i32;
}
interface BidirectionalIntIterator {
  extends ForwardIntIterator;
  fn Back[addr me: Self*]();
}
interface RandomAccessIntIterator {
  extends BidirectionalIntIterator;
  fn Skip[addr me: Self*](offset: i32);
  fn Difference[me: Self](that: Self) -> i32;
}

fn SearchInSortedList[IterT:! ForwardIntIterator]
    (begin: IterT, end: IterT, needle: i32) -> bool {
  ... // does linear search
}
// Will prefer the following overload when it matches
// since it is more specific.
fn SearchInSortedList[IterT:! RandomAccessIntIterator]
    (begin: IterT, end: IterT, needle: i32) -> bool {
  ... // does binary search
}
```

This would be an example of the more general rule that an interface `A`
requiring an implementation of interface `B` means `A` is more specific than
`B`.

## Type compatibility

None of the conversions between facet types change the implementation of any
interfaces for a type. So the result of a conversion does not depend on the
sequence of conversions you perform, just the original type and the final
type-of-type. That is, these types will all be equal:

-   `T as I`
-   `(T as A) as I`
-   `(((T as A) as B) as C) as I`

Now consider a type with a generic type parameter, like a hash map type:

```
interface Hashable { ... }
class HashMap(KeyT:! Hashable, ValueT:! Type) {
  fn Find[me:Self](key: KeyT) -> Optional(ValueT);
  // ...
}
```

A user of this type will provide specific values for the key and value types:

```
var hm: HashMap(String, i32) = ...;
var result: Optional(i32) = hm.Find("Needle");
```

Since the `Find` function is generic, it can only use the capabilities that
`HashMap` requires of `KeyT` and `ValueT`. This implies that the
_implementation_ of `HashMap(String, i32).Find` and
`HashMap(String as Hashable, i32).Find` are the same. In fact, we could
substitute any facet of `String`, and `Find` would still use
`String as Hashable` in its implementation. So these types:

-   `HashMap(String, i32)`
-   `HashMap(String as Hashable, i32 as Type)`
-   `HashMap(String as Printable, i32)`
-   `HashMap((String as Printable & Hashable) as Hashable, i32)`

are also facets of each other, and Carbon can freely allow casts and implicit
conversions between them.

This means we don't generally need to worry about getting the wrong facet type
as the argument for a generic type. This means we don't get type mismatches when
calling functions as in this example, where the type parameters have different
constraints than the type requires:

```
fn PrintValue
    [KeyT:! Printable & Hashable, ValueT:! Printable]
    (map: HashMap(KeyT, ValueT), key: KeyT) { ... }

var m: HashMap(String, i32) = ...;
PrintValue(m, "key");
```

However, those types are still different. A caller of `Find` observes that its
signature reflects the actual type parameters passed to `HashMap`, not their
projection onto the `Hashable` or `Type` facets. In particular, the return type
of `hm.Find` is `Optional(i32)`, not `Optional(i32 as Type)`. (Incidentally,
`Optional(i32)` and `Optional(i32 as Type)` are also facets of each other.)

## Adapting types

Since interfaces may only be implemented for a type once, and we limit where
implementations may be added to a type, there is a need to allow the user to
switch the type of a value to access different interface implementations. We
therefore provide a way to create new types
[compatible with](terminology.md#compatible-types) existing types with different
APIs, in particular with different interface implementations, by
[adapting](terminology.md#adapting-a-type) them:

```
interface Printable {
  fn Print[me: Self]();
}
interface Comparable {
  fn Less[me: Self](that: Self) -> bool;
}
class Song {
  impl as Printable { fn Print[me: Self]() { ... } }
}
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> bool { ... }
  }
}
adapter FormattedSong for Song {
  impl as Printable { fn Print[me: Self]() { ... } }
}
adapter FormattedSongByTitle for Song {
  impl as Printable = FormattedSong as Printable;
  impl as Comparable = SongByTitle as Comparable;
}
```

This allows us to provide implementations of new interfaces (as in
`SongByTitle`), provide different implementations of the same interface (as in
`FormattedSong`), or mix and match implementations from other compatible types
(as in `FormattedSongByTitle`). The rules are:

-   You can add any declaration that you could add to a class except for
    declarations that would change the representation of the type. This means
    you can add functions, interface implementations, and aliases, but not
    fields, base classes, or virtual functions.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `Song`, `SongByTitle`, `FormattedSong`,
    and `FormattedSongByTitle` end up compatible with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit conversion between
    these types (unlike between a type and one of its facet types / impls).
-   For the purposes of generics, we only need to support adding interface
    implementations. But this `adapter` feature could be used more generally,
    such as to add methods.

Inside an adapter, the `Self` type matches the adapter. Members of the original
type may be accessed like any other facet type; either by a cast:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> bool {
      return (this as Song).Title() < (that as Song).Title();
    }
  }
}
```

or using qualified names:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> bool {
      return this.(Song.Title)() < that.(Song.Title)();
    }
  }
}
```

**Open question:** As an alternative to:

```
impl as Printable = FormattedSong as Printable;
```

we could allow users to write:

```
impl as Printable = FormattedSong;
```

This would remove ceremony that the compiler doesn't need. The concern is
whether it makes sense or is a category error. In this example, is
`FormattedSong`, a type, a suitable value to provide when asking for a
`Printable` implementation? An argument for this terser syntax is that the
implicit conversion is legal in other contexts:

```
// ✅ Legal implicit conversion
var v:! Printable = FormattedSong;
```

**Comparison with other languages:** This matches the Rust idiom called
"newtype", which is used to implement traits on types while avoiding coherence
problems, see
[here](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-the-newtype-pattern-to-implement-external-traits-on-external-types)
and
[here](https://github.com/Ixrec/rust-orphan-rules#user-content-why-are-the-orphan-rules-controversial).
Rust's mechanism doesn't directly support reusing implementations, though some
of that is provided by macros defined in libraries. Haskell has a
[`newtype` feature](https://wiki.haskell.org/Newtype) as well. Haskell's feature
doesn't directly support reusing implementations either, but the most popular
compiler provides it as
[an extension](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/newtype_deriving.html).

### Adapter compatibility

The framework from the [type compatibility section](#type-compatibility) allows
us to evaluate when we can convert between two different arguments to a
parameterized type. Consider three compatible types, all of which implement
`Hashable`:

```
class Song {
  impl as Hashable { ... }
  impl as Printable { ... }
}
adapter SongHashedByTitle for Song {
  impl as Hashable { ... }
}
adapter PlayableSong for Song {
  impl as Hashable = Song as Hashable;
  impl as Media { ... }
}
```

Observe that `Song as Hashable` is different from
`SongHashedByTitle as Hashable`, since they have different definitions of the
`Hashable` interface even though they are compatible types. However
`Song as Hashable` and `PlayableSong as Hashable` are almost the same. In
addition to using the same data representation, they both implement one
interface, `Hashable`, and use the same implementation for that interface. The
one difference between them is that `Song as Hashable` may be implicitly
converted to `Song`, which implements interface `Printable`, and
`PlayableSong as Hashable` may be implicilty converted to `PlayableSong`, which
implements interface `Media`. This means that it is safe to convert between
`HashMap(Song, i32)` and `HashMap(PlayableSong, i32)` (though maybe only with an
explicit cast), since the implementation of all the methods will use the same
implementation of the `Hashable` interface. But
`HashMap(SongHashedByTitle, i32)` is incompatible. This is a relief, because we
know that in practice the invariants of a `HashMap` implementation rely on the
hashing function staying the same.

### Extending adapter

Frequently we expect that the adapter type will want to preserve most or all of
the API of the original type. The two most common cases expected are adding and
replacing an interface implementation. Users would indicate that an adapter
starts from the original type's existing API by using the `extends` keyword
instead of `for`:

```
class Song {
  impl as Hashable { ... }
  impl as Printable { ... }
}

adapter SongByArtist extends Song {
  // Add an implementation of a new interface
  impl as Comparable { ... }

  // Replace an existing implementation of an interface
  // with an alternative.
  impl as Hashable { ... }
}
```

The resulting type `SongByArtist` would:

-   implement `Comparable`, unlike `Song`,
-   implement `Hashable`, but differently than `Song`, and
-   implement `Printable`, inherited from `Song`.

Unlike the similar `class B extends A` notation, `adaptor B extends A` is
permitted even if `A` is a final class. Also, there is no implicit conversion
from `B` to `A`, matching `adapter`...`for` but unlike class extension.

To avoid or resolve name conflicts between interfaces, an `impl` may be declared
[`external`](#external-impl). The names in that interface may then be pulled in
individually or renamed using `alias` declarations.

```
adapter SongRenderToPrintDriver extends Song {
  // Add a new `Print()` member function.
  fn Print[me: Self]() { ... }

  // Avoid name conflict with new `Print` function by making
  // the implementation of the `Printable` interface external.
  external impl as Printable = Song as Printable;

  // Make the `Print` function from `Printable` available
  // under the name `PrintToScreen`.
  alias PrintToScreen = Printable.Print;
}
```

### Use case: Using independent libraries together

Imagine we have two packages that are developed independently. Package
`CompareLib` defines an interface `CompareLib.Comparable` and a generic
algorithm `CompareLib.Sort` that operates on types that implement
`CompareLib.Comparable`. Package `SongLib` defines a type `SongLib.Song`.
Neither has a dependency on the other, so neither package defines an
implementation for `CompareLib.Comparable` for type `SongLib.Song`. A user that
wants to pass a value of type `SongLib.Song` to `CompareLib.Sort` has to define
an adapter that provides an implementation of `CompareLib.Comparable` for
`SongLib.Song`. This adapter will probably use the
[`extends` facility of adapters](#extending-adapter) to preserve the
`SongLib.Song` API.

```
import CompareLib;
import SongLib;

adapter Song extends SongLib.Song {
  impl as CompareLib.Comparable { ... }
}
// Or, to keep the names from CompareLib.Comparable out of Song's API:
adapter Song extends SongLib.Song { }
external impl Song as CompareLib.Comparable { ... }
```

The caller can either convert `SongLib.Song` values to `Song` when calling
`CompareLib.Sort` or just start with `Song` values in the first place.

```
var lib_song: SongLib.Song = ...;
CompareLib.Sort((lib_song as Song,));

var song: Song = ...;
CompareLib.Sort((song,));
```

### Adapter with stricter invariants

**Future work:** Rust also uses the newtype idiom to create types with
additional invariants or other information encoded in the type
([1](https://doc.rust-lang.org/rust-by-example/generics/new_types.html),
[2](https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction),
[3](https://www.worthe-it.co.za/blog/2020-10-31-newtype-pattern-in-rust.html)).
This is used to record in the type system that some data has passed validation
checks, like `ValidDate` with the same data layout as `Date`. Or to record the
units associated with a value, such as `Seconds` versus `Milliseconds` or `Feet`
versus `Meters`. We should have some way of restricting the casts between a type
and an adapter to address this use case.

### Application: Defining an impl for use by other types

Let's say we want to provide a possible implementation of an interface for use
by types for which that implementation would be appropriate. We can do that by
defining an adapter implementing the interface that is parameterized on the type
it is adapting. That impl may then be pulled in using the `impl as ... = ...;`
syntax.

```
interface Comparable {
  fn Less[me: Self](that: Self) -> bool;
}
adapter ComparableFromDifferenceFn
    (T:! Type, Difference:! fnty(T, T)->i32) for T {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> bool {
      return Difference(this, that) < 0;
    }
  }
}
class IntWrapper {
  var x: i32;
  fn Difference(this: Self, that: Self) {
    return that.x - this.x;
  }
  impl as Comparable =
      ComparableFromDifferenceFn(IntWrapper, Difference)
      as Comparable;
}
```

## Associated constants

In addition to associated methods, we allow other kinds of
[associated entities](terminology.md#associated-entity). For consistency, we use
the same syntax to describe a constant in an interface as in a type without
assigning a value. As constants, they are declared using the `let` introducer.
For example, a fixed-dimensional point type could have the dimension as an
associated constant.

```
interface NSpacePoint {
  let N:! i32;
  // The following require: 0 <= i < N.
  fn Get[addr me: Self*](i: i32) -> f64;
  fn Set[addr me: Self*](i: i32, value: f64);
  // Associated constants may be used in signatures:
  fn SetAll[addr me: Self*](value: Array(f64, N));
}
```

Implementations of `NSpacePoint` for different types might have different values
for `N`:

```
class Point2D {
  impl as NSpacePoint {
    let N:! i32 = 2;
    fn Get[addr me: Self*](i: i32) -> f64 { ... }
    fn Set[addr me: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr me: Self*](value: Array(f64, 2)) { ... }
  }
}

class Point3D {
  impl as NSpacePoint {
    let N:! i32 = 3;
    fn Get[addr me: Self*](i: i32) -> f64 { ... }
    fn Set[addr me: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr me: Self*](value: Array(f64, 3)) { ... }
  }
}
```

And these values may be accessed as members of the type:

```
Assert(Point2D.N == 2);
Assert(Point3D.N == 3);

fn PrintPoint[PointT:! NSpacePoint](p: PointT) {
  for (var i: i32 = 0; i < PointT.N; ++i) {
    if (i > 0) { Print(", "); }
    Print(p.Get(i));
  }
}

fn ExtractPoint[PointT:! NSpacePoint](
    p: PointT,
    dest: Array(f64, PointT.N)*) {
  for (var i: i32 = 0; i < PointT.N; ++i) {
    (*dest)[i] = p.Get(i);
  }
}
```

**Comparison with other languages:** This feature is also called
[associated constants in Rust](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants).

**Aside:** In general, the use of `:!` here means these `let` declarations will
only have compile-time and not runtime storage associated with them.

### Associated class functions

To be consistent with normal
[class function](/docs/design/classes.md#class-functions) declaration syntax,
associated class functions are written:

```
interface DeserializeFromString {
  fn Deserialize(serialized: String) -> Self;
}

class MySerializableType {
  var i: i32;

  impl as DeserializeFromString {
    fn Deserialize(serialized: String) -> Self {
      return (.i = StringToInt(serialized));
    }
  }
}

var x: MySerializableType = MySerializableType.Deserialize("3");

fn Deserialize(T:! DeserializeFromString, serialized: String) -> T {
  return T.Deserialize(serialized);
}
var y: MySerializableType = Deserialize(MySerializableType, "4");
```

This is instead of declaring an associated constant using `let` with a function
type.

Together associated methods and associated class functions are called
_associated functions_, much like together methods and class functions are
called [member functions](/docs/design/classes.md#member-functions).

## Associated types

Associated types are [associated entities](terminology.md#associated-entity)
that happen to be types. These are particularly interesting since they can be
used in the signatures of associated methods or functions, to allow the
signatures of methods to vary from implementation to implementation. We already
have one example of this: the `Self` type discussed
[above in the "Interfaces" section](#interfaces). For other cases, we can say
that the interface declares that each implementation will provide a type under a
specific name. For example:

```
interface StackAssociatedType {
  let ElementType:! Type;
  fn Push[addr me: Self*](value: ElementType);
  fn Pop[addr me: Self*]() -> ElementType;
  fn IsEmpty[addr me: Self*]() -> bool;
}
```

Here we have an interface called `StackAssociatedType` which defines two
methods, `Push` and `Pop`. The signatures of those two methods declare them as
accepting or returning values with the type `ElementType`, which any implementer
of `StackAssociatedType` must also define. For example, maybe `DynamicArray`
implements `StackAssociatedType`:

```
class DynamicArray(T:! Type) {
  class IteratorType { ... }
  fn Begin[addr me: Self*]() -> IteratorType;
  fn End[addr me: Self*]() -> IteratorType;
  fn Insert[addr me: Self*](pos: IteratorType, value: T);
  fn Remove[addr me: Self*](pos: IteratorType);

  impl as StackAssociatedType {
    // Set the associated type `ElementType` to `T`.
    let ElementType:! Type = T;
    fn Push[addr me: Self*](value: ElementType) {
      this->Insert(this->End(), value);
    }
    fn Pop[addr me: Self*]() -> ElementType {
      var pos: IteratorType = this->End();
      Assert(pos != this->Begin());
      --pos;
      returned var ret: ElementType = *pos;
      this->Remove(pos);
      return var;
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return this->Begin() == this->End();
    }
  }
}
```

**Alternatives considered:** See
[other syntax options considered for specifying associated types](/proposals/p0731.md#syntax-for-associated-constants).
In particular, it was deemed that
[Swift's approach of inferring the associated type from method signatures in the impl](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID190)
was unneeded complexity.

The definition of the `StackAssociatedType` is sufficient for writing a generic
function that operates on anything implementing that interface, for example:

```
fn PeekAtTopOfStack[StackType:! StackAssociatedType](s: StackType*)
    -> StackType.ElementType {
  var top: StackType.ElementType = s->Pop();
  s->Push(top);
  return top;
}

var my_array: DynamicArray(i32) = (1, 2, 3);
// PeekAtTopOfStack's `StackType` is set to
// `DynamicArray(i32) as StackAssociatedType`.
// `StackType.ElementType` becomes `i32`.
Assert(PeekAtTopOfStack(my_array) == 3);
```

Associated types can also be implemented using a
[member type](/docs/design/classes.md#member-type).

```
interface Container {
  let IteratorType:! Iterator;
  ...
}

class DynamicArray(T:! Type) {
  ...
  impl as Container {
    class IteratorType { ... }
    ...
  }
}
```

For context, see
["Interface type parameters and associated types" in the generics terminology document](terminology.md#interface-type-parameters-versus-associated-types).

**Comparison with other languages:** Both
[Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189)
support associated types.

### Model

The associated type can be modeled by a witness table field in the interface's
witness table.

```
interface Iterator {
  fn Advance[addr me: Self*]();
}

interface Container {
  let IteratorType:! Iterator;
  fn Begin[addr me: Self*]() -> IteratorType;
}
```

is represented by:

```
class Iterator(Self:! Type) {
  var Advance: fnty(this: Self*);
  ...
}
class Container(Self:! Type) {
  // Representation type for the iterator.
  let IteratorType:! Type;
  // Witness that IteratorType implements Iterator.
  var iterator_impl: Iterator(IteratorType)*;

  // Method
  var Begin: fnty (this: Self*) -> IteratorType;
  ...
}
```

## Parameterized interfaces

Associated types don't change the fact that a type can only implement an
interface at most once.

If instead you want a family of related interfaces, one per possible value of a
type parameter, multiple of which could be implemented for a single type, you
would use parameterized interfaces. To write a parameterized version the stack
interface instead of using associated types, write a parameter list after the
name of the interface instead of the associated type declaration:

```
interface StackParameterized(ElementType:! Type) {
  fn Push[addr me: Self*](value: ElementType);
  fn Pop[addr me: Self*]() -> ElementType;
  fn IsEmpty[addr me: Self*]() -> bool;
}
```

Then `StackParameterized(Fruit)` and `StackParameterized(Veggie)` would be
considered different interfaces, with distinct implementations.

```
class Produce {
  var fruit: DynamicArray(Fruit);
  var veggie: DynamicArray(Veggie);
  impl as StackParameterized(Fruit) {
    fn Push[addr me: Self*](value: Fruit) {
      this->fruit.Push(value);
    }
    fn Pop[addr me: Self*]() -> Fruit {
      return this->fruit.Pop();
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return this->fruit.IsEmpty();
    }
  }
  impl as StackParameterized(Veggie) {
    fn Push[addr me: Self*](value: Veggie) {
      this->veggie.Push(value);
    }
    fn Pop[addr me: Self*]() -> Veggie {
      return this->veggie.Pop();
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return this->veggie.IsEmpty();
    }
  }
}
```

Unlike associated types in interfaces and parameters to types, interface
parameters can't be deduced. For example, if we were to rewrite
[the `PeekAtTopOfStack` example in the "associated types" section](#associated-types)
for `StackParameterized(T)` it would generate a compile error:

```
// ❌ Error: can't deduce interface parameter `T`.
fn BrokenPeekAtTopOfStackParameterized
    [T:! Type, StackType:! StackParameterized(T)]
    (s: StackType*) -> T { ... }
```

This error is because the compiler can not determine if `T` should be `Fruit` or
`Veggie` when passing in argument of type `Produce*`. The function's signature
would have to be changed so that the value for `T` could be determined from the
explicit parameters.

```
fn PeekAtTopOfStackParameterized
    [T:! Type, StackType:! StackParameterized(T)]
    (s: StackType*, _: singleton_type_of(T)) -> T { ... }

var produce: Produce = ...;
var top_fruit: Fruit =
    PeekAtTopOfStackParameterized(&produce, Fruit);
var top_veggie: Veggie =
    PeekAtTopOfStackParameterized(&produce, Veggie);
```

The pattern `_: singleton_type_of(T)` is a placeholder syntax for an expression
that will only match `T`, until issue
[#578: Value patterns as function parameters](https://github.com/carbon-language/carbon-lang/issues/578)
is resolved. Using that pattern in the explicit parameter list allows us to make
`T` available earlier in the declaration so it can be passed as the argument to
the parameterized interface `StackParameterized`.

This approach is useful for the `ComparableTo(T)` interface, where a type might
be comparable with multiple other types, and in fact interfaces for
[operator overloads](#operator-overloading) more generally. Example:

```
interface EquatableWith(T:! Type) {
  fn Equals[me: Self](that: T) -> bool;
  ...
}
class Complex {
  var real: f64;
  var imag: f64;
  // Can implement this interface more than once as long as it has different
  // arguments.
  impl as EquatableWith(Complex) { ... }
  impl as EquatableWith(f64) { ... }
}
```

All interface parameters must be marked as "generic", using the `:!` syntax.
This reflects these two properties of these parameters:

-   They must be resolved at compile-time, and so can't be passed regular
    dynamic values.
-   We allow either generic or template values to be passed in.

**Context:** See
[interface type parameters](terminology.md#interface-type-parameters-versus-associated-types)
in the terminology doc.

**Note:** Interface parameters aren't required to be types, but that is the vast
majority of cases. As an example, if we had an interface that allowed a type to
define how the tuple-member-read operator would work, the index of the member
could be an interface parameter:

```
interface ReadTupleMember(index:! u32) {
  let T:! Type;
  // Returns me[index]
  fn Get[me: Self]() -> T;
}
```

This requires that the index be known at compile time, but allows different
indices to be associated with different types.

**Caveat:** When implementing an interface twice for a type, you need to be sure
that the interface parameters will always be different. For example:

```
interface Map(FromType:! Type, ToType:! Type) {
  fn Map[addr me: Self*](needle: FromType) -> Optional(ToType);
}
class Bijection(FromType:! Type, ToType:! Type) {
  impl as Map(FromType, ToType) { ... }
  impl as Map(ToType, FromType) { ... }
}
// ❌ Error: Bijection has two impls of interface Map(String, String)
var oops: Bijection(String, String) = ...;
```

In this case, it would be better to have an [adapting type](#adapting-types) to
contain the `impl` for the reverse map lookup, instead of implementing the `Map`
interface twice:

```
class Bijection(FromType:! Type, ToType:! Type) {
  impl as Map(FromType, ToType) { ... }
}
adapter ReverseLookup(FromType:! Type, ToType:! Type)
    for Bijection(FromType, ToType) {
  impl as Map(ToType, FromType) { ... }
}
```

**Comparison with other languages:** Rust calls
[traits with type parameters "generic traits"](https://doc.rust-lang.org/reference/items/traits.html#generic-traits)
and
[uses them for operator overloading](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading).
Note that Rust further supports defaults for those type parameters (such as
`Self`).

[Rust uses the term "type parameters"](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#clearer-trait-matching)
for both interface type parameters and associated types. The difference is that
interface parameters are "inputs" since they _determine_ which `impl` to use,
and associated types are "outputs" since they are determined _by_ the `impl`,
but play no role in selecting the `impl`.

### Impl lookup

Let's say you have some interface `I(T, U(V))` being implemented for some type
`A(B(C(D), E))`. To satisfy the orphan rule for coherence, that `impl` must be
defined in some library that must be imported in any code that looks up whether
that interface is implemented for that type. This requires that `impl` is
defined in the same library that defines the interface or one of the names
needed by the type. That is, the `impl` must be defined with one of `I`, `T`,
`U`, `V`, `A`, `B`, `C`, `D`, or `E`. We further require anything looking up
this `impl` to import the _definitions_ of all of those names. Seeing a forward
declaration of these names is insufficient, since you can presumably see forward
declarations without seeing an `impl` with the definition. This accomplishes a
few goals:

-   The compiler can check that there is only one definition of any `impl` that
    is actually used, avoiding
    [One Definition Rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule)
    problems.
-   Every attempt to use an `impl` will see the exact same `impl`, making the
    interpretation and semantics of code consistent no matter its context, in
    accordance with the
    [low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).
-   Allowing the `impl` to be defined with either the interface or the type
    addresses the
    [expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

Note that [the rules for specialization](#lookup-resolution-and-specialization)
do allow there to be more than one `impl` to be defined for a type, as long as
one can unambiguously be picked as most specific.

**References:** Implementation coherence is
[defined in terminology](terminology.md#coherence), and is
[a goal for Carbon](goals.md#coherence). More detail can be found in
[this appendix with the rationale and alternatives considered](appendix-coherence.md).

### Parameterized named constraints

We should also allow the [named constraint](#named-constraints) construct to
support parameters. Parameters would work the same way as for interfaces.

## Where constraints

So far, we have restricted a generic type parameter by saying it has to
implement an interface or a set of interfaces. There are a variety of other
constraints we would like to be able to express, such as applying restrictions
to its associated types and associated constants. This is done using the `where`
operator that adds constraints to a type-of-type.

The where operator can be applied to a type-of-type in a declaration context:

```
// Constraints on function parameters:
fn F[V:! D where ...](v: V) { ... }

// Constraints on a class parameter:
class S(T:! B where ...) {
  // Constraints on a method:
  fn G[me: Self, V:! D where ...](v: V);
}

// Constraints on an interface parameter:
interface A(T:! B where ...) {
  // Constraints on an associated type:
  let U:! C where ...;
  // Constraints on an associated method:
  fn G[me: Self, V:! D where ...](v: V);
}
```

We also allow you to name constraints using a `where` operator in a `let` or
`constraint` definition. The expressions that can follow the `where` keyword are
described in the ["constraint use cases"](#constraint-use-cases) section, but
generally look like boolean expressions that should evaluate to `true`.

The result of applying a `where` operator to a type-of-type is another
type-of-type. Note that this expands the kinds of requirements that
type-of-types can have from just interface requirements to also include the
various kinds of constraints discussed later in this section. In addition, it
can introduce relationships between different type variables, such as that a
member of one is equal to the member of another.

**Comparison with other languages:** Both Swift and Rust use `where` clauses on
declarations instead of in the expression syntax. These happen after the type
that is being constrained has been given a name and use that name to express the
constraint.

Rust also supports
[directly passing in the values for associated types](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types)
when using a trait as a constraint. This is helpful when specifying concrete
types for all associated types in a trait in order to
[make it object safe so it can be used to define a trait object type](https://rust-lang.github.io/rfcs/0195-associated-items.html#trait-objects).

Rust is adding trait aliases
([RFC](https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md),
[tracking issue](https://github.com/rust-lang/rust/issues/41517)) to support
naming some classes of constraints.

### Constraint use cases

#### Set an associated constant to a specific value

We might need to write a function that only works with a specific value of an
[associated constant](#associated-constants) `N`. In this case, the name of the
associated constant is written first, followed by an `=`, and then the value:

```
fn PrintPoint2D[PointT:! NSpacePoint where .N = 2](p: PointT) {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Similarly in an interface definition:

```
interface {
  let PointT:! NSpacePoint where .N = 2;
}
```

To name such a constraint, you may use a `let` or a `constraint` declaration:

```
let Point2DInterface:! auto = NSpacePoint where .N = 2;
constraint Point2DInterface {
  extends NSpacePoint where .N = 2;
}
```

**Concern:** Using `=` for this use case is not consistent with other `where`
clauses that write a boolean expression that evaluates to `true` when the
constraint is satisfied.

A constraint to say that two associated constants should have the same value
without specifying what specific value they should have must use `==` instead of
`=`:

```
interface PointCloud {
  let Dim:! i32;
  let PointT:! NSpacePoint where .N == Dim;
}
```

#### Same type constraints

##### Set an associated type to a specific value

Functions accepting a generic type might also want to constrain one of its
associated types to be a specific, concrete type. For example, we might want to
have a function only accept stacks containing integers:

```
fn SumIntStack[T:! Stack where .ElementType = i32](s: T*) -> i32 {
  var sum: i32 = 0;
  while (!s->IsEmpty()) {
    // s->Pop() has type `T.ElementType` == i32:
    sum += s->Pop();
  }
  return sum;
}
```

To name these sorts of constraints, we could use `let` statements or
`constraint` definitions:

```
let IntStack:! auto = Stack where .ElementType = i32;
constraint IntStack {
  extends Stack where .ElementType = i32;
}
```

##### Equal generic types

Alternatively, two generic types could be constrained to be equal to each other,
without specifying what that type is. This uses `==` instead of `=`. For
example, we could make the `ElementType` of an `Iterator` interface equal to the
`ElementType` of a `Container` interface as follows:

```
interface Iterator {
  let ElementType:! Type;
  ...
}
interface Container {
  let ElementType:! Type;
  let IteratorType:! Iterator where .ElementType == ElementType;
  ...
}
```

Given an interface with two associated types

```
interface PairInterface {
  let Left:! Type;
  let Right:! Type;
}
```

we can constrain them to be equal in a function declaration:

```
fn F[MatchedPairType:! PairInterface where .Left == .Right]
    (x: MatchedPairType*);
```

or in an interface definition:

```
interface HasEqualPair {
  let P:! PairInterface where .Left == .Right;
}
```

This kind of constraint can be named:

```
let EqualPair:! auto =
    PairInterface where .Left == .Right;
constraint EqualPair {
  extends PairInterface where .Left == .Right;
}
```

Another example of same type constraints is when associated types of two
different interfaces are constrained to be equal:

```
fn Map[CT:! Container,
       FT:! Function where .InputType == CT.ElementType]
      (c: CT, f: FT) -> Vector(FT.OutputType);
```

###### Satisfying both type-of-types

If the two types being constrained to be equal have been declared with different
type-of-types, then the actual type value they are set to will have to satisfy
both constraints. For example, if `SortedContainer.ElementType` is declared to
be `Comparable`, then in this declaration:

```
fn Contains
    [SC:! SortedContainer,
     CT:! Container where .ElementType == SC.ElementType]
    (haystack: SC, needles: CT) -> bool;
```

the `where` constraint means `CT.ElementType` must satisfy `Comparable` as well.
However, inside the body of `Contains`, `CT.ElementType` will only act like the
implementation of `Comparable` is [external](#external-impl). That is, items
from the `needles` container won't have an unqualified `Compare` method member,
but can still be implicitly converted to `Comparable` and can still call
`Compare` using the qualified member syntax, `needle.(Comparable.Compare)(elt)`.
The rule is that an `==` `where` constraint between two type variables does not
modify the set of unqualified member names of either type. (If you write
`where .ElementType = String` with a `=` and a concrete type, then
`.ElementType` is actually set to `String` including the complete unqualified
`String` API.)

Note that `==` constraints are symmetric, so the previous declaration of
`Contains` is equivalent to an alternative declaration where `CT` is declared
first and the `where` clause is attached to `SortedContainer`:

```
fn Contains
    [CT:! Container,
     SC:! SortedContainer where .ElementType == CT.ElementType]
    (haystack: SC, needles: CT) -> bool;
```

#### Type bound for associated type

A `where` clause can express that a type must implement an interface. This is
more flexible than the usual approach of including that interface in the type
since it can be applied to associated type members as well.

##### Type bounds on associated types in declarations

In the following example, normally the `ElementType` of a `Container` can be any
type. The `SortContainer` function, however, takes a pointer to a type
satisfying `Container` with the additional constraint that its `ElementType`
must satisfy the `Comparable` interface.

```
interface Container {
  let ElementType:! Type;
  ...
}

fn SortContainer
    [ContainerType:! Container where .ElementType is Comparable]
    (container_to_sort: ContainerType*);
```

In contrast to [a same type constraint](#same-type-constraints), this does not
say what type `ElementType` exactly is, just that it must satisfy some
type-of-type.

**Open question:** How do you spell that? Provisionally we are writing `is`,
following Swift, but maybe we should have another operator that more clearly
returns a boolean like `has_type`?

**Note:** `Container` defines `ElementType` as having type `Type`, but
`ContainerType.ElementType` has type `Comparable`. This is because
`ContainerType` has type `Container where .ElementType is Comparable`, not
`Container`. This means we need to be a bit careful when talking about the type
of `ContainerType` when there is a `where` clause modifying it.

##### Type bounds on associated types in interfaces

Given these definitions (omitting `ElementType` for brevity):

```
interface IteratorInterface { ... }
interface ContainerInterface {
  let IteratorType:! IteratorInterface;
  ...
}
interface RandomAccessIterator {
  extends IteratorInterface;
  ...
}
```

We can then define a function that only accepts types that implement
`ContainerInterface` where its `IteratorType` associated type implements
`RandomAccessIterator`:

```
fn F[ContainerType:! ContainerInterface
     where .IteratorType is RandomAccessIterator]
    (c: ContainerType);
```

We would like to be able to name this constraint, defining a
`RandomAccessContainer` to be a type-of-type whose types satisfy
`ContainerInterface` with an `IteratorType` satisfying `RandomAccessIterator`.

```
let RandomAccessContainer:! auto =
    ContainerInterface where .IteratorType is RandomAccessIterator;
// or
constraint RandomAccessContainer {
  extends ContainerInterface
      where .IteratorType is RandomAccessIterator;
}

// With the above definition:
fn F[ContainerType:! RandomAccessContainer](c: ContainerType);
// is equivalent to:
fn F[ContainerType:! ContainerInterface
     where .IteratorType is RandomAccessIterator]
    (c: ContainerType);
```

#### Combining constraints

Constraints can be combined by separating constraint clauses with the `and`
keyword. This example expresses a constraint that two associated types are equal
and satisfy an interface:

```
fn EqualContainers
    [CT1:! Container,
     CT2:! Container where .ElementType is HasEquality
                       and .ElementType == CT1.ElementType]
    (c1: CT1*, c2: CT2*) -> bool;
```

**Comparison with other languages:** Swift and Rust use commas `,` to separate
constraint clauses, but that only works because they place the `where` in a
different position in a declaration. In Carbon, the `where` is attached to a
type in a parameter list that is already using commas to separate parameters.

#### Recursive constraints

We sometimes need to constrain a type to equal one of its associated types. In
this first example, we want to represent the function `Abs` which will return
`Self` for some but not all types, so we use an associated type `MagnitudeType`
to encode the return type:

```
interface HasAbs {
  extends Numeric;
  let MagnitudeType:! Numeric;
  fn Abs[me: Self]() -> MagnitudeType;
}
```

For types representing subsets of the real numbers, such as `i32` or `f32`, the
`MagnitudeType` will match `Self`, the type implementing an interface. For types
representing complex numbers, the types will be different. For example, the
`Abs()` applied to a `Complex64` value would produce a `f32` result. The goal is
to write a constraint to restrict to the first case.

In a second example, when you take the slice of a type implementing `Container`
you get a type implementing `Container` which may or may not be the same type as
the original container type. However, taking the slice of a slice always gives
you the same type, and some functions want to only operate on containers whose
slice type is the same as the container type.

To solve this problem, we think of `Self` as an actual associated type member of
every interface. We can then address it using `.Self` in a `where` clause, like
any other associated type member.

```
fn Relu[T:! HasAbs where .MagnitudeType == .Self](x: T) {
  // T.MagnitudeType == T so the following is allowed:
  return (x.Abs() + x) / 2;
}
fn UseContainer[T:! Container where .SliceType == .Self](c: T) -> bool {
  // T.SliceType == T so `c` and `c.Slice(...)` can be compared:
  return c == c.Slice(...);
}
```

Notice that in an interface definition, `Self` refers to the type implementing
this interface while `.Self` refers to the associated type currently being
defined.

```
interface Container {
  let ElementType:! Type;

  let SliceType:! Container
      where .ElementType == ElementType and
            .SliceType == .Self;

  fn GetSlice[addr me: Self*]
      (start: IteratorType, end: IteratorType) -> SliceType;
}
```

These recursive constraints can be named:

```
let RealAbs:! auto = HasAbs where .MagnitudeType == .Self;
constraint RealAbs {
  extends HasAbs where .MagnitudeType == Self;
}
let ContainerIsSlice:! auto =
    Container where .SliceType == .Self;
constraint ContainerIsSlice {
  extends Container where .SliceType == Self;
}
```

Note that using the `constraint` approach we can name these constraints using
`Self` instead of `.Self`, since they are facets of the same type: `Self` is the
facet corresponding to the containing interface and `.Self` is the facet
corresponding to the interface being extended.

#### Parameterized type implements interface

There are times when a function will pass a generic type parameter of the
function as an argument to a parameterized type, as in the previous case, and in
addition the function needs the result to implement a specific interface.

```
// Some parametized type.
class Vector(T:! Type) { ... }

// Parameterized type implements interface only for some arguments.
external impl Vector(String) as Printable { ... }

// Constraint: `T` such that `Vector(T)` implements `Printable`
fn PrintThree
    [T:! Type where Vector(.Self) is Printable]
    (a: T, b: T, c: T) {
  var v: Vector(T) = (a, b, c);
  Print(v);
}
```

**Comparison with other languages:** This use case was part of the
[Rust rationale for adding support for `where` clauses](https://rust-lang.github.io/rfcs/0135-where.html#motivation).

#### Another type implements parameterized interface

In this case, we need some other type to implement an interface parameterized by
a generic type parameter. The syntax for this case follows the previous case,
except now the `.Self` parameter is on the interface to the right of the `is`.
For example, we might need a type parameter `T` to support explicit conversion
from an integer type like `i32`:

```
interface As(T:! Type) {
  fn Convert[me: Self]() -> T;
}

fn Double[T:! Mul where i32 is As(.Self)](x: T) -> T {
  return x * (2 as T);
}
```

### Implied constraints

Imagine we have a generic function that accepts an arbitrary `HashMap`:

```
fn LookUp[KeyType:! Type](hm: HashMap(KeyType, i32)*,
                          k: KeyType) -> i32;

fn PrintValueOrDefault[KeyType:! Printable,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

The `KeyType` in these declarations does not satisfy the requirements of
`HashMap`, which requires the type implement `Hashable` and other interfaces:

```
class HashMap(
    KeyType:! Hashable & EqualityComparable & Movable,
    ...) { ... }
```

In this case, `KeyType` gets `Hashable` and so on as _implied constraints_.
Effectively that means that these functions are automatically rewritten to add a
`where` constraint on `KeyType` attached to the `HashMap` type:

```
fn LookUp[KeyType:! Type]
    (hm: HashMap(KeyType, i32)*
        where KeyType is Hashable & EqualityComparable & Movable,
     k: KeyType) -> i32;

fn PrintValueOrDefault[KeyType:! Printable,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT)
        where KeyType is Hashable & EqualityComparable & Movable,
     key: KeyT);
```

In this case, Carbon will accept the definition and infer the needed constraints
on the generic type parameter. This is both more concise for the author of the
code and follows the
["don't repeat yourself" principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
This redundancy is undesirable since it means if the needed constraints for
`HashMap` are changed, then the code has to be updated in more locations.
Further it can add noise that obscures relevant information. In practice, any
user of these functions will have to pass in a valid `HashMap` instance, and so
will have already satisfied these constraints.

This implied constraint is equivalent to the explicit constraint that each
parameter and return type [is legal](#must-be-legal-type-argument-constraints).

**Note:** These implied constraints affect the _requirements_ of a generic type
parameter, but not its _unqualified member names_. This way you can always look
at the declaration to see how name resolution works, without having to look up
the definitions of everything it is used as an argument to.

**Limitation:** To limit readability concerns and ambiguity, this feature is
limited to a single signature. Consider this interface declaration:

```
interface GraphNode {
  let Edge:! Type;
  fn EdgesFrom[me: Self]() -> HashSet(Edge);
}
```

One approach would be to say the use of `HashSet(Edge)` in the signature of the
`EdgesFrom` function would imply that `Edge` satisfies the requirements of an
argument to `HashSet`, such as being `Hashable`. Another approach would be to
say that the `EdgesFrom` would only be conditionally available when `Edge` does
satisfy the constraints on `HashSet` arguments. Instead, Carbon will reject this
definition, requiring the user to include all the constraints required for the
other declarations in the interface in the declaration of the `Edge` associated
type. Similarly, a parameter to a class must be declared with all the
constraints needed to declare the members of the class that depend on that
parameter.

**Comparison with other languages:** Both Swift
([1](https://www.swiftbysundell.com/tips/inferred-generic-type-constraints/),
[2](https://github.com/apple/swift/blob/main/docs/Generics.rst#constraint-inference))
and
[Rust](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=0b2d645bd205f24a7a6e2330d652c32e)
support some form of this feature as part of their type inference.

#### Must be legal type argument constraints

Now consider the case that the generic type parameter is going to be used as an
argument to a parameterized type in a function body, not in the signature. If
the parameterized type was explicitly mentioned in the signature, the implied
constraint feature would ensure all of its requirements were met. The developer
can create a trivial
[parameterized type implements interface](#parameterized-type-implements-interface)
`where` constraint to just say the type is a legal with this argument, by saying
that the parameterized type implements `Type`, which all types do.

For example, a function that adds its parameters to a `HashSet` to deduplicate
them, needs them to be `Hashable` and so on. To say "`T` is a type where
`HashSet(T)` is legal," we can write:

```
fn NumDistinct[T:! Type where HashSet(.Self) is Type]
    (a: T, b: T, c: T) -> i32 {
  var set: HashSet(T);
  set.Add(a);
  set.Add(b);
  set.Add(c);
  return set.Size();
}
```

This has the same advantages over repeating the constraints on `HashSet`
arguments in the type of `T` as the general implied constraints above.

### Open question: referencing names in the interface being defined

Should the constraint in a `where` clause be required to only reference earlier
names from this scope, as in this example?

```
interface Graph {
  let E: Edge;
  let V: Vert where .E == E and .Self == E.V;
}
```

The downside is that if you could reference later names, there is a more
pleasingly symmetric formulation of those same constraints:

```
interface Graph {
  let E: Edge where .V == V;
  let V: Vert where .E == E;
}
```

**TODO:** Revisit this question once issue
[#472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472)
and proposal
[#875: Principle: information accumulation](https://github.com/carbon-language/carbon-lang/pull/875)
are resolved.

### Manual type equality

Imagine we have some function with generic parameters:

```
fn F[T:! SomeInterface](x: T) {
  x.G(x.H());
}
```

We want to know if the return type of method `T.H` is the same as the parameter
type of `T.G` in order to typecheck the function. However, determining whether
two type expressions are transitively equal is in general undecidable, as
[has been shown in Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).

Carbon's approach is to only allow implicit conversions between two type
expressions that are constrained to be equal in a single where clause. This
means that if two type expressions are only transitively equal, the user will
need to include a sequence of casts or use an
[`observe` declaration](#observe-declarations) to convert between them.

Given this interface `Transitive` that has associated types that are constrained
to all be equal, with interfaces `P`, `Q`, and `R`:

```
interface P { fn InP[me:Self](); }
interface Q { fn InQ[me:Self](); }
interface R { fn InR[me:Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[me: Self]() -> A;
  fn TakesC[me:Self](c: C);
}
```

A cast to `B` is needed to call `TakesC` with a value of type `A`, so each step
only relies on one equality:

```
fn F[T:! Transitive](t: T) {
  // Allowed
  t.TakesC(t.GetA() as T.B);

  // Allowed
  let b: T.B = t.GetA();
  t.TakesC(b);

  // Not allowed: t.TakesC(t.GetA());
}
```

A value of type `A`, such as the return value of `GetA()`, has the API of `P`.
Any such value also implements `Q`, and since the compiler can see that by way
of a single `where` equality, values of type `A` are treated as if they
implement `Q` externally. However, the compiler will require a cast to `B` or
`C` to see that the type implements `R`.

```
fn TakesPQR[U:! P & Q & R](u: U);

fn G[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // Allowed: `T.A` implements `P`.
  a.InP();

  // Allowed: `T.A` implements `Q` externally.
  a.(Q.InQ)();

  // Not allowed: a.InQ();

  // Allowed: values of type `T.A` may be cast
  // to `T.B`, which implements `Q`.
  (a as T.B).InQ();

  // Allowed: `T.B` implements `R` externally.
  (a as T.B).(R.InR)();

  // Not allowed: TakesPQR(a);

  // Allowed: `T.B` implements `P`, `Q`, and
  // `R`, though the implementations of `P`
  // and `R` are external.
  TakesPQR(a as T.B);
}
```

The compiler may have several different `where` clauses to consider,
particularly when an interface has associated types that recursively satisfy the
same interface. For example, given this interface `Commute`:

```
interface Commute {
  let X:! Commute;
  let Y:! Commute where .X == X.Y;

  fn GetX[me: Self]() -> X;
  fn GetY[me: Self]() -> Y;
  fn TakesXXY[me:Self](xxy: X.X.Y);
}
```

and a function `H` taking a value with some type implementing this interface,
then the following would be legal statements in `H`:

```
fn H[C: Commute](c: C) {
  // Legal: argument has type `C.X.X.Y`
  c.TakesXXY(c.GetX().GetX().GetY());

  // Legal: argument has type `C.X.Y.X` which is equal
  // to `C.X.X.Y` following only one `where` clause.
  c.TakesXXY(c.GetX().GetY().GetX());

  // Legal: cast is legal since it matches a `where`
  // clause, and produces an argument that has type
  // `C.X.Y.X`.
  c.TakesXXY(c.GetY().GetX().GetX() as C.X.Y.X);
}
```

That last call would not be legal without the cast, though.

**Comparison with other languages:** Other languages such as Swift and Rust
instead perform automatic type equality. In practice this means that their
compiler can reject some legal programs based on heuristics simply to avoid
running for an unbounded length of time.

The benefits of the manual approach include:

-   fast compilation, since the compiler does not need to explore a potentially
    large set of combinations of equality restrictions, supporting
    [Carbon's goal of fast and scalable development](/docs/project/goals.md#fast-and-scalable-development);
-   expressive and predictable semantics, since there are no limitations on how
    complex a set of constraints can be supported; and
-   simplicity.

The main downsides are:

-   manual work for the source code author to prove to the compiler that types
    are equal; and
-   verbosity.

We expect that rich error messages and IDE tooling will be able to suggest
changes to the source code when a single equality constraint is not sufficient
to show two type expressions are equal, but a more extensive automated search
can find a sequence that prove they are equal.

#### `observe` declarations

An `observe` declaration lists a sequence of type expressions that are equal by
some same-type `where` constraints. These `observe` declarations may be included
in an `interface` definition or a function body, as in:

```
interface Commute {
  let X:! Commute;
  let Y:! Commute where .X == X.Y;
  ...
  observe X.X.Y == X.Y.X == Y.X.X;
}

fn H[C: Commute](c: C) {
  observe C.X.Y.Y == C.Y.X.Y == C.Y.Y.X;
  ...
}
```

Every type expression after the first must be equal to some earlier type
expression in the sequence by a single `where` equality constraint. In this
example,

```
interface Commute {
  let X:! Commute;
  let Y:! Commute where .X == X.Y;
  ...
  // Legal:
  observe X.X.Y.Y == X.Y.X.Y == Y.X.X.Y == X.Y.Y.X;
}
```

the expression `X.Y.Y.X` is one equality away from `X.Y.X.Y` and so it is
allowed. This is even though `X.Y.X.Y` isn't the type expression immediately
prior to `X.Y.Y.X`.

After an `observe` declaration, all of the listed type expressions are
considered equal to each other using a single `where` equality. In this example,
the `observe` declaration in the `Transitive` interface definition provides the
link between associated types `A` and `C` that allows function `F` to type
check.

```
interface P { fn InP[me:Self](); }
interface Q { fn InQ[me:Self](); }
interface R { fn InR[me:Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[me: Self]() -> A;
  fn TakesC[me:Self](c: C);

  // Without this `observe` statement, the
  // calls in `F` below would not be allowed.
  observe A == B == C;
}

fn TakesPQR[U:! P & Q & R](u: U);

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // Allowed: `T.A` == `T.C`
  t.TakesC(a);
  a.(R.InR());

  // Allowed: `T.A` implements `P`,
  // `T.A` == `T.B` that implements `Q`, and
  // `T.A` == `T.C` that implements `R`.
  TakesPQR(a);
}
```

Since adding an `observe` statement only adds external implementations of
interfaces to generic types, they may be added without breaking existing code.

## Other constraints as type-of-types

There are some constraints that we will naturally represent as named
type-of-types. These can either be used directly to constrain a generic type
parameter, or in a `where ... is ...` clause to constrain an associated type.

The compiler determines which types implement these interfaces, developers can
not explicitly implement these interfaces for their own types.

**Open question:** Are these names part of the prelude or in a standard library?

### Is a derived class

Given a type `T`, `Extends(T)` is a type-of-type whose values are types that are
derived from `T`. That is, `Extends(T)` is the set of all types `U` that are
subtypes of `T`.

```
fn F[T:! Extends(BaseType)](p: T*);
fn UpCast[T:! Type](p: T*, U:! Type where T is Extends(.Self)) -> U*;
fn DownCast[T:! Type](p: T*, U:! Extends(T)) -> U*;
```

**Open question:** Alternatively, we could define a new `extends` operator:

```
fn F[T:! Type where .Self extends BaseType](p: T*);
fn UpCast[T:! Type](p: T*, U:! Type where T extends .Self) -> U*;
fn DownCast[T:! Type](p: T*, U:! Type where .Self extends T) -> U*;
```

**Comparison to other languages:** In Swift, you can
[add a required superclass to a type bound using `&`](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282).

### Type compatible with another type

Given a type `U`, define the type-of-type `CompatibleWith(U)` as follows:

> `CompatibleWith(U)` is a type whose values are types `T` such that `T` and `U`
> are [compatible](terminology.md#compatible-types). That is values of types `T`
> and `U` can be cast back and forth without any change in representation (for
> example `T` is an [adapter](#adapting-types) for `U`).

To support this, we extend the requirements that type-of-types are allowed to
have to include a "data representation requirement" option.

`CompatibleWith` determines an equivalence relationship between types.
Specifically, given two types `T1` and `T2`, they are equivalent if
`T1 is CompatibleWith(T2)`. That is, if `T1` has the type `CompatibleWith(T2)`.

**Note:** Just like interface parameters, we require the user to supply `U`,
they may not be deduced. Specifically, this code would be illegal:

```
fn Illegal[U:! Type, T:! CompatibleWith(U)](x: T*) ...
```

In general there would be multiple choices for `U` given a specific `T` here,
and no good way of picking one. However, similar code is allowed if there is
another way of determining `U`:

```
fn Allowed[U:! Type, T:! CompatibleWith(U)](x: U*, y: T*) ...
```

#### Same implementation restriction

In some cases, we need to restrict to types that implement certain interfaces
the same way as the type `U`.

> The values of type `CompatibleWith(U, TT)` are types satisfying
> `CompatibleWith(U)` that have the same implementation of `TT` as `U`.

For example, if we have a type `HashSet(T)`:

```
class HashSet(T:! Hashable) { ... }
```

Then `HashSet(T)` may be cast to `HashSet(U)` if
`T is CompatibleWith(U, Hashable)`. The one-parameter interpretation of
`CompatibleWith(U)` is recovered by letting the default for the second `TT`
parameter be `Type`.

#### Example: Multiple implementations of the same interface

This allows us to represent functions that accept multiple implementations of
the same interface for a type.

```
enum CompareResult { Less, Equal, Greater }
interface Comparable {
  fn Compare[me: Self](that: Self) -> CompareResult;
}
fn CombinedLess[T:! Type](a: T, b: T,
                          U:! CompatibleWith(T) & Comparable,
                          V:! CompatibleWith(T) & Comparable) -> bool {
  match ((a as U).Compare(b as U)) {
    case CompareResult.Less => { return True; }
    case CompareResult.Greater => { return False; }
    case CompareResult.Equal => {
      return (a as V).Compare(b as V) == CompareResult.Less;
    }
  }
}
```

Used as:

```
class Song { ... }
adapter SongByArtist for Song { impl as Comparable { ... } }
adapter SongByTitle for Song { impl as Comparable { ... } }
var s1: Song = ...;
var s2: Song = ...;
assert(CombinedLess(s1, s2, SongByArtist, SongByTitle) == True);
```

We might generalize this to a list of implementations:

```
fn CombinedCompare[T:! Type]
    (a: T, b: T, CompareList:! List(CompatibleWith(T) & Comparable))
    -> CompareResult {
  for (let U:! auto in CompareList) {
    var result: CompareResult = (a as U).Compare(b);
    if (result != CompareResult.Equal) {
      return result;
    }
  }
  return CompareResult.Equal;
}

assert(CombinedCompare(Song(...), Song(...), (SongByArtist, SongByTitle)) ==
       CompareResult.Less);
```

**Open question:** How are compile-time lists of types declared and iterated
through? They will also be needed for
[variadic argument support](#variadic-arguments).

#### Example: Creating an impl out of other impls

And then to package this functionality as an implementation of `Comparable`, we
combine `CompatibleWith` with [type adaptation](#adapting-types):

```
adapter ThenCompare(T:! Type,
                    CompareList:! List(CompatibleWith(T) & Comparable)) for T {
  impl as Comparable {
    fn Compare[me: Self](that: Self) -> CompareResult {
      for (let U:! auto in CompareList) {
        var result: CompareResult = (this as U).Compare(that);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

let SongByArtistThenTitle: auto = ThenCompare(Song, (SongByArtist, SongByTitle));
var s1: Song = ...;
var s2: SongByArtistThenTitle = Song(...) as SongByArtistThenTitle;
assert((s1 as SongByArtistThenTitle).Compare(s2) == CompareResult.Less);
```

### Type facet of another type

Similar to `CompatibleWith(T)`, `FacetOf(T)` introduces an equivalence
relationship between types. `T1 is FacetOf(T2)` if both `T1` and `T2` are facets
of the same type.

### Sized types and type-of-types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`i32`, `f64`, and so on), most
    [classes](/docs/design/classes.md), and most other concrete types.
-   It could be known generically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a
    [dynamic type](#runtime-type-fields), a slice, variable-sized type (such as
    [found in Rust](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)),
    or you could dereference a pointer to a base class that could actually point
    to a [derived class](/docs/design/classes.md#inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

A type is called "sized" if it is in the first two categories, and "unsized"
otherwise. Note: something with size 0 is still considered "sized". The
type-of-type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring variables of that type,
taking values of that type as parameters, returning values of that type, and
defining arrays of that type. Users will not typically need to express the
`Sized` constraint explicitly, though, since it will usually be a dependency of
some other constraint the type will need such as `Movable`.

**Note:** The compiler will determine which types are "sized", this is not
something types will implement explicitly like ordinary interfaces.

Example:

```
// In the Carbon standard library
interface DefaultConstructible {
  // Types must be sized to be default constructible.
  impl as Sized;
  fn Default() -> Self;
}

// Classes are "sized" by default.
class Name {
  impl as DefaultConstructible {
    fn Default() -> Self { ... }
  }
  ...
}

fn F[T:! Type](x: T*) {  // T is unsized.
  // ✅ Allowed: may access unsized values through a pointer.
  var y: T* = x;
  // ❌ Illegal: T is unsized.
  var z: T;
}

// T is sized, but its size is only known generically.
fn G[T: DefaultConstructible](x: T*) {
  // ✅ Allowed: T is default constructible, which means sized.
  var y: T = T.Default();
}

var z: Name;
// ✅ Allowed: `Name` is sized and implements `DefaultConstructible`.
G(&z);
```

**Open question:** Even if the size is fixed, it won't be known at the time of
compiling the generic function if we are using the dynamic strategy. Should we
automatically
[box](<https://en.wikipedia.org/wiki/Object_type_(object-oriented_programming)#Boxing>)
local variables when using the dynamic strategy? Or should we only allow
`MaybeBox` values to be instantiated locally? Or should this just be a case
where the compiler won't necessarily use the dynamic strategy?

**Open question:** Should the `Sized` type-of-type expose an associated constant
with the size? So you could say `T.ByteSize` in the above example to get a
generic int value with the size of `T`. Similarly you might say `T.ByteStride`
to get the number of bytes used for each element of an array of `T`.

#### Model

This requires a special integer field be included in the witness table type to
hold the size of the type. This field will only be known generically, so if its
value is used for type checking, we need some way of evaluating those type tests
symbolically.

### `TypeId`

There are some capabilities every type can provide. For example, every type
should be able to return its name or identify whether it is equal to another
type. It is rare, however, for code to need to access these capabilities, so we
relegate these capabilities to an interface called `TypeId` that all types
automatically implement. This way generic code can indicate that it needs those
capabilities by including `TypeId` in the list of requirements. In the case
where no type capabilities are needed, for example the code is only manipulating
pointers to the type, you would write `T:! Type` and get the efficiency of
`void*` but without giving up type safety.

```
fn SortByAddress[T:! Type](v: Vector(T*)*) { ... }
```

In particular, the compiler should in general avoid monomorphizing to generate
multiple instantiations of the function in this case.

**Note:** To achieve this goal, the user will not even be allowed to destroy a
value of type `T` in this case.

**Open question:** Should `TypeId` be implemented externally for types to avoid
name pollution (`.TypeName`, `.TypeHash`, etc.) unless the function specifically
requests those capabilities?

## Future work

### Conditional conformance

[The problem](terminology.md#conditional-conformance) we are trying to solve
here is expressing that we have an `impl` of some interface for some type, but
only if some additional type restrictions are met.

### Parameterized impls

Also known as "blanket `impl`s", these are when you have an `impl` definition
that is parameterized so it applies to more than a single type and interface
combination.

#### Lookup resolution and specialization

For this to work, we need a rule that picks a single `impl` in the case where
there are multiple `impl` definitions that match a particular type and interface
combination.

### Dynamic types

Generics provide enough structure to support runtime dispatch for values with
types that vary at runtime, without giving up type safety. Both Rust and Swift
have demonstrated the value of this feature.

#### Runtime type parameters

This feature is about allowing a function's type parameter to be passed in as a
dynamic (non-generic) parameter. All values of that type would still be required
to have the same type.

#### Runtime type fields

Instead of passing in a single type parameter to a function, we could store a
type per value. This changes the data layout of the value, and so is a somewhat
more invasive change. It also means that when a function operates on multiple
values they could have different real types.

### Abstract return types

This lets you return an anonymous type implementing an interface from a
function. In Rust this is the
[`impl Trait` return type](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).

In Swift, there are discussions about implementing this feature under the name
"reverse generics" or "opaque result types":
[1](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--reverse-generics),
[2](https://forums.swift.org/t/reverse-generics-and-opaque-result-types/21608),
[3](https://forums.swift.org/t/se-0244-opaque-result-types/21252),
[4](https://forums.swift.org/t/se-0244-opaque-result-types-reopened/22942),
Swift is considering spelling this `<V: Collection> V` or `some Collection`.

### Interface defaults

Rust supports specifying defaults for
[interface parameters](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading),
[methods](https://doc.rust-lang.org/book/ch10-02-traits.html#default-implementations),
[associated constants](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants-examples).
We should support this too. It is helpful for evolution, as well as reducing
boilerplate. Defaults address the gap between the minimum necessary for a type
to provide the desired functionality of an interface and the breadth of API that
user's desire.

### Evolution

There are a collection of use cases for making different changes to interfaces
that are already in use. These should be addressed either by describing how they
can be accomplished with existing generics features, or by adding features.

In addition, evolution from (C++ or Carbon) templates to generics needs to be
supported and made safe.

### Testing

The idea is that you would write tests alongside an interface that validate the
expected behavior of any type implementing that interface.

### Operator overloading

We will need a story for defining how an operation is overloaded for a type by
implementing an interface for that type.

### Impls with state

A feature we might consider where an `impl` itself can have state.

### Generic associated types and higher-ranked types

This would be some way to express the requirement that there is a way to go from
a type to an implementation of an interface parameterized by that type.

#### Generic associated types

Generic associated types are about when this is a requirement of an interface.
These are also called "associated type constructors."

#### Higher-ranked types

Higher-ranked types are used to represent this requirement in a function
signature. They can be
[emulated using generic associated types](https://smallcultfollowing.com/babysteps//blog/2016/11/03/associated-type-constructors-part-2-family-traits/).

### Field requirements

We might want to allow interfaces to express the requirement that any
implementing type has a particular field. This would be to match the
expressivity of inheritance, which can express "all subtypes start with this
list of fields."

### Generic type specialization

See [generic specialization](terminology.md#generic-specialization) for a
description of what this might involve.

### Bridge for C++ customization points

See details in [the goals document](goals.md#bridge-for-c-customization-points).

### Variadic arguments

Some facility for allowing a function to generically take a variable number of
arguments.

### Range constraints on generic integers

We currently only support `where` clauses on type-of-types. We may want to also
support constraints on generic integers. The constraint with the most expected
value is the ability to do comparisons like `<`, or `>=`. For example, you might
constrain the `N` member of [`NSpacePoint`](#associated-constants) using an
expression like `PointT:! NSpacePoint where 2 <= .N and .N <= 3`.

The concern here is supporting this at compile time with more benefit than
complexity. For example, we probably don't want to support integer-range based
types at runtime, and there are also concerns about reasoning about comparisons
between multiple generic integer parameters. For example, if `J < K` and
`K <= L`, can we call a function that requires `J < L`? There is also a
secondary syntactic concern about how to write this kind of constraint on a
parameter, as opposed to an associated type, as in `N:! u32 where ___ >= 2`.

## References

-   [#553: Generics details part 1](https://github.com/carbon-language/carbon-lang/pull/553)
-   [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)
-   [#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818)
