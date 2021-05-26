# Carbon deep dive: combined interfaces

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
    -   [Rejected: out-of-line impl](#rejected-out-of-line-impl)
    -   [Qualified member names](#qualified-member-names)
-   [Generics](#generics)
    -   [Model](#model)
-   [Interfaces recap](#interfaces-recap)
-   [Type-types and facet types](#type-types-and-facet-types)
-   [Structural interfaces](#structural-interfaces)
    -   [Subtyping between type-types](#subtyping-between-type-types)
    -   [Future work: method constraints](#future-work-method-constraints)
-   [Combining interfaces by anding type-types](#combining-interfaces-by-anding-type-types)
-   [Interface requiring other interfaces](#interface-requiring-other-interfaces)
    -   [Interface extension](#interface-extension)
        -   [`extends` and `impl` with structural interfaces](#extends-and-impl-with-structural-interfaces)
        -   [Diamond dependency issue](#diamond-dependency-issue)
    -   [Use case: overload resolution](#use-case-overload-resolution)
-   [Type compatibility](#type-compatibility)
-   [Future work](#future-work)
    -   [Adapting types](#adapting-types)
    -   [Associated constants](#associated-constants)
    -   [Associated types](#associated-types)
    -   [Parameterized interfaces](#parameterized-interfaces)
        -   [Impl lookup](#impl-lookup)
    -   [Constraints](#constraints)
    -   [Conditional conformance](#conditional-conformance)
    -   [Parameterized impls](#parameterized-impls)
        -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
    -   [Other constraints as type-types](#other-constraints-as-type-types)
        -   [Sized types and type-types](#sized-types-and-type-types)
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
    -   [Reverse generics for return types](#reverse-generics-for-return-types)

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
type-types" model by saying the type of `T` is `ConvertibleToString`.

Since we can figure out `T` from the type of `val`, we don't need the caller to
pass in `T` explicitly, it can be an
[implicit argument](terminology.md#implicit-parameter) (also see
[implicit argument](overview.md#implicit-arguments) in the Generics overview
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
[type's size](#sized-types-and-type-types) (represented by an integer constant
member of the type) is an optional member of an interface and its
implementation. There are a few cases why we would include another interface
implementation as a member:

-   [associated types](#associated-types)
-   [type parameters](#parameterized-interfaces)
-   [interface requirements](#interface-requiring-other-interfaces)

The function can decide whether that type argument is passed in
[statically](terminology.md#static-dispatch-witness-table) (basically generating
a separate function body for every different type passed in) by using the
"generic argument" syntax (`:$`, see [the generics section](#generics) below) or
[dynamically](terminology.md#dynamic-dispatch-witness-table) using the regular
argument syntax (just a colon, `:`, see
[the runtime type parameters section](#runtime-type-parameters) below). Either
way, the interface contains enough information to
[type and definition check](terminology.md#complete-definition-checking) the
function body -- you can only call functions defined in the interface in the
function body. Contrast this with making the type a template argument, where you
could just use `Type` instead of an interface and it will work as long as the
function is only called with types that allow the definition of the function to
compile. You are still allowed to declare templated type arguments as having an
interface type, and this will add a requirement that the type satisfy the
interface independent of whether that is needed to compile the function body,
but it is strictly optional. You might still do this to get clearer error
messages, document expectations, or express that a type has certain semantics
beyond what is captured in its member function names and signatures).

The last piece of the puzzle is how the caller of the function can produce a
value with the right type. Let's say the user has a value of type `Widget`, and
of course widgets have all sorts of functionality. If we want a `Widget` to be
printed using the `PrintToStdout` function, it needs to implement the
`ConvertibleToString` interface. Note that we _don't_ say that `Widget` is of
type `ConvertibleToString` but instead that it has a "facet type". This means
there is another type, called `Widget as ConvertibleToString`, with the
following properties:

-   `Widget as ConvertibleToString` has the same _data representation_ as
    `Widget`.
-   `Widget as ConvertibleToString` is an implementation of the interface
    `ConvertibleToString`. The functions of `Widget as ConvertibleToString` are
    just implementations of the names and signatures defined in the
    `ConvertibleToString` interface, like `ToString`, and not the functions
    defined on `Widget` values.
-   Carbon will implicitly cast values from type `Widget` to type
    `Widget as ConvertibleToString` when calling a function that can only accept
    types of type `ConvertibleToString`.
-   Typically, `Widget` would also have definitions for the methods of
    `ConvertibleToString`, such as `ToString`, unless the implementation of
    `ConvertibleToString` for `Widget` was defined as `external`.
-   You may access the `ToString` function for a `Widget` value `w` by writing a
    _qualified_ function call, like `w.(ConvertibleToString.ToString)()`. This
    qualified syntax is available whether or not the implementation is defined
    as `external`.
-   If other interfaces are implemented for `Widget`, they are also implemented
    for `Widget as ConvertibleToString` as well. The only thing that changes
    when casting a `Widget` `w` to `Widget as ConvertibleToString` are the names
    that are accessible without using the qualification syntax.

We define these facet types (alternatively, interface implementations) either
with the type, with the interface, or somewhere else where Carbon can be
guaranteed to see when needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

If `Widget` doesn't implement an interface or we would like to use a different
implementation of that interface, we can define another type that also has the
same data representation as `Widget` that has whatever different interface
implementations we want. However, Carbon won't implicitly cast to that other
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
  method (Self: a) Add(Self: b) -> Self;
  method (Self: a) Scale(Double: v) -> Self;
}
```

The syntax here is intended to match how the same members would be defined in a
type. Each declaration in the interface defines an _associated item_ (same
[terminology as Rust](https://doc.rust-lang.org/reference/items/associated-items.html)).
In this example, `Vector` has two associated methods, `Add` and `Scale`.

An interface defines a type-type, that is a type whose values are types. The
values of an interface are specifically
[facet types](terminology.md#facet-type), by which we mean types that are
declared as specifically implementing **exactly** this interface, and which
provide definitions for all the functions (and other members) declared in the
interface.

## Implementing interfaces

Carbon interfaces are ["nominal"](terminology.md#nominal-interfaces), which
means that types explicitly describe how they implement interfaces. An
["impl"](terminology.md#impls-implementations-of-interfaces) defines how one
interface is implemented for a type. Every associated item is given a
definition. Different types satisfying `Vector` can have different definitions
for `Add` and `Scale`, so we say their definitions are associated with what type
is implementing `Vector`. The impl defines what is associated with the type for
that interface.

Impls may be defined inline inside the type definition:

```
struct Point {
  var Double: x;
  var Double: y;
  impl Vector {
    // In this scope, "Self" is an alias for "Point".
    method (Self: a) Add(Self: b) -> Self {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    }
    method (Self: a) Scale(Double: v) -> Self {
      return Point(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

Interfaces that are implemented inline contribute to the type's API:

```
var Point: p1 = (.x = 1.0, .y = 2.0);
var Point: p2 = (.x = 2.0, .y = 4.0);
Assert(p1.Scale(2.0) == p2);
Assert(p1.Add(p1) == p2);
```

### Facet type

The impl definition defines a [facet type](terminology.md#facet-type):
`Point as Vector`. While the API of `Point` includes the two fields `x` and `y`
along with the `Add` and `Scale` methods, the API of `Point as Vector` _only_
has the `Add` and `Scale` methods of the `Vector` interface. The facet type
`Point as Vector` is [compatible](terminology.md#compatible-types) with `Point`,
meaning their data representations are the same, so we allow you to cast between
the two freely:

```
var Point: a = (.x = 1.0, .y = 2.0);
// `a` has `Add` and `Scale` methods:
a.Add(a.Scale(2.0));

// Cast from Point implicitly
var Point as Vector: b = a;
// `b` has `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

// Will also implicitly cast when calling functions:
fn F(Point as Vector: c, Point: d) {
  d.Add(c.Scale(2.0));
}
F(a, b);

// Explicit casts
var Point as Vector: z = (a as (Point as Vector)).Scale(3.0);
z.Add(b);
var Point: w = z as Point;
```

These [casts](terminology.md#subtyping-and-casting) change which names are
exposed in the type's API, but as much as possible we don't want the meaning of
any given name to change. Instead we want these casts to simply change the
subset of names that are visible.

**Note:** In general the above is written assuming that casts are written
"`a as T`" where `a` is a value and `T` is the type to cast to. When we write
`Point as Vector`, the value `Point` is a type, and `Vector` is a type of a
type, or a "type-type".

**Note:** A type may implement any number of different interfaces, but may
provide at most one implementation of any single interface. This makes the act
of selecting an implementation of an interface for a type unambiguous throughout
the whole program, so for example `Point as Vector` is well defined.

We don't expect users to ordinarily name facet types explicitly in source code.
Instead, values are implicitly cast to a facet type as part of calling a generic
function, as described in the [Generics](#generics) section.

### Implementing multiple interfaces

To implement more than one interface when defining a type, simply include an
`impl` block per interface.

```
struct Point {
  var Double: x;
  var Double: y;
  impl Vector {
    method (Self: a) Add(Self: b) -> Self { ... }
    method (Self: a) Scale(Double: v) -> Self { ... }
  }
  impl Drawable {
    method (Self: a) Draw() { ... }
  }
}
```

In this case, all the functions `Add`, `Scale`, and `Draw` end up a part of the
API for `Point`. This means you can't implement two interfaces that have a name
in common.

```
struct GameBoard {
  impl Drawable {
    method (Self: this) Draw() { ... }
  }
  impl EndOfGame {
    // Error: `GameBoard` has two methods named
    // `Draw` with the same signature.
    method (Self: this) Draw() { ... }
    method (Self: this) Winner(Int: player) { ... }
  }
}
```

**Open question:** Should we have some syntax for the case where you want both
names to be given the same implementation? It seems like that might be a common
case, but we won't really know if this is an important case until we get more
experience.

```
struct Player {
  var String: name;
  impl Icon {
    method (Self: this) Name() -> String { return this.name; }
    // ...
  }
  impl GameUnit {
    // Possible syntax for defining `GameUnit.Name` as
    // the same as `Icon.Name`:
    alias Name = Icon.Name;
    // ...
  }
}
```

### External impl

Interfaces may also be implemented for a type externally, by using the `extend`
construct which takes the name of an existing type:

```
struct Point2 {
  var Double: x;
  var Double: y;
}

extend Point2 {
  // In this scope, "Self" is an alias for "Point2".
  impl Vector {
    method (Self: a) Add(Self: b) -> Self {
      return Point2(.x = a.x + b.x, .y = a.y + b.y);
    }
    method (Self: a) Scale(Double: v) -> Self {
      return Point2(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

The `extend` statement is allowed to be defined in a different library from
`Point2`, restricted by [the coherence/orphan rules](#impl-lookup) that ensure
that the implementation of an interface won't change based on imports. In
particular, the `extend` statement is allowed in the library defining the
interface (`Vector` in this case) in addition to the library that defines the
type (`Point2` here). This (at least partially) addresses
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

We don't want the API of `Point2` to change based on what is imported though. So
the `extend` statement does not add the interface's methods to the type. It
would be particularly bad if two different libraries implemented interfaces with
conflicting names both affected the API of a single type. The result is you can
find all the names of direct (unqualified) members of a type in the definition
of that type. The only thing that may be in another library is an `impl` of an
interface.

On the other hand, if we cast to the facet type, those methods do become
visible:

```
var Point2: a = (.x = 1.0, .y = 2.0);
// `a` does *not* have `Add` and `Scale` methods:
// Error: a.Add(a.Scale(2.0));

// Cast from Point2 implicitly
var Point2 as Vector: b = a;
// `b` does have `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

fn F(Point2 as Vector: c) {
  // Can call `Add` and `Scale` on `c` even though we can't on `a`.
  c.Add(c.Scale(2.0));
}
F(a);
```

You might intentionally use `extend` to implement an interface for a type to
avoid cluttering the API of that type, for example to avoid a name collision. A
syntax for reusing method implementations allows us to do this selectively when
needed:

```
struct Point3 {
  var Double: x;
  var Double: y;
  method (Self: a) Add(Self: b) -> Self {
    return Point3(.x = a.x + b.x, .y = a.y + b.y);
  }
}

extend Point3 {
  impl Vector {
    alias Add = Point3.Add;  // Syntax TBD
    method (Self: a) Scale(Double: v) -> Self {
      return Point3(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

With this definition, `Point3` includes `Add` in its API but not `Scale`, while
`Point3 as Vector` includes both. This maintains the property that you can
determine the API of a type by looking at its definition.

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
implementation. The concept of doing this as an "extension" of the original type
is more similar to
[Swift](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID277),
but, unlike Swift, we don't allow a type's API to be modified outside its
definition. In Rust, all implementations are external as in
[this example](https://doc.rust-lang.org/rust-by-example/trait.html).

### Rejected: out-of-line impl

We considered an out-of-line syntax for declaring and defining interface `impl`
blocks, to replace both the inline syntax and the `extend` statement. For,
example:

```
struct Point { ... }
impl Vector for Point { ... }
```

The main advantage of this syntax was that it was uniform across many cases,
including [conditional conformance](#conditional-conformance). It wasn't ideal
across a number of dimensions though:

-   it was redundant and verbose,
-   it was difficult to read and author, and
-   it could affect the API of the type outside of the type definition.

### Qualified member names

Given a value of type `Point2` and an interface `Vector` implemented for that
type, you can access the methods from that interface using the member's
_qualified name_, whether or not the implementation is done externally with an
`extend` statement:

```
var Point2: p1 = (.x = 1.0, .y = 2.0);
var Point2: p2 = (.x = 2.0, .y = 4.0);
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
  method (Self this) Draw();
}

extend Points.Point2 {
  impl Drawable { ... }
}
```

You could access `Draw` with a qualified name:

```
import Plot;
import Points;

var Points.Point2: p = (.x = 1.0, .y = 2.0);
p.(Plot.Drawable.Draw)();
```

**Comparison with other languages:** This is intended to be analogous to, in
C++, adding `ClassName::` in front of a member name to disambiguate, such as
[names defined in both a parent and child class](https://stackoverflow.com/questions/357307/how-to-call-a-parent-class-function-from-derived-class-function).

## Generics

Now let us write a function that can accept values of any type that has
implemented the `Vector` interface:

```
fn AddAndScaleGeneric[Vector:$ T](T: a, T: b, Double: s) -> T {
  return a.Add(b).Scale(s);
}
var Point: v = AddAndScaleGeneric(a, w, 2.5);
```

Here `T` is a type whose type is `Vector`. The `:$` syntax means that `T` is a
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
      Point as Vector: a, Point as Vector: b, Double: s)
      -> Point as Vector {
  return a.Add(b).Scale(s);
}
// May still be called with Point arguments, due to implicit casts.
// Similarly the return value can be implicitly cast to a Point.
var Point: v2 = AddAndScaleForPointAsVector(a, w, 2.5);
```

Since `Point` implements `Vector` inline, `Point` also has definitions for `Add`
and `Scale`:

```
fn AddAndScaleForPoint(Point: a, Point: b, Double: s) -> Point {
  return a.Add(b).Scale(s);
}

AddAndScaleForPoint(a, w, 2.5);
```

However, for another type implementing `Vector` but out-of-line using an
`extend` statement, such as `Point2`, the situation is different:

```
fn AddAndScaleForPoint2(Point2: a, Point2: b, Double: s) -> Point2 {
  // ERROR: `Point2` doesn't have `Add` or `Scale` methods.
  return a.Add(b).Scale(s);
}
```

Even though `Point2` doesn't have `Add` and `Scale` methods, it still implements
`Vector` and so can still call `AddAndScaleGeneric`:

```
var Point2: a2 = (.x = 1.0, .y = 2.0);
var Point2: w2 = (.x = 3.0, .y = 4.0);
var Point2: v3 = AddAndScaleGeneric(a, w, 2.5);
```

### Model

The underlying model here is interfaces are
[type-types](terminology.md#type-type), in particular, the type of
[facet types](terminology.md#facet-type):

-   [Interfaces](#interfaces) are types of
    [witness tables](terminology.md#witness-tables)
-   Facet types (defined by [Impls](#implementing-interfaces)) are
    [witness table](terminology.md#witness-tables) values
-   The compiler rewrites functions with an implicit type argument
    (`fn Foo[InterfaceName:$ T](...)`) to have an actual argument with type
    determined by the interface, and supplied at the callsite using a value
    determined by the impl.

For the example above, [the Vector interface](#interfaces) could be thought of
defining a witness table type like:

```
struct Vector {
  // Self is the representation type.
  var Type:$ Self;
  // `fnty` is **placeholder** syntax for a "function type",
  // so `Add` is a function that takes two `Self` parameters
  // and returns a value of type `Self`.
  var fnty(Self: a, Self: b) -> Self : Add;
  var fnty(Self: a, Double: v) -> Self : Scale;
}
```

The [impl of Vector for Point](#implementing-interfaces) would be a value of
this type:

```
var Vector : VectorForPoint = (
    .Self = Point,
    // `lambda` is **placeholder** syntax for defining a
    // function value.
    .Add = lambda(Point: a, Point: b) -> Point {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    },
    .Scale = lambda(Point: a, Double: v) -> Point {
      return Point(.x = a.x * v, .y = a.y * v);
    },
);
```

Finally we can define a generic function and call it, like
[`AddAndScaleGeneric` from the "Generics" section](#generics) by making the
witness table an explicit argument to the function:

```
fn AddAndScaleGeneric
    (Vector:$ impl, impl.Self: a, impl.Self: b, Double: s) -> impl.Self {
  return impl.Scale(impl.Add(a, b), s);
}
// Point implements Vector.
var Point: v = AddAndScaleGeneric(VectorForPoint, a, w, 2.5);
```

The rule is that generic arguments (declared using `:$`) are passed at compile
time, so the actual value of the `impl` argument here can be used to generate
the code for `AddAndScaleGeneric`. So `AddAndScaleGeneric` is using a
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
different names are different, just like two structs with the same definition
but different names are considered different types. For example, lets say we
define another interface, say `LegoFish`, with the same `Add` and `Scale` method
signatures. Implementing `Vector` would not imply an implementation of
`LegoFish`, because the `impl` definition explicitly refers to the name
`Vector`.

An interface's name may be used in a few different contexts:

-   to define [an `impl` for a type](#implementing-interfaces),
-   as a namespace name in [a qualified name](#qualified-member-names), and
-   as a [type-type](terminology.md#type-type) for
    [a generic type parameter](#generics).

While interfaces are examples of type-types, type-types are a more general
concept, for which interfaces are a building block.

## Type-types and facet types

A [type-type](terminology.md#type-type) consists of a set of requirements and a
set of names. Requirements are typically a set of interfaces that a type must
satisfy (though other kinds of requirements are added below). The names are
aliases for qualified names in those interfaces.

An interface is one particularly simple example of a type-type. For example,
`Vector` as a type-type has a set of requirements consisting of the single
interface `Vector`. Its set of names consists of `Add` and `Scale` which are
aliases for the corresponding qualified names inside `Vector` as a namespace.

The requirements determine which types may be cast to a given type-type. The
result of casting a type `T` to a type-type `I` (written `T as I`) is called a
facet type, you might say a facet type `F` is the `I` facet of `T` if `F` is
`T as I`. The API of `F` is determined by the set of names in the type-type.

This general structure of type-types holds not just for interfaces, but others
described in the rest of this document.

## Structural interfaces

If the nominal interfaces discussed above are the building blocks for
type-types, [structural interfaces](terminology.md#structural-interfaces)
describe how they may be composed together. Unlike nominal interfaces, the name
of a structural interface is not a part of its value. Two different structural
interfaces with the same definition are equivalent even if they have different
names. This is because types don't explicitly specify which structural
interfaces they implement, types automatically implement any structural
interfaces they can satisfy.

A structural interface definition can contain interface requirements using
`impl` declarations and names using `alias` declarations. Note that this allows
us to declare the aspects of a type-type directly.

```
structural interface VectorLegoFish {
  // Interface implementation requirements
  impl Vector;
  impl LegoFish;
  // Names
  alias Scale = Vector.Scale;
  alias VAdd = Vector.Add;
  alias LFAdd = LegoFish.Add;
}
```

We don't expect users do directly define many structural interfaces, but other
constructs we do expect them to use will be defined in terms of them. For
example, we can define the Carbon builtin `Type` as:

```
structural interface Type { }
```

That is, `Type` is the type-type with no requirements (so matches every type),
and defines no names.

```
fn Identity[Type:$ T](T: x) -> T {
  // Can accept values of any type. But, since we no nothing about the
  // type, we don't know about any operations on `x` inside this function.
  return x;
}

var Int: i = Identity(3);
var String: s = Identity("string");
```

**Aside:** We can define `auto` as syntactic sugar for `(Type:$$ _)`. This
definition allows you to use `auto` as the type for a local variable whose type
can be statically determined by the compiler. It also allows you to use `auto`
as the type of a function parameter, to mean "accepts a value of any type, and
this function will be instantiated separately for every different type." This is
consistent with the
[use of `auto` in the C++20 Abbreviated function template feature](https://en.cppreference.com/w/cpp/language/function_template#Abbreviated_function_template).

In general we should support the same kinds of declarations in a
`structural interface` definitions as in an `interface`. Generally speaking
declarations in one kind of interface make sense in the other, and there is an
anology between them. If an `interface` `I` has (non-`alias`) declarations `X`,
`Y`, and `Z`, like so:

```
interface I {
  X;
  Y;
  Z;
}
```

(Here, `X` could be something like `method (Self: this) F()`.)

Then a type implementing `I` would have `impl I` with definitions for `X`, `Y`,
and `Z`, as in:

```
struct ImplementsI {
  // ...
  impl I {
    X { ... }
    Y { ... }
    Z { ... }
  }
}
```

But the corresponding `structural interface`, `S`:

```
structural interface S {
  X;
  Y;
  Z;
}
```

would match any type with definitions for `X`, `Y`, and `Z` directly:

```
struct ImplementsS {
  // ...
  X { ... }
  Y { ... }
  Z { ... }
}
```

### Subtyping between type-types

There is a subtyping relationship between type-types that allows you to call one
generic function from another as long as you are calling a function with a
subset of your requirements.

Given a generic type `T` with type-type `I1`, it may be
[implicitly cast](terminology.md#subtyping-and-casting) to a type-type `I2`,
resulting in `T as I2`, as long as the requirements of `I1` are a superset of
the requirements of `I2`. Further, given a value `x` of type `T`, it can be
implicitly cast to `T as I2`. For example:

```
interface Printable { method (Self: this) Print(); }
interface Renderable { method (Self: this) Draw(); }

structural interface PrintAndRender {
  impl Printable;
  impl Renderable;
}
structural interface JustPrint {
  impl Printable;
}

fn PrintIt[JustPrint:$ T2](T2: x2) {
  x2.(Printable.Print)();
}
fn PrintDrawPrint[PrintAndRender:$ T1](T1: x1) {
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

### Future work: method constraints

Structural interfaces are a reasonable mechanism for describing other structural
type constraints, which we will likely want for template constraints. For
example, a method definition in a structural interface would match any type that
has a method with that name and signature. This is only for templates, not
generics, since "the method with a given name and signature" can change when
casting to a facet type. For example:

```
structural interface ShowPrintable {
  impl Printable;
  alias Show = Printable.Print;
}

structural interface ShowRenderable {
  impl Renderable;
  alias Show = Renderable.Draw;
}

structural interface HasShow {
  method (Self: this) Show();
}

// Template, not generic, since this relies on structural typing.
fn CallShow[HasShow:$$ T](T: x) {
  x.Show();
}

fn ViaPrintable[ShowPrintable:$ T](T: x) {
  // Calls Printable.Print().
  CallShow(x);
}

fn ViaRenderable[ShowRenderable:$ T](T: x) {
  // Calls Renderable.Draw().
  CallShow(x);
}

struct Sprite {
  impl Printable { ... }
  impl Renderable { ... }
}

var Sprite: x = ();
ViaPrintable(x);
ViaRenderable(x);
// Not allowed, no method `Show`:
CallShow(x);
```

We could similarly support associated constant and
[instance data field](#field-requirements) requirements. This is future work
though, as it does not directly impact generics in Carbon.

## Combining interfaces by anding type-types

In order to support functions that require more than one interface to be
implemented, we provide a combination operator on type-types, written `&`. This
operator gives the type-type with the union of all the requirements and the
union of the names minus any conflicts.

```
interface Printable {
  method (Self: this) Print();
}
interface Renderable {
  method (Self: this) Center() -> (Int, Int);
  method (Self: this) Draw();
}

// `Printable & Renderable` is syntactic sugar for this type-type:
structural interface {
  impl Printable;
  impl Renderable;
  alias Print = Printable.Print;
  alias Center = Renderable.Center;
  alias Draw = Renderable.Draw;
}

fn PrintThenDraw[Printable & Renderable:$ T](T: x) {
  // Can use methods of `Printable` or `Renderable` on `x` here.
  x.Print();  // Same as `x.(Printable.Print)();`.
  x.Draw();  // Same as `x.(Renderable.Draw)();`.
}

struct Sprite {
  // ...
  impl Printable {
    method (Self: this) Print() { ... }
  }
  impl Renderable {
    method (Self: this) Center() -> (Int, Int) { ... }
    method (Self: this) Draw() { ... }
  }
}

var Sprite: s = ...;
PrintThenDraw(s);
```

Any conflicting names between the two types are replaced with a name that is an
error to use.

```
interface Renderable {
  method (Self: this) Center() -> (Int, Int);
  method (Self: this) Draw();
}
interface EndOfGame {
  method (Self: this) Draw();
  method (Self: this) Winner(Int: player);
}
// `Renderable & EndOfGame` is syntactic sugar for this type-type:
structural interface {
  impl Renderable;
  impl EndOfGame;
  alias Center = Renderable.Center;
  // Open question: `forbidden`, `invalid`, or something else?
  forbidden Draw
    message "Ambiguous, use either `(Renderable.Draw)` or `(EndOfGame.Draw)`.";
  alias Winner = EndOfGame.Winner;
}
```

Conflicts can be resolved at the call site using
[the qualified name syntax](#qualified-member-names), or by defining a
structural interface explicitly and renaming the methods:

```
structural interface RenderableAndEndOfGame {
  impl Renderable;
  impl EndOfGame;
  alias Center = Renderable.Center;
  alias RenderableDraw = Renderable.Draw;
  alias TieGame = EndOfGame.Draw;
  alias Winner = EndOfGame.Winner;
}

fn RenderTieGame[RenderableAndEndOfGame:$ T](T: x) {
  // Calls Renderable.Draw()
  x.RenderableDraw();
  // Calls EndOfGame.Draw()
  x.TieGame();
}
```

Reserving the name when there is a conflict is part of resolving what happens
when you combine more than two type-types. If `x` is forbidden in `A`, it is
forbidden in `A & B`, whether or not `B` defines the name `x`. This makes `&`
associative and commutative, and so it is well defined on sets of interfaces, or
other type-types, independent of order.

Note that we do _not_ consider two type-types using the same name to mean the
same thing to be a conflict. For example, combining a type-type with itself
gives itself, `MyTypeType & MyTypeType == MyTypeType`. Also, given two
[interface extensions](#interface-extension) of a common base interface, the sum
should not conflict on any names in the common base.

**Rejected alternative:** Instead of using `&` as the combining operator, we
considered using `+`,
[like Rust](https://rust-lang.github.io/rfcs/0087-trait-bounds-with-plus.html).
See [#531](https://github.com/carbon-language/carbon-lang/issues/531) for the
discussion.

**Future work:** We may want to define another operator on type-types for adding
requirements to a type-type without affecting the names, and so avoid the
possibility of name conflicts. Note this means the operation is not commutative.
If we call this operator `[&]`, then `A [&] B` has the names of `A` and
`B [&] A` has the names of `B`.

```
// `Printable [&] Renderable` is syntactic sugar for this type-type:
structural interface {
  impl Printable;
  impl Renderable;
  alias Print = Printable.Print;
}

// `Renderable [&] EndOfGame` is syntactic sugar for this type-type:
structural interface {
  impl Renderable;
  impl EndOfGame;
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
similarly to Rust, with the main difference being how you
[qualify names when there is a conflict](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html).

## Interface requiring other interfaces

Some interfaces will depend on other interfaces being implemented for the same
type. For example, in C++,
[the `Container` concept](https://en.cppreference.com/w/cpp/named_req/Container#Other_requirements)
requires all containers to also satisfy the requirements of
`DefaultConstructible`, `CopyConstructible`, `EqualityComparable`, and
`Swappable`. This is already a capability for
[type-types in general](#type-types-and-facet-types). For consistency we will
use the same semantics and syntax as we do for
[structural interfaces](#structural-interfaces):

```
interface Equatable { method (Self: this) Equals(Self: that) -> Bool; }

interface Iterable {
  method (Ptr(Self): this) Advance() -> Bool;
  impl Equatable;
}

def DoAdvanceAndEquals[Iterable:$ T](T: x) {
  // `x` has type `T` that implements `Iterable`, and so has `Advance`.
  x.Advance();
  // `Iterable` requires an implementation of `Equatable`,
  // so `T` also implements `Equatable`.
  x.(Equatable.Equals)(x);
}

struct Iota {
  impl Iterable { method (Self: this) Advance() { ... } }
  impl Equatable { method (Self: this) Equals(Self: that) -> Bool { ... } }
}
var Iota: x;
DoAdvanceAndEquals(x);
```

Like with structural interfaces, an interface implementation requirement doesn't
by itself add any names to the interface, but again those can be added with
`alias` declarations:

```
interface Hashable {
  method (Self: this) Hash() -> UInt64;
  impl Equatable;
  alias Equals = Equatable.Equals;
}

def DoHashAndEquals[Hashable:$ T](T: x) {
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
struct Song {
  impl Hashable {
    method (Self: this) Hash() -> UInt64 { ... }
    method (Self: this) Equals(Self: that) -> Bool { ... }
  }
}
var Song: y;
DoHashAndEquals(y);
```

This allows us to say that `Hashable`
["extends" or "refines"](terminology.md#extendingrefining-an-interface)
`Equatable`, with some benefits:

-   This allows `Equatable` to be an implementation detail of `Hashable`.
-   This allows types implementing `Hashable` to implement all of its API in one
    place.
-   This reduces the boilerplate for types implementing `Hashable`.

We expect this concept to be common enough to warrant dedicated syntax:

```
interface Equatable { method (Self: this) Equals(Self: that) -> Bool; }

interface Hashable {
  extends Equatable;
  method (Self: this) Hash() -> UInt64;
}
// is equivalent to the definition of Hashable from before:
// interface Hashable {
//   impl Equatable;
//   alias Equals = Equatable.Equals;
//   method (Self: this) Hash() -> UInt64;
// }
```

No names in `Hashable` are allowed to conflict with names in `Equatable` (unless
those names are marked as `upcoming` or `deprecated` as in
[evolution future work](#evolution)). Hopefully this won't be a problem in
practice, since interface extension is a very closely coupled relationship, but
this may be something we will have to revisit in the future.

**Concern:** Having both `extends` and [`extend`](#external-impl) with different
meanings is going to be confusing. One should be renamed.

Examples:

-   The C++
    [Boost.Graph library](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/)
    [graph concepts](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/graph_concepts.html#fig:graph-concepts)
    has many refining relationships between concepts.
    [Carbon generics use case: graph library](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?usp=sharing&resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA)
    shows how those concepts might be translated into Carbon interfaces.
-   The [C++ concepts](https://en.cppreference.com/w/cpp/named_req) for
    containers, iterators, and concurrency include many refinement
    relationships.
-   Swift protocols, such as
    [`Collection](https://developer.apple.com/documentation/swift/collection).

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
interface ConvertibleTo(Type:$ T) { ... }

// A type can only implement `PreferredConversion` once.
interface PreferredConversion {
  var Type:$ AssociatedType;
  extends ConvertibleTo(AssociatedType);
}
```

#### `extends` and `impl` with structural interfaces

The `extends` declaration makes sense with the same meaning inside a
[`structural interface`](#structural-interfaces), and should also be supported.

```
interface Media {
  method (Self this) Play();
}
interface Job {
  method (Self this) Run();
}

structural interface Combined {
  extends Media;
  extends Job;
}
```

This definition of `Combined` is equivalent to requiring both the `Media` and
`Job` interfaces being implemented, and aliases their methods.

```
// Equivalent
structural interface Combined {
  impl Media;
  alias Play = Media.Play;
  impl Job;
  alias Run = Job.Run;
}
```

Notice how `Combined` has aliases for all the methods in the interfaces it
requires. That condition is sufficient to allow a type to `impl` the structural
interface:

```
struct Song {
  impl Combined {
    method (Self this) Play() { ... }
    method (Self this) Run() { ... }
  }
}
```

This is equivalent to implementing the required interfaces directly:

```
struct Song {
  impl Media {
    method (Self this) Play() { ... }
  }
  impl Job {
    method (Self this) Run() { ... }
  }
}
```

This is just like you get an implementation of `Equatable` by implementing
`Hashable` when `Hashable` extends `Equatable`. This provides a tool useful for
[evolution](#evolution).

#### Diamond dependency issue

Consider this set of interfaces, simplified from
[this example generic graph library doc](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA#):

```
interface Graph {
  method (Ptr(Self): this) Source(EdgeDescriptor: e) -> VertexDescriptor;
  method (Ptr(Self): this) Target(EdgeDescriptor: e) -> VertexDescriptor;
}

interface IncidenceGraph {
  extends Graph;
  method (Ptr(Self): this) OutEdges(VertexDescriptor: u)
    -> (EdgeIterator, EdgeIterator);
}

interface EdgeListGraph {
  extends Graph;
  method (Ptr(Self): this) Edges() -> (EdgeIterator, EdgeIterator);
}
```

We need to specify what happens how a graph type would implement both
`IncidenceGraph` and `EdgeListGraph`, since both interfaces extend the `Graph`
interface.

```
struct MyEdgeListIncidenceGraph {
  impl IncidenceGraph { ... }
  impl EdgeListGraph { ... }
}
```

The rule is that we need one definition of each method of `Graph`. Each method
though could be defined in the `impl` block of `IncidenceGraph`,
`EdgeListGraph`, or `Graph`. These would all be valid:

-   `IncidenceGraph` implements all methods of `Graph`, `EdgeListGraph`
    implements none of them.

```
struct MyEdgeListIncidenceGraph {
  impl IncidenceGraph {
    method (Self: this) Source(EdgeDescriptor: e) -> VertexDescriptor { ... }
    method (Self: this) Target(EdgeDescriptor: e) -> VertexDescriptor { ... }
    method (Ptr(Self): this) OutEdges(VertexDescriptor: u)
        -> (EdgeIterator, EdgeIterator) { ... }
  }
  impl EdgeListGraph {
    method (Ptr(Self): this) Edges() -> (EdgeIterator, EdgeIterator) { ... }
  }
}
```

-   `IncidenceGraph` and `EdgeListGraph` implement all methods of `Graph`
    between them, but with no overlap.

```
struct MyEdgeListIncidenceGraph {
  impl IncidenceGraph {
    method (Self: this) Source(EdgeDescriptor: e) -> VertexDescriptor { ... }
    method (Ptr(Self): this) OutEdges(VertexDescriptor: u)
        -> (EdgeIterator, EdgeIterator) { ... }
  }
  impl EdgeListGraph {
    method (Self: this) Target(EdgeDescriptor: e) -> VertexDescriptor { ... }
    method (Ptr(Self): this) Edges() -> (EdgeIterator, EdgeIterator) { ... }
  }
}
```

-   We explicitly implement `Graph`.

```
struct MyEdgeListIncidenceGraph {
  impl Graph {
    method (Self: this) Source(EdgeDescriptor: e) -> VertexDescriptor { ... }
    method (Self: this) Target(EdgeDescriptor: e) -> VertexDescriptor { ... }
  }
  impl IncidenceGraph { ... }
  impl EdgeListGraph { ... }
}
```

### Use case: overload resolution

Implementing an extended interface is an example of a more specific match for
[lookup resolution](#lookup-resolution-and-specialization). For example, this
could be used to provide different implementations of an algorithm depending on
the capabilities of the iterator being passed in:

```
interface ForwardIterator {
  var Type:$ Element;
  method (Ptr(Self): this) Advance();
  method (Self: this) Get() -> Element;
}
interface BidirectionalIterator {
  extends ForwardIterator;
  method (Ptr(Self): this) Back();
}
interface RandomAccessIterator {
  extends BidirectionalIterator;
  method (Ptr(Self): this) Skip(Int: offset);
  method (Self: this) Difference(Self: that) -> Int;
}

fn SearchInSortedList
    [Comparable:$ T, ForwardIterator(.Element = T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  ... // does linear search
}
// Will prefer the following overload when it matches
// since it is more specific.
fn SearchInSortedList
    [Comparable:$ T, RandomAccessIterator(.Element = T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  ... // does binary search
}
```

This would be an example of the more general rule that an interface `A`
requiring an implementation of interface `B` means `A` is more specific than
`B`.

## Type compatibility

None of the casts between facet types change the implementation of any
interfaces for a type. So the result of a cast does not depend on the sequence
of casts you perform, just the original type and the final type-type. That is,
these types will all be equal:

-   `T as I`
-   `(T as A) as I`
-   `(((T as A) as B) as C) as I`

Now consider a type with a generic type parameter, like a hash map type:

```
interface Hashable { ... }
struct HashMap(Hashable:$ KeyT, Type:$ ValueT) { ... }
```

If we write something like `HashMap(String, Int)` the type we actually get is:

```
HashMap(String as Hashable, Int as Type)
```

This is the same type we will get if we pass in some other facet types in, so
all of these types are equal:

-   `HashMap(String, Int)`
-   `HashMap(String as Hashable, Int as Type)`
-   `HashMap((String as Printable) as Hashable, Int)`
-   `HashMap((String as Printable & Hashable) as Hashable, Int)`

This means we don't generally need to worry about getting the wrong facet type
as the argument for a generic type. This means we don't get type mismatches when
calling functions as in this example, where the type parameters have different
constraints than the type requires:

```
fn PrintValue
    [Printable & Hashable:$ KeyT, Printable:$ ValueT]
    (HashMap(KeyT, ValueT): map, KeyT: key) { ... }

var HashMap(String, Int): m;
PrintValue(m, "key");
```

## Future work

### Adapting types

Since interfaces may only be implemented for a type once, and we limit where
implementations may be added to a type, there is a need to allow the user to
switch the type of a value to access different interface implementations. See
["adapting a type" in the terminology document](terminology.md#adapting-a-type).

### Associated constants

In addition to associated methods, we will allow other kinds of associated items
associating values with types implementing an interface.

### Associated types

Associated types are associated constants that happen to be types. These are
particularly interesting since they can be used in the signatures of associated
methods or functions, to allow the signatures of methods to vary from
implementation to implementation.

### Parameterized interfaces

Associated types don't change the fact that a type can only implement an
interface at most once. If instead you want a family of related interfaces, each
of which could be implemented for a given type, you could use parameterized
interfaces instead.

#### Impl lookup

We will have rules limiting where interface implementations are defined for
coherence.

### Constraints

We will need to be able to express constraints beyond "type implements these
interfaces."

### Conditional conformance

[The problem](terminology.md#conditional-conformance) we are trying to solve
here is expressing that we have an impl of some interface for some type, but
only if some additional type restrictions are met.

### Parameterized impls

Also known as "blanket `impl`s", these are when you have an `impl` definition
that is parameterized so it applies to more than a single type and interface
combination.

#### Lookup resolution and specialization

For this to work, we need a rule that picks a single `impl` in the case where
there are multiple `impl` definitions that match a particular type and interface
combination.

### Other constraints as type-types

There are some constraints that we will naturally represent as named type-types
that the user can specify.

#### Sized types and type-types

Like Rust, we may have types that have values whose size is only determined at
runtime. Many functions may want to restrict to types with known size.

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

This lets you return am anonymous type implementing an interface from a
function.
[Rust has this feature](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).

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

### Testing

The idea is that you would write tests alongside an interface that validate the
expected behavior of any type implementing that interface.

### Operator overloading

We will need a story for defining how an operation is overloaded for a type by
implementing an interface for that type.

### Impls with state

A feature we might consider where an impl itself can have state.

### Generic associated types and higher-ranked types

This would be some way to express the requirement that there is a way to go from
a type to an implementation of an interface parameterized by that type.

#### Generic associated types

Generic associated types are about when this is a requirement of an interface.

#### Higher-ranked types

Higher-ranked types are used to represent this requirement in a function
signature.

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

### Reverse generics for return types

In Rust this is
[return type of "`impl Trait`"](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).
In Swift,
[this feature is in discussion](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--reverse-generics).
Swift is considering spelling this `<V: Collection> V` or `some Collection`.
