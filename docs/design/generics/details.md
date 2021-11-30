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
    -   [Qualified member names](#qualified-member-names)
-   [Generics](#generics)
    -   [Model](#model)
-   [Interfaces recap](#interfaces-recap)
-   [Type-types and facet types](#type-types-and-facet-types)
-   [Structural interfaces](#structural-interfaces)
    -   [Subtyping between type-types](#subtyping-between-type-types)
-   [Combining interfaces by anding type-types](#combining-interfaces-by-anding-type-types)
-   [Interface requiring other interfaces](#interface-requiring-other-interfaces)
    -   [Interface extension](#interface-extension)
        -   [`extends` and `impl` with structural interfaces](#extends-and-impl-with-structural-interfaces)
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
    -   [Associated functions](#associated-functions)
-   [Associated types](#associated-types)
    -   [Inferring associated types](#inferring-associated-types)
    -   [Model](#model-1)
-   [Parameterized interfaces](#parameterized-interfaces)
    -   [Impl lookup](#impl-lookup)
    -   [Parameterized structural interfaces](#parameterized-structural-interfaces)
-   [Constraints](#constraints)
    -   [Contexts where you might need constraints](#contexts-where-you-might-need-constraints)
    -   [Two approaches for expressing constraints](#two-approaches-for-expressing-constraints)
        -   [Where clauses](#where-clauses)
        -   [Argument passing](#argument-passing)
    -   [Constraint use cases](#constraint-use-cases)
        -   [Set to a specific value](#set-to-a-specific-value)
            -   [Associated constants](#associated-constants-1)
            -   [Associated types](#associated-types-1)
            -   [Concern](#concern)
        -   [Range constraints on associated constants](#range-constraints-on-associated-constants)
        -   [Type bounds](#type-bounds)
            -   [Type bounds on associated types in declarations](#type-bounds-on-associated-types-in-declarations)
            -   [Type bounds on associated types in interfaces](#type-bounds-on-associated-types-in-interfaces)
            -   [Naming type bound constraints](#naming-type-bound-constraints)
            -   [Type bound with interface argument](#type-bound-with-interface-argument)
        -   [Same type constraints](#same-type-constraints)
            -   [Naming same type constraints](#naming-same-type-constraints)
        -   [Combining constraints](#combining-constraints)
        -   [Rejected alternative: `ForSome(F)`](#rejected-alternative-forsomef)
        -   [Is a subtype](#is-a-subtype)
        -   [Parameterized type implements interface](#parameterized-type-implements-interface)
        -   [Recursive constraints](#recursive-constraints)
        -   [Type inequality](#type-inequality)
    -   [Implicit constraints](#implicit-constraints)
    -   [Covariant extension](#covariant-extension)
    -   [Generic type equality](#generic-type-equality)
        -   [Type equality with where clauses](#type-equality-with-where-clauses)
        -   [Type equality with argument passing](#type-equality-with-argument-passing)
            -   [Canonical types and type checking](#canonical-types-and-type-checking)
        -   [Restricted where clauses](#restricted-where-clauses)
        -   [Manual type equality](#manual-type-equality)
    -   [Deleted content](#deleted-content)
        -   [Normalized form](#normalized-form)
        -   [Recursive constraints](#recursive-constraints-1)
        -   [Terminating recursion](#terminating-recursion)
    -   [Options](#options)
-   [Parameterized impls](#parameterized-impls)
    -   [Conditional conformance](#conditional-conformance)
    -   [Bridge for C++ templates](#bridge-for-c-templates)
        -   [Calling C++ template code from Carbon](#calling-c-template-code-from-carbon)
        -   [Moving a C++ template to Carbon](#moving-a-c-template-to-carbon)
    -   [Subtlety around interfaces with parameters](#subtlety-around-interfaces-with-parameters)
    -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
-   [Other constraints as type-of-types](#other-constraints-as-type-of-types)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Same implementation restriction](#same-implementation-restriction)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other impls](#example-creating-an-impl-out-of-other-impls)
    -   [Type facet of another type](#type-facet-of-another-type)
    -   [Sized types and type-of-types](#sized-types-and-type-of-types)
        -   [Model](#model-2)
    -   [`TypeId`](#typeid)
-   [Dynamic types](#dynamic-types)
    -   [Runtime type parameters](#runtime-type-parameters)
    -   [Runtime type fields](#runtime-type-fields)
        -   [Dynamic pointer type](#dynamic-pointer-type)
            -   [Restrictions](#restrictions)
            -   [Model](#model-3)
        -   [Deref](#deref)
        -   [Boxed](#boxed)
        -   [DynBoxed](#dynboxed)
        -   [MaybeBoxed](#maybeboxed)
-   [Compiler-controlled dispatch strategy](#compiler-controlled-dispatch-strategy)
    -   [No address of generic functions](#no-address-of-generic-functions)
    -   [Static local variables](#static-local-variables)
-   [Future work](#future-work)
    -   [Abstract return types](#abstract-return-types)
    -   [Interface defaults](#interface-defaults)
    -   [Interface evolution](#interface-evolution)
    -   [Evolution from templates to generics](#evolution-from-templates-to-generics)
    -   [Testing](#testing)
    -   [Operator overloading](#operator-overloading)
    -   [Impls with state](#impls-with-state)
    -   [Generic associated types](#generic-associated-types)
    -   [Higher-ranked types](#higher-ranked-types)
    -   [Field requirements](#field-requirements)
    -   [Generic type specialization](#generic-type-specialization)
    -   [Bridge for C++ customization points](#bridge-for-c-customization-points)
    -   [Variadic arguments](#variadic-arguments)
    -   [Interaction with inheritance](#interaction-with-inheritance)
-   [Notes](#notes)
    -   [Other dynamic types](#other-dynamic-types)
-   [Broken links footnote](#broken-links-footnote)

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
compile. FIXME: You are still allowed to declare templated type arguments as
having an interface type, and this will add a requirement that the type satisfy
the interface independent of whether that is needed to compile the function
body, but it is strictly optional. You might still do this to get clearer error
messages, document expectations, or express that a type has certain semantics
beyond what is captured in its member function names and signatures).

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
-   Carbon will implicitly cast values from type `Song` to type
    `Song as ConvertibleToString` when calling a function that can only accept
    types of type `ConvertibleToString`.
-   Typically, `Song` would also have definitions for the methods of
    `ConvertibleToString`, such as `ToString`, unless the implementation of
    `ConvertibleToString` for `Song` was defined as `external`.
-   You may access the `ToString` function for a `Song` value `w` by writing a
    _qualified_ function call, like `w.(ConvertibleToString.ToString)()`. This
    qualified syntax is available whether or not the implementation is defined
    as `external`.
-   If other interfaces are implemented for `Song`, they are also implemented
    for `Song as ConvertibleToString` as well. The only thing that changes when
    casting a `Song` `w` to `Song as ConvertibleToString` are the names that are
    accessible without using the qualification syntax.

We define these facet types (alternatively, interface implementations) either
with the type, with the interface, or somewhere else where Carbon can be
guaranteed to see when needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

If `Song` doesn't implement an interface or we would like to use a different
implementation of that interface, we can define another type that also has the
same data representation as `Song` that has whatever different interface
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
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: Double) -> Self;
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
  var x: Double;
  var y: Double;
  impl Vector {
    // In this scope, "Self" is an alias for "Point".
    fn Add[me: Self](b: Self) -> Self {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    }
    fn Scale[me: Self](v: Double) -> Self {
      return Point(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

Interfaces that are implemented inline contribute to the type's API:

```
var p1: Point = (.x = 1.0, .y = 2.0);
var p2: Point = (.x = 2.0, .y = 4.0);
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
var a: Point = (.x = 1.0, .y = 2.0);
// `a` has `Add` and `Scale` methods:
a.Add(a.Scale(2.0));

// Cast from Point implicitly
var b: Point as Vector = a;
// `b` has `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

// Will also implicitly cast when calling functions:
fn F(c: Point as Vector, d: Point) {
  d.Add(c.Scale(2.0));
}
F(a, b);

// Explicit casts
var z: Point as Vector = (a as (Point as Vector)).Scale(3.0);
z.Add(b);
var w: Point = z as Point;
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
  var x: Double;
  var y: Double;
  impl Vector {
    fn Add[me: Self](b: Self) -> Self { ... }
    fn Scale[me: Self](v: Double) -> Self { ... }
  }
  impl Drawable {
    fn Draw[me: Self]() { ... }
  }
}
```

In this case, all the functions `Add`, `Scale`, and `Draw` end up a part of the
API for `Point`. This means you can't implement two interfaces that have a name
in common.

```
struct GameBoard {
  impl Drawable {
    fn Draw[me: Self]() { ... }
  }
  impl EndOfGame {
    // Error: `GameBoard` has two methods named
    // `Draw` with the same signature.
    fn Draw[me: Self]() { ... }
    fn Winner[me: Self](player: Int) { ... }
  }
}
```

**Open question:** Should we have some syntax for the case where you want both
names to be given the same implementation? It seems like that might be a common
case, but we won't really know if this is an important case until we get more
experience.

```
struct Player {
  var name: String;
  impl Icon {
    fn Name[me: Self]() -> String { return this.name; }
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
  var x: Double;
  var y: Double;
}

extend Point2 {
  // In this scope, "Self" is an alias for "Point2".
  impl Vector {
    fn Add[me: Self](b: Self) -> Self {
      return Point2(.x = a.x + b.x, .y = a.y + b.y);
    }
    fn Scale[me: Self](v: Double) -> Self {
      return Point2(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

**Note:** The external interface implementation syntax was decided in
[question-for-leads issue #575](https://github.com/carbon-language/carbon-lang/issues/575).
Additional alternatives considered are discussed in
[an appendix](appendix-impl-syntax.md).

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
var a: Point2 = (.x = 1.0, .y = 2.0);
// `a` does *not* have `Add` and `Scale` methods:
// Error: a.Add(a.Scale(2.0));

// Cast from Point2 implicitly
var b: Point2 as Vector = a;
// `b` does have `Add` and `Scale` methods:
b.Add(b.Scale(2.0));

fn F(c: Point2 as Vector) {
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
  var x: Double;
  var y: Double;
  fn Add[me: Self](b: Self) -> Self {
    return Point3(.x = a.x + b.x, .y = a.y + b.y);
  }
}

extend Point3 {
  impl Vector {
    alias Add = Point3.Add;  // Syntax TBD
    fn Scale[me: Self](v: Double) -> Self {
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

### Qualified member names

Given a value of type `Point2` and an interface `Vector` implemented for that
type, you can access the methods from that interface using the member's
_qualified name_, whether or not the implementation is done externally with an
`extend` statement:

```
var p1: Point2 = (.x = 1.0, .y = 2.0);
var p2: Point2 = (.x = 2.0, .y = 4.0);
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

var p: Points.Point2 = (.x = 1.0, .y = 2.0);
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
      Point as Vector: a, b: Point as Vector, s: Double)
      -> Point as Vector {
  return a.Add(b).Scale(s);
}
// May still be called with Point arguments, due to implicit casts.
// Similarly the return value can be implicitly cast to a Point.
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
`extend` statement, such as `Point2`, the situation is different:

```
fn AddAndScaleForPoint2(a: Point2, b: Point2, s: Double) -> Point2 {
  // ERROR: `Point2` doesn't have `Add` or `Scale` methods.
  return a.Add(b).Scale(s);
}
```

Even though `Point2` doesn't have `Add` and `Scale` methods, it still implements
`Vector` and so can still call `AddAndScaleGeneric`:

```
var a2: Point2 = (.x = 1.0, .y = 2.0);
var w2: Point2 = (.x = 3.0, .y = 4.0);
var v3: Point2 = AddAndScaleGeneric(a, w, 2.5);
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
    (`fn Foo[InterfaceName:! T](...)`) to have an actual argument with type
    determined by the interface, and supplied at the callsite using a value
    determined by the impl.

For the example above, [the Vector interface](#interfaces) could be thought of
defining a witness table type like:

```
struct Vector {
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
var VectorForPoint: Vector  = (
    .Self = Point,
    // `lambda` is **placeholder** syntax for defining a
    // function value.
    .Add = lambda(a: Point, b: Point) -> Point {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    },
    .Scale = lambda(a: Point, v: Double) -> Point {
      return Point(.x = a.x * v, .y = a.y * v);
    },
);
```

Finally we can define a generic function and call it, like
[`AddAndScaleGeneric` from the "Generics" section](#generics) by making the
witness table an explicit argument to the function:

```
fn AddAndScaleGeneric
    (impl:! Vector, a: impl.Self, b: impl.Self, s: Double) -> impl.Self {
  return impl.Scale(impl.Add(a, b), s);
}
// Point implements Vector.
var v: Point = AddAndScaleGeneric(VectorForPoint, a, w, 2.5);
```

The rule is that generic arguments (declared using `:!`) are passed at compile
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
fn Identity[T:! Type](x: T) -> T {
  // Can accept values of any type. But, since we no nothing about the
  // type, we don't know about any operations on `x` inside this function.
  return x;
}

var i: Int = Identity(3);
var s: String = Identity("string");
```

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
interface Printable { fn Print[me: Self](); }
interface Renderable { fn Draw[me: Self](); }

structural interface PrintAndRender {
  impl Printable;
  impl Renderable;
}
structural interface JustPrint {
  impl Printable;
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

## Combining interfaces by anding type-types

In order to support functions that require more than one interface to be
implemented, we provide a combination operator on type-types, written `&`. This
operator gives the type-type with the union of all the requirements and the
union of the names minus any conflicts.

```
interface Printable {
  fn Print[me: Self]();
}
interface Renderable {
  fn Center[me: Self]() -> (Int, Int);
  fn Draw[me: Self]();
}

// `Printable & Renderable` is syntactic sugar for this type-type:
structural interface {
  impl Printable;
  impl Renderable;
  alias Print = Printable.Print;
  alias Center = Renderable.Center;
  alias Draw = Renderable.Draw;
}

fn PrintThenDraw[T:! Printable & Renderable](x: T) {
  // Can use methods of `Printable` or `Renderable` on `x` here.
  x.Print();  // Same as `x.(Printable.Print)();`.
  x.Draw();  // Same as `x.(Renderable.Draw)();`.
}

struct Sprite {
  // ...
  impl Printable {
    fn Print[me: Self]() { ... }
  }
  impl Renderable {
    fn Center[me: Self]() -> (Int, Int) { ... }
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
  fn Center[me: Self]() -> (Int, Int);
  fn Draw[me: Self]();
}
interface EndOfGame {
  fn Draw[me: Self]();
  fn Winner[me: Self](player: Int);
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

fn RenderTieGame[T:! RenderableAndEndOfGame](x: T) {
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
interface Equatable { fn Equals[me: Self](that: Self) -> Bool; }

interface Iterable {
  fn Advance[addr me: Self*]() -> Bool;
  impl Equatable;
}

def DoAdvanceAndEquals[T:! Iterable](x: T) {
  // `x` has type `T` that implements `Iterable`, and so has `Advance`.
  x.Advance();
  // `Iterable` requires an implementation of `Equatable`,
  // so `T` also implements `Equatable`.
  x.(Equatable.Equals)(x);
}

struct Iota {
  impl Iterable { fn Advance[me: Self]() { ... } }
  impl Equatable { fn Equals[me: Self](that: Self) -> Bool { ... } }
}
var x: Iota;
DoAdvanceAndEquals(x);
```

Like with structural interfaces, an interface implementation requirement doesn't
by itself add any names to the interface, but again those can be added with
`alias` declarations:

```
interface Hashable {
  fn Hash[me: Self]() -> UInt64;
  impl Equatable;
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
struct Song {
  impl Hashable {
    fn Hash[me: Self]() -> UInt64 { ... }
    fn Equals[me: Self](that: Self) -> Bool { ... }
  }
}
var y: Song;
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
interface Equatable { fn Equals[me: Self](that: Self) -> Bool; }

interface Hashable {
  extends Equatable;
  fn Hash[me: Self]() -> UInt64;
}
// is equivalent to the definition of Hashable from before:
// interface Hashable {
//   impl Equatable;
//   alias Equals = Equatable.Equals;
//   fn Hash[me: Self]() -> UInt64;
// }
```

No names in `Hashable` are allowed to conflict with names in `Equatable` (unless
those names are marked as `upcoming` or `deprecated` as in
[evolution future work](#interface-evolution)). Hopefully this won't be a
problem in practice, since interface extension is a very closely coupled
relationship, but this may be something we will have to revisit in the future.

**Concern:** Having both `extends` and [`extend`](#external-impl) with different
meanings is going to be confusing. One should be renamed. Perhaps `extends`
would be better replaced by `provides`?

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
interface ConvertibleTo(T:! Type) { ... }

// A type can only implement `PreferredConversion` once.
interface PreferredConversion {
  leet AssociatedType: Type;
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
[evolution](#interface-evolution).

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
    fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    fn OutEdges[addr me: Self*](u: VertexDescriptor)
        -> (EdgeIterator, EdgeIterator) { ... }
  }
  impl EdgeListGraph {
    fn Edges[addr me: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
  }
}
```

-   `IncidenceGraph` and `EdgeListGraph` implement all methods of `Graph`
    between them, but with no overlap.

```
struct MyEdgeListIncidenceGraph {
  impl IncidenceGraph {
    fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    fn OutEdges[addr me: Self*](u: VertexDescriptor)
        -> (EdgeIterator, EdgeIterator) { ... }
  }
  impl EdgeListGraph {
    fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    fn Edges[addr me: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
  }
}
```

-   We explicitly implement `Graph`.

```
struct MyEdgeListIncidenceGraph {
  impl Graph {
    fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
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
  let Element:! Type;
  fn Advance[addr me: Self*]();
  fn Get[me: Self]() -> Element;
}
interface BidirectionalIterator {
  extends ForwardIterator;
  fn Back[addr me: Self*]();
}
interface RandomAccessIterator {
  extends BidirectionalIterator;
  fn Skip[addr me: Self*](offset: Int);
  fn Difference[me: Self](that: Self) -> Int;
}

fn SearchInSortedList
    [T:! Comparable, IterT:! ForwardIterator(.Element = T)]
    (begin: IterT, end: IterT, needle: T) -> Bool {
  ... // does linear search
}
// Will prefer the following overload when it matches
// since it is more specific.
fn SearchInSortedList
    [T:! Comparable, IterT:! RandomAccessIterator(.Element = T)]
    (begin: IterT, end: IterT, needle: T) -> Bool {
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
struct HashMap(KeyT:! Hashable, ValueT:! Type) { ... }
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
    [KeyT:! Printable & Hashable, ValueT:! Printable]
    (map: HashMap(KeyT, ValueT), key: KeyT) { ... }

var m: HashMap(String, Int);
PrintValue(m, "key");
```

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
  fn Less[me: Self](that: Self) -> Bool;
}
class Song {
  impl as Printable { fn Print[me: Self]() { ... } }
}
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> Bool { ... }
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

-   You may only add APIs, not change the representation of the type, unlike
    extending a type where you may add fields.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `Song`, `SongByTitle`, `FormattedSong`,
    and `FormattedSongByTitle` end up compatible with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit casting between these
    types (unlike between a type and one of its facet types / impls).
-   For the purposes of generics, we only need to support adding interface
    implementations. But this `adapter` feature could be used more generally,
    for example to add methods as well.

Inside an adapter, the `Self` type matches the adapter. Members of the original
type may be accessed like any other facet type; either by a cast:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> Bool {
      return (this as Song).Title() < (that as Song).Title();
    }
  }
}
```

or using qualified names:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> Bool {
      return this.(Song.Title)() < that(Song.Title)();
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
`Printable` implementation?

**Comparison with other languages:** This matches the Rust idiom called
"newtype", which is used to implement traits on types while avoiding coherence
problems, see
[here](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-the-newtype-pattern-to-implement-external-traits-on-external-types)
and
[here](https://github.com/Ixrec/rust-orphan-rules#user-content-why-are-the-orphan-rules-controversial).
Rust's mechanism doesn't directly support reusing implementations, though some
of that is provided by macros defined in libraries.

### Adapter compatibility

The framework from the [type compatibility section](#type-compatibility) allows
us to evaluate when we can cast between two different arguments to a
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
one difference between them is that `Song as Hashable` may be implicitly cast to
`Song`, which implements interface `Printable`, and `PlayableSong as Hashable`
may be implicilty cast to `PlayableSong`, which implements interface `Media`.
This means that it is safe to cast between
`HashMap(Song, Int) == HashMap(Song as Hashable, Int)` and
`HashMap(PlayableSong, Int) == HashMap(PlayableSong as Hashable, Int)` (though
maybe only with an explicit cast) but
`HashMap(SongHashedByTitle, Int) == Hashmap(SongHashByTitle as Hashable, Int)`
is incompatible. This is a relief, because we know that in practice the
invariants of a `HashMap` implementation rely on the hashing function staying
the same.

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

The result would be `SongByArtist` would:

-   implement `Comparable`, unlike `Song`,
-   implement `Hashable`, but differently than `Song`, and
-   implement `Printable`, inherited from `Song`.

**Open question:** We may need additional mechanisms for changing the API in the
adapter. For example, to resolve conflicts we might want to be able to move the
implementation of a specific interface into an [external impl](#external-impl).

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

The caller can either cast `SongLib.Song` values to `Song` when calling
`CompareLib.Sort` or just start with `Song` values in the first place.

```
var lib_song: SongLib.Song = ...;
CompareLib.Sort((lib_song as Song,));

var song: Song = ...;
CompareLib.Sort((song,));
```

### Adapter with stricter invariants

**Open question:** Rust also uses the newtype idiom to create types with
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
  fn Less[me: Self](that: Self) -> Bool;
}
adapter ComparableFromDifferenceFn
    (T:! Type, Difference:! fnty(T, T)->Int) for T {
  impl as Comparable {
    fn Less[me: Self](that: Self) -> Bool {
      return Difference(this, that) < 0;
    }
  }
}
class IntWrapper {
  var x: Int;
  fn Difference(this: Self, that: Self) {
    return that.x - this.x;
  }
  impl as Comparable =
      ComparableFromDifferenceFn(IntWrapper, Difference)
      as Comparable;
}
```

## Associated constants

In addition to associated methods, we allow other kinds of associated items. For
consistency, we use the same syntax to describe a constant in an interface as in
a type without assigning a value. As constants, they are declared using the
`let` introducer. For example, a fixed-dimensional point type could have the
dimension as an associated constant.

```
interface NSpacePoint {
  let N:! Int;
  // The following require: 0 <= i < N.
  fn Get[addr me: Self*](i: Int) -> Float64;
  fn Set[addr me: Self*](i: Int, value: Float64);
  // Associated constants may be used in signatures:
  fn SetAll[addr me: Self*](value: Array(Float64, N));
}
```

**Note:**
[Question-for-leads issue #565](https://github.com/carbon-language/carbon-lang/issues/565)
discussed but did not decide the syntax for associated constants.

Implementations of `NSpacePoint` for different types might have different values
for `N`:

```
class Point2D {
  impl as NSpacePoint {
    let N:! Int = 2;
    fn Get[addr me: Self*](i: Int) -> Float64 { ... }
    fn Set[addr me: Self*](i: Int, value: Float64) { ... }
    fn SetAll[addr me: Self*](value: Array(Float64, 2)) { ... }
  }
}

class Point3D {
  impl as NSpacePoint {
    let N:! Int = 3;
    fn Get[addr me: Self*](i: Int) -> Float64 { ... }
    fn Set[addr me: Self*](i: Int, value: Float64) { ... }
    fn SetAll[addr me: Self*](value: Array(Float64, 3)) { ... }
  }
}
```

And these values may be accessed as members of the type:

```
Assert(Point2D.N == 2);
Assert(Point3D.N == 3);

fn PrintPoint[PointT:! NSpacePoint](p: PointT) {
  for (var i: Int = 0; i < PointT.N; ++i) {
    if (i > 0) { Print(", "); }
    Print(p.Get(i));
  }
}

fn ExtractPoint[PointT:! NSpacePoint](
    p: PointT,
    dest: Array(Float64, PointT.N)*) {
  for (var i: Int = 0; i < PointT.N; ++i) {
    (*dest)[i] = p.Get(i);
  }
}
```

**Comparison with other languages:** This feature is also called
[associated constants in Rust](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants).

**Aside:** In general, `let` fields will only have compile-time and not runtime
storage associated with them.

### Associated functions

To be consistent with normal function declaration syntax, function constants are
written:

```
interface DeserializeFromString {
  fn Deserialize(serialized: String) -> Self;
}

class MySerializableType {
  var i: Int;

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

## Associated types

Associated types are associated constants that happen to be types. These are
particularly interesting since they can be used in the signatures of associated
methods or functions, to allow the signatures of methods to vary from
implementation to implementation. We already have one example of this: the
`Self` type discussed [above in the "Interfaces" section](#interfaces). For
other cases, we can say that the interface declares that each implementation
will provide a type under a specific name. For example:

```
interface StackAssociatedType {
  let ElementType:! Type;
  fn Push[addr me: Self*](value: ElementType);
  fn Pop[addr me: Self*]() -> ElementType;
  fn IsEmpty[addr me: Self*]() -> Bool;
}
```

Here we have an interface called `StackAssociatedType` which defines two
methods, `Push` and `Pop`. The signatures of those two methods declared as
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
    // Note: open syntax question
    let ElementType = T;
    fn Push[addr me: Self*](value: ElementType) {
      this->Insert(this->End(), value);
    }
    fn Pop[addr me: Self*]() -> ElementType {
      var pos: IteratorType = this->End();
      Assert(pos != this->Begin());
      --pos;
      var ret: ElementType = *pos;
      this->Remove(pos);
      return ret;
    }
    fn IsEmpty[addr me: Self*]() -> Bool {
      return this->Begin() == this->End();
    }
  }
}
```

**Open question:** What syntax should `DynamicArray(T)` use to specify the value
of `StackAssociatedType.ElementType`?

```
let ElementType = T;
```

or:

```
let ElementType:! Type = T;
```

The definition of the `StackAssociatedType` is sufficient for writing a generic
function that operates on anything implementing that interface, for example:

```
fn PeekAtTopOfStack[StackType:! StackAssociatedType](s: StackType*)
    -> StackType.ElementType {
  var top: StackType.ElementType = s->Pop();
  s->Push(top);
  return top;
}

var my_array: DynamicArray(Int) = (1, 2, 3);
// PeekAtTopOfStack's `StackType` is set to
// `DynamicArray(Int) as StackAssociatedType`.
// `StackType.ElementType` becomes `Int`.
Assert(PeekAtTopOfStack(my_array) == 3);
```

For context, see
["Interface type parameters versus associated types" in the generics terminology document](terminology.md#interface-type-parameters-versus-associated-types).

**Comparison with other languages:** Both
[Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189)
support associated types.

### Inferring associated types

**Open question:**
[Swift allows the value of an associated type to be omitted when it can be determined from the method signatures in the implementation](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID190).
For the above example, this would mean figuring out `ElementType == T` from
context:

```
class DynamicArray(T:! Type) {
  // ...

  impl as StackAssociatedType {
    // Not needed: let ElementType = T;
    fn Push[addr me: Self*](value: T) { ... }
    fn Pop[addr me: Self*]() -> T { ... }
    fn IsEmpty[addr me: Self*]() -> Bool { ... }
  }
}
```

Should we do the same thing in Carbon?

One benefit is that it allows an interface to evolve by adding an associated
type, without having to then modify all implementations of that interface.

One concern is this might be a little more complicated in the presence of method
overloads with [default implementations](interface-defaults), since it might not
be clear how they should match up, as in this example:

```
interface Has2OverloadsWithDefaults {
  let T:! StackAssociatedType;
  fn F[me: Self](x: DynamicArray(T), y: T) { ... }
  fn F[me: Self](x: T, y: T.ElementType) { ... }
}

class S {
  impl as Has2OverloadsWithDefaults {
     // Unclear if T == DynamicArray(Int) or
     // T == DynamicArray(DynamicArray(Int)).
     fn F[me: Self](
         x: DynamicArray(DynamicArray(Int)),
         y: DynamicArray(Int)) { ... }
  }
}
```

Not to say this can't be resolved, but it does add complexity.
[Swift considered](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#associated-type-inference)
removing this feature because it was the one thing in Swift that required global
type inference, which they otherwise avoided. They
[ultimately decided to keep the feature](https://github.com/apple/swift-evolution/blob/main/proposals/0108-remove-assoctype-inference.md).

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
  var Begin: fnty (this: Self*) -> IteratorType
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
  fn IsEmpty[addr me: Self*]() -> Bool;
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
    fn IsEmpty[addr me: Self*]() -> Bool {
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
    fn IsEmpty[addr me: Self*]() -> Bool {
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
// Error: can't deduce interface parameter `T`.
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
    (s: StackType*, _: type_of(T)) -> T { ... }

var produce: Produce = ...;
var top_fruit: Fruit =
    PeekAtTopOfStackParameterized(&produce, Fruit);
var top_veggie: Veggie =
    PeekAtTopOfStackParameterized(&produce, Veggie);
```

The pattern `_: type_of(T)` will only match `T` since `T` is a type so
`type_of(T)` returns a single-value type-of-type. Using that pattern in the
explicit parameter list allows us to make `T` available earlier in the
declaration so it can be passed as the argument to the parameterized interface
`StackParameterized`.

**Open question:** Perhaps `type_of` should be spelled `singleton_type_of` or
`single_value_type_of`, and only take a type argument?

> **Alternative considered:** We could also allow value patterns without a `:`,
> as in:
>
> ```
> fn PeekAtTopOfStackParameterized
>     [T:! Type, StackType:! StackParameterized(T)]
>     (s: StackType*, T) -> T { ... }
> ```
>
> However, we don't want to allow value patterns more generally so we can reject
> declarations like `fn F(Int)` when users almost certainly meant
> `fn F(i: Int)`.

This approach is useful for the `ComparableTo(T)` interface, where a type might
be comparable with multiple other types, and in fact interfaces for
[operator overloads](#operator-overloading) more generally. Example:

```
interface EquatableWith(T:! Type) {
  fn Equals[me: Self](that: T) -> Bool;
  ...
}
class Complex {
  var real: Float64;
  var imag: Float64;
  // Can implement this interface more than once as long as it has different
  // arguments.
  impl as EquatableWith(Complex) { ... }
  impl as EquatableWith(Float64) { ... }
}
```

All interface parameters must be marked as "generic", using the `:!` syntax.
This reflects these two properties of these parameters:

-   They must be resolved at compile-time, and so can't be passed regular
    dynamic values.
-   We allow either generic or template values to be passed in.

**Context:** See
[type parameters for interfaces](terminology.md#interface-type-parameters-versus-associated-types)
in the terminology doc.

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
// Error: Bijection has two impls of interface Map(String, String)
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
both interface type parameters and associated types. The difference is that
interface parameters are "inputs" since they _determine_ which `impl` to use,
and associated types are "outputs" since they are determined _by_ the `impl`,
but play no role in selecting the `impl`.

**Rejected alternative:** We considered and then rejected the idea that we would
have two kinds of parameters. "Multi" parameters would work as described above.
"Deducible" type parameters would only allow one implementation of an interface,
not one per interface & type parameter combination. These deducible type
parameters could be inferred like associated types are. See
[the detailed rationale](appendix-deduced-interface-params.md)

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
[defined in terminology](terminology.md#coherence), is
[a goal for Carbon](goals.md#coherence). More detail can be found in
[this appendix with the rationale and alternatives considered](appendix-coherence.md).

### Parameterized structural interfaces

We should also allow the [structural interface](#structural-interfaces)
construct to support parameters. Parameters would work the same way as for
regular, that is nominal or non-structural, interfaces.

## Constraints

TODO: Fix this up a lot

### Contexts where you might need constraints

-   In a declaration of a function, type, interface, or impl.
-   Within the body of an interface definition.
-   Naming a new type-of-type that represents the constraint (typically a `let`
    or `structural interface` definition).

To handle this last use case, we expand the kinds of requirements that
type-of-types can have from just interface requirements to also include the
various kinds of constraints discussed later in this section.

### Two approaches for expressing constraints

**Open question:** It is undecided which approach we will go with. We may also
decide to go with a combination of these approaches.

#### Where clauses

This approach is to specify constraints using boolean expressions that are
required to evaluate to true. These expressions come after the thing they
constrain since they are written in terms of those names.

The keyword to introduce these constraints could alternatively be spelled
`requires` or `if`, but Swift
([1](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID553),
[2](https://docs.swift.org/swift-book/ReferenceManual/GenericParametersAndArguments.html#ID408))
and Rust ([1](https://doc.rust-lang.org/rust-by-example/generics/where.html),
[2](https://doc.rust-lang.org/book/ch10-02-traits.html#clearer-trait-bounds-with-where-clauses))
both use `where`. Note Swift also uses `where` to write
[filter conditions evaluated at runtime](https://medium.com/@shubhamkaliyar255/how-to-use-where-clause-in-for-in-loops-e61d0860debe),
so far Carbon has been using `if` in those locations.

We would attach `where` clauses to individual declarations, following Rust, as
in:

```
// Constraints on function parameters:
fn F[V:! D](v: V) where ... { ... }

// Constraints on a type parameter:
class S(T:! B) where ... { ... }

// Constraints on an interface parameter:
interface A(T:! B) where ... {
  // Constraints on an associated type or constant:
  let U:! C where ...;
  // Constraints on a method:
  fn G[me: Self, V:! D](v: V) where ...;
}
```

If there are multiple constraints on a single declaration, those constraints are
separated by commas (`,`), following both Rust and Swift.

Advantages:

-   Can express the full range of constraints.
-   Works uniformly across contexts.
-   Familiar to users of Rust and Swift.
-   Keeps the definition of constraints after information that readers generally
    care about more.

Disadvantages:

-   Given the full generality of constraints expressible using this syntax,
    determining canonical types or type equality in a generic context
    [becomes undecidable](#type-equality-with-where-clauses). This means that
    either the syntax is restricted heavily, or algorithms become heuristic and
    potentially slow. Further, the boundary between acceptable code and rejected
    code becomes fuzzy and unpredictable.
-   Awkward to produce type-of-types that have specified values for all
    associated types for use with `DynPtr` and `DynBox`.
-   Some inconsistency with how interface parameters are specified.
-   Adds a redundant way of expressing some constraints.

#### Argument passing

This approach is to constrain the inputs to the thing being constrained. It
requires that anything that might be constrained be able to be specified as an
input. In particular, this means the associated types of an interface can be
specified as optional named arguments.

In this approach, the constraints come before the thing being constrained.

Rust supports using this syntax to set associated types to specific values. This
is particularly used for the case of
[`dyn` traits](https://doc.rust-lang.org/std/keyword.dyn.html) where all
associated types must be specified. Also used for parameter trait bounds in Rust
([1](https://doc.rust-lang.org/rust-by-example/generics/bounds.html),
[2](https://doc.rust-lang.org/book/ch10-02-traits.html#trait-bound-syntax)).
[Swift is considering adopting a similar syntax as well](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--directly-expressing-constraints).

Advantages:

-   More uniform treatment of [interface parameters](#parameterized-interfaces)
    and [associated types](#associated-types).
-   Determining canonical types/type equality is fast and straightforward, see
    [the "generic type equality" section](#generic-type-equality).

Disadvantages:

-   Inventive to use it broadly, beyond
    ["set to a specific value" constraints](#set-to-a-specific-value).
-   More difficulty naming constraints and expressing constraints in interface
    definitions.
-   Some types of constraints are hard to express
    ([1](#parameterized-type-implements-interface), [2](#type-inequality)) or
    need additional syntax, such as [`.Self`](#recursive-constraints) or "for
    some" `[...]` deduced variables
    ([1](#type-bounds-on-associated-types-in-interfaces),
    [2](#same-type-constraints)).
-   type-of-types would become callable, and when called would return a
    type-of-type that could then be called again.

### Constraint use cases

#### Set to a specific value

Useful for associated constants and associated types, with little difference
between them.

##### Associated constants

For [associated constants](#associated-constants), we might need to write a
function that only works with a specific value of `N`. We can solve this using
argument passing, as in:

```
fn PrintPoint2D[PointT:! NSpacePoint(.N = 2)](p: PointT) {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Or `where` clauses:

```
fn PrintPoint2D[PointT:! NSpacePoint](p: PointT) where PointT.N == 2 {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Similarly in an interface definition:

```
interface {
  // Argument passing:
  let PointT:! NSpacePoint(.N = 2);
  // versus `where` clause:
  let PointT:! NSpacePoint where PointT.N == 2;
}
```

To name such a constraint:

```
// Argument passing:
let Point2DInterface:! auto = NSpacePoint(.N = 2);
structural interface Point2DInteface {
  extends NSpacePoint(.N = 2);
}

// versus `where` clause:
let Point2DInterface:! auto = NSpacePoint where Point2D.N == 2;
structural interface Point2DInterface {
  extends NSpacePoint where NSpacePoint.N == 2;
}
```

As you can see, this is a good case for argument passing. Rust supports using
argument passing for this case, even though it generally supports `where`
clauses.

##### Associated types

For example, we could make a the `ElementType` of an `Iterator` interface equal
to the `ElementType` of a `Container` interface as follows:

```
interface Iterator {
  let ElementType:! Type;
  ...
}
interface Container {
  let ElementType:! Type;
  // Argument passing:
  let IteratorType:! Iterator(.ElementType = ElementType);
  // versus `where` clause:
  let IteratorType:! Iterator where IteratorType.ElementType == ElementType;
  ...
}
```

Functions accepting a generic type might also want to constrain an associated
type. For example, we might want to have a function only accept stacks
containing integers:

```
// Argument passing:
fn SumIntStack[T:! Stack(.ElementType = Int)](s: T*) -> Int {
// versus `where` clause:
fn SumIntStack[T:! Stack](s: T*) -> Int where T.ElementType == Int {

// Same implementation in either case:
  var sum: Int = 0;
  while (!s->IsEmpty()) {
    sum += s->Pop();
  }
  return sum;
}
```

To name these sorts of constraints, we could use `let` statements or
`structural interface` definitions.

```
// Argument passing:
let IntStack = Stack(.ElementType = Int);
structural interface IntStack {
  extends Stack(.ElementType = Int);
}

// versus `where` clause:
let IntStack = Stack where IntStack.ElementType == Int;
structural interface IntStack {
  extends Stack where Stack.ElementType == Int;
}
```

[Rust uses trait aliases](https://rust-lang.github.io/rfcs/1733-trait-alias.html)
for this case.

##### Concern

Sometimes we may need a single type-of-type without any parameters or
unspecified associated constants/types, such as to define a `DynPtr(TT)` (as
[described in the following dynamic pointer type section](#dynamic-pointer-type)).
To do this with a `where` clause approach you would have to name the constraint
before using it, an annoying extra step. For Rust, it is part of
[the motivation of supporting argument passing to set values of associated types](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types).

#### Range constraints on associated constants

**Concern:** It is difficult to express mathematical constraints on values in
the argument passing framework. For example, the constraint "`NTuple` where `N`
is at least 2" naturally translates into a `where` clause:

```
fn TakesAtLeastAPair[N:! Int](x: NTuple(N, Int)) where N >= 2 { ... }
```

Similarly for now we only have a `where` clause formulation for constraining the
`N` member of `NSpacePoint` from
[the "associated constants" section](#associated-constants)

```
fn PrintPoint2Or3[PointT:! NSpacePoint](p: PointT)
  where 2 <= PointT.N, PointT.N <= 3 { ... }
```

The same syntax would be used in an interface definition:

```
interface HyperPointInterface {
  let N:! Int where N > 3;
  fn Get[addr me: Self*](i: Int) -> Float64;
}
```

or naming this kind of constraint:

```
let HyperPoint = NSpacePoint where HyperPoint.N > 3;
structural interface HyperPoint {
  extends NSpacePoint where NSpacePoint.N > 3;
}
```

#### Type bounds

TODO

##### Type bounds on associated types in declarations

You might constrain the element type to satisfy an interface (`Comparable` in
this example) without saying exactly what type it is:

```
fn SortContainer[ElementType:! Comparable,
                 ContainerType:! Container(.ElementType = ElementType)]
    (container_to_sort: ContainerType*);
```

You might read this as "for some `ElementType` of type `Comparable`, ...".

To do this with a `where` clause, we need some way of saying a type bound, which
unfortunately is likely to be redundant and inconsistent with how it is said
outside of a `where` clause.

**Open question:** How do you spell that? This proposal provisionally uses `is`,
which matches Swift, but maybe we should have another operator that more clearly
returns a boolean like `has_type`?

```
fn SortContainer[ContainerType:! Container]
    (container_to_sort: ContainerType*)
    where ContainerType.ElementType is Comparable;
```

**Note:** `Container` defines `ElementType` as having type `Type`, but
`ContainerType`, despite being superficially declared as having type
`ContainerType`, is different. `ContainerType.ElementType` has type
`Comparable`. This means we need to be a bit careful when talking about the type
of `ContainerType` when there is a `where` clause that mentiones it.

##### Type bounds on associated types in interfaces

TODO

Now note that inside the `PeekAtTopOfStack` function from the example in the
["associated types" section](#associated-types), we don't know anything about
`StackType.ElementType`, so we can't perform any operations on values of that
type, other than pass them to `Stack` methods. We can define an interface that
has an associated type constrained to satisfy an interface (or any
[other type-of-type](#adapting-types)). For example, we might say interface
`Container` has a `Begin` method returning values with type satisfying the
`Iterator` interface:

```
interface Iterator {
  fn Advance[addr me: Self*]();
  ...
}
interface Container {
  let IteratorType:! Iterator;
  fn Begin[addr me: Self*]() -> IteratorType;
  ...
}
```

With this additional information, a function can now call `Iterator` methods on
the return value of `Begin`:

```
fn OneAfterBegin[T:! Container](c: T*) -> T.IteratorType {
  var iter: T.IteratorType = c->Begin();
  iter.Advance();
  return iter;
}
```

##### Naming type bound constraints

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

We would like to be able to define a `RandomAccessContainer` to be a
type-of-type whose types satisfy `ContainerInterface` with an `IteratorType`
satisfying `RandomAccessIterator`.

**Concern:** We would need to introduce some sort of "for some" operator to
support this with argument passing. We might use a `[...]` to indicate that the
introduced parameter is deduced.

```
// Argument passing:
fn F[IterType:! RandomAccessIterator,
     ContainerType:! ContainerInterface(.IteratorType=IterType)]
    (c: ContainerType);
// versus `where` clause:
fn F[ContainerType:! ContainerInterface](c: ContainerType)
    where ContainerType.IteratorType is RandomAccessIterator;

// WANT a definition of RandomAccessContainer such that the above
// is equivalent to:
fn F[ContainerType:! RandomAccessContainer](c: ContainerType);

// Argument passing:
let RandomAccessContainer =
    [IterType:! RandomAccessIterator]
    ContainerInterface(.IteratorType=IterType);
// versus `where` clause:
let RandomAccessContainer = ContainerInterface
    where RandomAccessContainer.IteratorType is RandomAccessIterator;
```

##### Type bound with interface argument

Use case: we want a function that can take two values `x` and `y`, with
potentially different types, and multiply them. So `x` implements the
`MultipliesBy(R)` interface for `R`, the type of `y`.

```
fn F[R:! Type, L:! MutipliesBy(R)](x: L, y: R) {
  x * y;
}
```

#### Same type constraints

```
interface PairInterface {
  let Left:! Type;
  let Right:! Type;
}

// Argument passing:
fn F[T:! Type, MatchedPairType:! PairInterface(.Left = T, .Right = T)]
    (x: MatchedPairType*);

// versus `where` clause:
fn F[MatchedPairType:! PairInterface](x: MatchedPairType*)
    where MatchedPairType.Left == MatchedPairType.Right;
```

Constraint in an interface definition:

Argument passing approach needs the "for some" `[...]` syntax for deduced
associated types that don't introduce new names into the interface just to
represent the constraint.

```
// Argument passing:
interface HasEqualPair {
  [let T:! Type];
  let P:! PairInterface(.Left = T, .Right = T);
}

// versus `where` clause:
interface HasEqualPair {
  let P:! PairInterface where P.Left == P.Right;
}
```

##### Naming same type constraints

Again, the argument passing approach also needs the "for some" `[...]` syntax
for deduced associated types. Otherwise this first `EqualPair` interface would
only match types that had a type member named `T`.

```
// Argument passing:
let EqualPair = [let T:! Type]
    PairInterface(.Left = T, .Right = T);
structural interface EqualPair {
  [let T:! Type];
  extends PairInterface(.Left = T, .Right = T);
}

// versus `where` clause:
let EqualPair = PairInterface
    where EqualPair.Left == EqualPair.Right;
structural interface EqualPair {
  extends PairInterface
      where PairInterface.Left == PairInterface.Right;
}
```

#### Combining constraints

These different types of constraints can be combined. For example, this example
expresses a constraint that two associated types are equal and satisfy an
interface:

```
// Argument passing:
fn EqualContainers[ET:! HasEquality,
                   Container(CT1:! .ElementType = ET),
                   Container(CT2:! .ElementType = ET)]
    (c1: CT1*, c2: CT2*) -> Bool;

interface HasEqualContainers {
  [let ET:! HasEquality];
  let CT1:! Container(.ElementType = ET);
  let CT2:! Container(.ElementType = ET);
}

// versus `where` clause:
fn EqualContainers[CT1:! Container, CT2:! Container]
    (c1: CT1*, c2: CT2*) -> Bool
    where CT1.ElementType == CT2.ElementType,
          CT1.ElementType is HasEquality;

interface HasEqualContainers {
  let CT1:! Container where CT1.ElementType is HasEquality;
  let CT2:! Container where CT1.ElementType == CT2.ElementType;
}
```

#### Rejected alternative: `ForSome(F)`

Another way to solve the [type bounds](#type-bounds) and
[same type](#same-type-constraints) constraint use cases using argument passing
without the "for some" `[...]` operator would be to have a `ForSome(F)`
construct, where `F` is a function from types to type-of-types.

> `ForSome(F)`, where `F` is a function from type `T` to type-of-type `TT`, is a
> type whose values are types `U` with type `TT=F(T)` for some type `T`.

**Example:** Pairs of values where both values have the same type might be
written as

```
fn F[T:! ForSome(lambda (Type) =>
        PairInterface(MatchedPairType:! .Left = T, .Right = T))]
    (x: MatchedPairType*) { ... }
```

This would be equivalent to:

```
fn F[T:! Type, MatchedPairType:! PairInterface(T, T)]
    (x: MatchedPairType*) { ... }
```

**Example:** Containers where the elements implement the `HasEquality` interface
might be written as:

```
fn F[T:! ForSome(lambda (HasEquality) =>
        Container(ContainerType:! .ElementType = T))]
    (x: ContainerType*) { ... }
```

This would be equivalent to:

```
fn F[T:! HasEquality, ContainerType:! Container(T)]
  (x: ContainerType*) { ... }
```

#### Is a subtype

**Concern:** We need to add some operator to express this, with either argument
passing or where clauses.

For `where` clause we could represent this by a binary `extends` operator
returning a boolean. For argument passing, we'd introduce an `Extends(T)`
type-of-type, whose values are types that extend `T`, that is types `U` that are
subtypes of `T`.

```
// Argument passing:
fn F[T:! Extends(BaseType)](p: T*);
fn UpCast[U:! Type, T:! Extends(U)](p: T*, U) -> U*;
fn DownCast[T:! Type](p: T*, U:! Extends(T)) -> U*;

// versus `where` clause:
fn F[T:! Type](p: T*) where T extends BaseType;
fn UpCast[T:! Type](p: T*, U:! Type) -> U* where T extends U;
fn DownCast[T:! Type](p: T*, U:! Type) -> U* where U extends T;
```

In Swift, you can
[add a required superclass to a type bound using `&`](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282).

#### Parameterized type implements interface

TODO: This use case was part of the
[Rust rationale for adding support for `where` clauses](https://rust-lang.github.io/rfcs/0135-where.html#motivation).

**Concern:** Right now this is only easily expressed using `where` clauses.

```
// Some parametized type.
class Vector(T:! Type) { ... }

// Parameterized type implements interface only for some arguments.
external impl Vector(String) as Printable { ... }

// Constraint: `T` such that `Vector(T)` implements `Printable`
fn PrintThree[T:! Type](a: T, b: T, c: T) where Vector(T) is Printable {
  var v: Vector(T) = (a, b, c);
  Print(v);
}
```

#### Recursive constraints

Just like we use `Self` to refer to the type implementing an interface, we
sometimes need to constrain a type to equal one of its associated types. In this
first example, we want to represent the function `Abs` which will return `Self`
for some but not all types, so we use an associated type `MagnitudeType` to
encode the return type:

```
interface HasAbs {
  extends Numeric;
  let MagnitudeType:! Numeric;
  fn Abs[me: Self]() -> MagnitudeType;
}
```

For types representing subsets of the real numbers, such as `Int32` or
`Float32`, the `MagnitudeType` will match `Self`. For types representing complex
numbers, the types will be different. For example, the `Abs()` applied to a
`Complex64` value would produce a `Float32` result. The challenge is to write a
constraint to restrict to the first case.

In a second example, when you take the slice of a type implementing `Container`
you get a type implementing `Container` which may or may not be the same type as
the original container type. However, taking the slice of a slice always gives
you the same type, and some functions want to only operate on containers whose
slice type is the same.

These problems can be solved directly using the `where` clause approach, but for
argument passing we need to introduce a name for "the type we are in the middle
of declaring". We can't use `Self` directly for this, since we might be in the
middle of an `interface` definition where `Self` already has a meaning.
Provisionally we'll write this `.Self`.

Function declaration:

```
// Argument passing
fn Relu[T:! HasAbs(.MagnitudeType = .Self)](x: T) {
  // T.MagnitudeType == T so the following is allowed:
  return (x.Abs() + x) / 2;
}
fn UseContainer[T:! Container(.SliceType = .Self)](c: T) -> Bool {
  // T.SliceType == T so `c` and `c.Slice(...)` can be compared:
  return c == c.Slice(...);
}

// versus `where` clause
fn Relu[T:! HasAbs](x: T) where T.MagnitudeType == T {
  return (x.Abs() + x) / 2;
}
fn UseContainer[T:! Container](c: T) -> Bool where T.SliceType == T {
  return c == c.Slice(...);
}
```

Interface definition:

```
interface Container {
  let ElementType:! Type;

  // Argument passing:
  let SliceType:! Container(.ElementType = ElementType, .SliceType = .Self);
  // versus `where` clause
  let SliceType:! Container where SliceType.ElementType == ElementType,
                                 Slicetype.SliceType == SliceType;

  fn GetSlice[addr me: Self*]
      (start: IteratorType, end: IteratorType) -> SliceType;
}
```

Naming these constraints:

```
// Argument passing
let RealAbs = HasAbs(.MagnitudeType = .Self);
structural interface RealAbs {
  extends HasAbs(.MagnitudeType = Self);
}
let ContainerIsSlice = Container(.SliceType = .Self);
structural interface ContainerIsSlice {
  extends Container(.SliceType = Self);
}

// versus `where` clause
let RealAbs = HasAbs where RealAbs.MagnitudeType == RealAbs;
structural interface RealAbs {
  extends HasAbs where HasAbs.MagnitudeType == Self;
}
let ContainerIsSlice = Container
    where ContainerIsSlice.SliceType == ContainerIsSlice;
structural interface ContainerIsSlice {
  extends Container where Container.SliceType == Self;
}
```

Note that using the `structural interface` approach we can name these
constraints without using `.Self` even with argument passing. However, you can't
always avoid using `.Self`, since naming the constraint before using it doesn't
allow you to define the `Container` interface above, since the named constraint
refers to `Container` in its definition.

**Rejected alternative:** To use this `structural interface` trick to define
`Container`, you'd have to allow it to be defined inline in the `Container`
definition:

```
interface Container {
  let ElementType:! Type;

  structural interface ContainerIsSlice {
    extends Container where Container.SliceType == Self;
  }
  let SliceType:! ContainerIsSlice(.ElementType = ElementType);

  fn GetSlice[addr me: Self*](start: IteratorType,
                                    IteratorType: end) -> SliceType;
}
```

**Rejected alternative:** If we were to write variable declarations with the
name first instead of the type, we could use that name inside the type
declaration, as in `T:! HasAbs(.MagnitudeType = T)`.

FIXME: There is an important difference here, constraints equating `.Self` or
`Self` to some other type don't change the interfaces that are implemented like
[same type constraints](#same-type-constraints). Instead it will just check that
the type represented by `Self` or `.Self` satisfies the constraints on the other
type.

Example: This does not type check:

```
interface A {
    let Q:! E;
    let P:! E;
    let Y:! A where .Q == Q and .P == Q;
}

interface B {
    let X:! A where .Y == .Self;
}
```

But this does:

```
interface A {
    let Q:! E;
    let P:! E;
    let Y:! A where .Q == Q and .P == Q;
}

interface B {
    let X:! A where .Y == .Self and .P == .Q;
}
```

And this does:

```
interface A {
    let Q:! E;
    let P:! E;
    let Y:! A where .Q == Q;
}

interface B {
    let X:! A where .Y == .Self;
}
```

#### Type inequality

TODO: inequality type constraints (for example "type is not `Bool`").

You might need an inequality type constraint, for example, to control overload
resolution:

```
fn F[T:! Type](x: T) -> T { return x; }
fn F(x: Bool) -> String {
  if (x) return "True"; else return "False";
}

fn G[T:! Type](x: T) -> T {
  // We need a T != Bool constraint for this to type check.
  return F(x);
}
```

Another use case for inequality type constraints would be to say something like
"define `ComparableTo(T1)` for `T2` if `ComparableTo(T2)` is defined for `T1`
and `T1 != T2`".

**Concern:** Right now this is only easily expressed using `where` clauses.

```
fn G[T:! Type](x: T) -> T where T != Bool { return F(x); }
```

### Implicit constraints

Imagine we have a generic function that accepts a arbitrary `HashMap`:

```
fn LookUp[KeyType:! Type](hm: HashMap(KeyType, Int)*,
                          k: KeyType) -> Int;

fn PrintValueOrDefault[KeyType:! Printable,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

The `KeyType` in these declarations does not satisfy the requirements of
`HashMap`, which requires the type to at least implement `Hashable` and probably
others like `Sized`, `EqualityComparable`, `Movable`, and so on.

```
class HashMap(
    KeyType:! Hashable & Sized & EqualityComparable & Movable,
    ...) { ... }
```

**Open question:** Should we allow those function declarations, and implicitly
add needed constraints to `KeyType` implied by being used as an argument to a
parameter with those constraints? Or should we require `KeyType` to name all
needed constraints as part of its declarations?

In this specific case, Swift will accept the definition and infer the needed
constraints on the generic type parameter
([1](https://www.swiftbysundell.com/tips/inferred-generic-type-constraints/),
[2](https://github.com/apple/swift/blob/main/docs/Generics.rst#constraint-inference)).
This is both more concise for the author of the code and follows the
["don't repeat yourself" principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
This redundancy is undesirable since it means if the needed constraints for
`HashMap` are changed, then the code has to be updated in more locations.
Further it can add noise that obscures relevant information. In practice, any
user of these functions will have to pass in a valid `HashMap` instance, and so
will have already satisfied these constraints.

**Note:** These implied constraints should affect the _requirements_ of a
generic type parameter, but not the _names_. This way you can always look at the
declaration to see how name resolution works, without having to look up the
definitions of everything it is used as an argument to.

**Alternative:** As an alternative, we could make it so the user would need to
explicitly opt in to this behavior by adding `& auto` to their type constraint,
as in:

```
fn LookUp[KeyType:! Type & auto](hm: HashMap(KeyType, Int)*,
                                 k: KeyType) -> Int;

fn PrintValueOrDefault[KeyType:! Printable & auto,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

**Caveat:** These constraints can be obscured:

```
interface I(A:! Type, B:! Type, C:! Type, D:! Type, E:! Type) {
  let SwapType:! I(B, A, C, D, E);
  let CycleType:! I(B, C, D, E, A);
  fn LookUp(hm: HashMap(D, E)*) -> E;
  fn Foo(x: Bar(A, B));
}
```

All type arguments to "I" must actually implement `Hashable` (since
[an adjacent swap and a cycle generate the full symmetry group on 5 elements](https://www.mathcounterexamples.net/generating-the-symmetric-group-with-a-transposition-and-a-maximal-length-cycle/)).
And additional restrictions on those types depend on the definition of `Bar`.
For example, this definition

```
class Bar(A:! Type, B:! ComparableWith(A)) { ... }
```

would imply that all the type arguments to `I` would have to be comparable with
every other. This propagation problem means that allowing implicit constraints
to be inferred in this context is substantial (potentially unbounded?) work for
the compiler, and these implied constraints are not at all clear to human
readers of the code either.

**Conclusion:** The initial declaration part of an `interface`, type definition,
or associated type declaration should include complete description of all needed
constraints.

Furthermore, inferring that two types are equal (in contrast to the type bound
constraints described so far) introduces additional problems for establishing
which types are equal in a generic context.

### Covariant extension

Under C++ type inheritance, there are some changes allowed to the members of the
supertype in the subtype. For example, the signatures of functions can change as
long as the result is compatible, see
[covariance and contravariance on Wikipedia](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>).
For interfaces, the analogous thing is to [extend](#interface-extension) a
constrained interface.

For containers, imagine that we have a `ForwardContainer` interface that
supports iteration in one direction, represented by having an `IteratorType`
implementing `ForwardIterator`. To define a `BidirectionalContainer` extension
of `ForwardContainer` that supports iteration in the reverse direction, we would
need to:

-   define `BidirectionalIterator` extending `ForwardIterator`, and
-   enforce that the `IteratorType` of a `BidirectionalContainer` implements
    `BidirectionalIterator`.

Since `BidirectionalIterator` extends `ForwardIterator`, any type satisfying
`BidirectionalIterator` can be used as the `IteratorType` of a
`ForwardContainer`.

```
interface ForwardIterator {
  // ...
  fn Advance[addr me: Self*]();
}

interface BidirectionalIterator {
  extends ForwardIterator;
  fn Back[addr me: Self*]();
}

interface ForwardContainer {
  let IteratorType:! ForwardIterator;
  fn Begin[addr me: Self*]() -> IteratorType;
  fn End[addr me: Self*]() -> IteratorType;
  // ...
}

```

To define `BidirectionalContainer`, we need to use a
[type bound constraint](#type-bounds) on `IteratorType`. Using the argument
passing approach, you would use an deduced associated type:

```
interface BidirectionalContainer {
  // `Extended` is some new name so we don't collide with
  // `IteratorType`. The `[...]` mean this new name is
  // only used as a constraint, and is not part of the
  // `BidirectionalContainer` API.
  [let Extended:! BidirectionalIterator];
  extends ForwardContainer(.IteratorType = Extended);
}
```

To do this with a a [`where` clause](#where-clauses):

```
interface BidirectionalContainer {
  extends ForwardContainer
      where ForwardContainer.IteratorType is BidirectionalIterator;
}
```

With C++ type inheritance, you might define implementations of parent methods in
the child type. The analogous thing for interfaces would be to provide
[default implementations](#interface-defaults) of parent methods or
[a blanket impl](#parameterized-impls) of the parent interface, however the
specifics of how this would be done are future work.

**Open question:** We may want to support extension of other items as well, such
as methods. This would be part of matching the features of C++ `class`
inheritance.

**Open question:** We might support a dedicated syntax for this kind of
extension when extending an interface, if we observe it to be a common case or
otherwise cumbersome. One possibility would be to allow a block of these kinds
of extensions in place of a terminating semicolon (`;`) for `impl` and `extends`
declarations in an interface, as in:

```
interface BidirectionalContainer {
  extends ForwardContainer {
    // Redeclaration of `IteratorType` with a more specific bound.
    let IteratorType:! BidirectionalIterator;
  }
}
```

This syntax would more naturally support adding default implementations of
parent methods and refining parent method signatures.

### Generic type equality

Imagine we have some function with generic parameters:

```
fn F1[T:! SomeInterface](x: T) {
  x.G(x.H());
}
```

We want to know if the return type of method `T.H` is the same as the parameter
type of `T.G` in order to typecheck the function.

#### Type equality with where clauses

With the full expressive power of `where` clauses, determining whether two type
expressions are equal is in general undecidable, as
[has been shown in Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).
There is ongoing work in Swift
([1](https://forums.swift.org/t/formalizing-swift-generics-as-a-term-rewriting-system/45175),
[2](https://gist.github.com/slavapestov/75dbec34f9eba5fb4a4a00b1ee520d0b))
iterating on how to approach this problem. This new approach has the advantage
of depending less on ad hoc heuristics, and instead formalizes the problem in
terms of term rewriting and applies the
[Knuth–Bendix completion algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Bendix_completion_algorithm)
to convert it into an equivalent convergent term rewrite system that can resolve
types efficiently and without backtracking. Note that there are
[cases](https://en.wikipedia.org/wiki/Knuth%E2%80%93Bendix_completion_algorithm#A_non-terminating_example)
where the algorithm will never terminate, and there exists research into making
the
[algorithm faster](https://www.researchgate.net/publication/221521331_Reducing_the_Complexity_of_the_Knuth-Bendix_Completion-Algorithm_A_Unification_of_Different_Approaches)
and
[complete successfully more often](https://homepage.divms.uiowa.edu/~astump/papers/thesis-wehrman.pdf).
This last reference gives running times ranging from 3 seconds for a 10-rule
completion to about 7 hours for a completing a theory with 21 identities.

#### Type equality with argument passing

However, with enough constraints, we can make an efficient decision procedure
for the argument passing formulation. The way we do this is by assigning every
type expression a canonical type, and then two types expressions are equal if
and only if they are assigned the same canonical type. To show how to assign
canonical types, lets work an example with interfaces `A` and `B` (letters from
the end of the alphabet will represent types), and this function declaration:

```
fn F2[Z:! A, V:! B(.Y = Z, .X = Z.W)](...) { ... }
```

We require the following rules to be enforced by the language definition and
compiler:

-   No forward references in a function declaration.
-   No forward references between items in an `interface` definition.
-   No implicit type equality constraints.

From these rules, we derive rules about which type expressions are canonical.
The first rule is:

> For purposes of type checking a function, the names of types declared in the
> function declaration to the left of the `:!` are all canonical.

This is because these can all be given distinct types freely, only their
associated types can be constrained to be equal to some other type. In this
example, this means that the types `Z` and `V` are both canonical. The second
rule comes from there being no forward references in declarations, and no
implicit type equality constraints:

> No declaration can affect type equality for any declaration to its left.

This means that the canonical types for type expressions starting with `Z.` are
completely determined by the declaration `Z:! A`. Furthermore, since the set of
type expressions starting with `Z.` might be infinite, we adopt the lazy
strategy of only evaluating expressions that are needed for something explicitly
mentioned.

We do need to evaluate `Z.W` though for the `V:! B(.Y = Z, .X = Z.W)`
expression. This is an easy case, though since `Z:! A` doesn't include any
assignments to any associated types. In this case, the associated types of `A`
are all canonical. An alias defined in `A` would of course not be, it would be
set to the canonical type for whatever it is an alias for. For example:

```
interface A {
  // `W` is canonical.
  let W:! A;
  // `U` is not canonical, is equal to `W.W`.
  alias U = W.W;
  // `T` is canonical, but `T.Y` is not.
  let T:! B(.Y = Self);
}
```

Next lets examine the definition of `B` so we can resolve expressions starting
with `V.`.

```
interface B {
  let S:! A;
  let Y:! A(.W = S);
  let X:! A;
  let R:! B(.X = S);
}
```

This time we also have assignments `V.Y = Z` and `V.X = Z.W`. As a consequence,
neither `V.Y` nor `V.X` are canonical, and their canonical type is determined
from their assignments. Furthermore, the assignment to `Y` determines `S`, since
`B.S = B.Y.W`, so `V.S` also isn't canonical, it is `V.Y.W` (not canonical)
which is `Z.W` (canonical). Observe that `V.R` is canonical since nothing
constrains it to equal any other type, even though `V.R.X` is not, since it is
`V.S == Z.W`.

The property that there are no forward references between items in interface
definitions ensures that we don't have any cycles that could lead to infinite
loops. That is, the members of an associated type in an interface definition can
only be constrained to equal values that don't depend on that member.

This is almost enough to ensure that the process terminates, except when an
associated type bound is the same interface recursively. The bad case is:

```
interface Broken {
  let Q:! Broken;
  let R:! Broken(.R = Q.R.R);
}

fn F[T:! Broken](x: T) {
  // T.R.R not canonical
  // == T.Q.R.R not canonical
  // == T.Q.Q.R.R not canonical
  // etc.
}
```

The problem here is that while we have a ordering for expressions that guaranees
there are no loops, we don't have a guarantee that there are only finitely many
smaller expressions when we have recursion. With recursion, we can create an
infinite sequence of smaller expressions by allowing their length to grow
without bound. This means we need to add one more rule to ensure that the
algorithm terminates:

> It is illegal to constrain a member of an associated type to (transitively)
> equal a longer expression with the same interface bound.

A few notes on this rule:

-   The word "transitively" is needed if mutual recursion is allowed between
    interfaces (as in `A` and `B` above).
-   There is an additional restriction if the expression has the same length
    that it only refer to earlier names. Without mutual recursion, this is
    already precluded by the "no forward references" rule.
-   We are relying on there being a finite number of interfaces, so we ignore
    [interface parameters](#parameterized-interfaces) when checking this
    condition.
-   This never applies to function declarations, since there is no recursion
    involved in that context.

The fix for this situation is to introduce new deduced associated types:

```
interface Fixed {
  [let RR:! Fixed];
  [let QR:! Fixed(.R = RR)];
  let Q:! Fixed(.R = QR);
  let R:! Fixed(.R = RR);
}

fn F[T:! Fixed](x: T) {
  // T.RR canonical
  // T.R.R == T.RR
  // T.Q.R.R == T.Q.RR == T.QR.R == T.RR
  // T.Q.Q.R.R == T.Q.Q.RR == T.Q.QR.R == T.Q.RR == T.QR.R == T.RR
  // etc.
}
```

The last concern is what happens when an expression is assigned twice. This is
only a problem if it is assigned to two values that resolve to two different
canonical types. That happens in this example:

```
fn F3[N:! A, P:! A, Q:! B(.S = N, .Y = P)](...) { ... }
```

The compiler is required to report an error rejecting this declaration. This is
because the constraints declared in `B` require that `Q.Y.W == Q.S == N` so
`P.Y == N`. This violates the "no implicit type equality constraint" rule since
`P` is not declared with any constraint forcing that to hold. We can't let `Q`'s
declaration affect earlier declarations, otherwise our algorithm would
potentially have to resolve cycles. The compiler should recommend the user
rewrite their code to:

```
fn F3[N:! A, P:! A(.Y = N), Q:! B(.S = N, .Y = P)](...) { ... }
```

This resolves the issue, and with this change the compiler can now correctly
determine canonical types.

**Note:** This algorithm still works with the `.Self` feature from the
["recursive constraints" section](#recursive-constraints). For example, the
expression `let Y:! A(.X = .Self)` means `Y.X == Y` and so the `.Self` on the
right-side represents a shorter and earlier type expression. This precludes
introducing a loop and so is safe.

**Open question:** Can we relax any of the restrictions? For example, perhaps we
would like to allow items in an interface to reference each other, as in:

```
interface D {
  let E:! A(.W = V);
  let V:! A(.W = E);
}
```

This example may come up for graphs where `E` is the edge type and `V` is the
vertex type. In this case `D.E.W == D.V` and `D.V.W == D.E` and we would need
some way of deciding which were canonical (probably `D.E` and `D.V`). This would
have to be restricted to cases where the expression on the right has no `.` to
avoid cycles or type expression that grow without bound. Another concern is if
there are type constructors involved:

```
interface Graph {
  let Edges:! A(.W = Vector(Verts));
  let Verts:! A(.W = Vector(Edges));
}
```

**Open question:** Is this expressive enough to represent the equality
constraints needed by users in practice?

##### Canonical types and type checking

TODO: For assignment to type check, argument has to have the same or a more
restrictive type-of-type than the parameter. This means that the canonical type
expression would have the right (most restrictive) type-of-type to use for all
expressions equal to it, with the exception of
[implicit constraints](#implicit-constraints).

#### Restricted where clauses

This leads to the question of whether we can describe a set of restrictions on
`where` clauses that would allow us to directly translate them into the argument
passing form. If so, we could allow the `where` clause syntax and still use the
above efficient decision procedure.

Consider an interface with one associate type that has `where` constraints:

```
interface Foo {
  // Some associated types
  let A:! ...;
  let B:! Z where B.X == ..., B.Y == ...;
  let C:! ...;
}
```

These forms of `where` clauses are allowed because we can rewrite them into the
argument passing form:

| `where` form                   | argument passing form   |
| ------------------------------ | ----------------------- |
| `let B:! Z where B.X == A`     | `let B:! Z(.X = A)`     |
| `let B:! Z where B.X == A.T.U` | `let B:! Z(.X = A.T.U)` |
| `let B:! Z where B.X == Self`  | `let B:! Z(.X = Self)`  |
| `let B:! Z where B.X == B`     | `let B:! Z(.X = .Self)` |

Note that the second example would not be allowed if `A.T.U` had the same type
as `B.X`, to avoid non-terminating recursion.

These forms of `where` clauses are forbidden:

| Example forbidden `where` form           | Rule                                     |
| ---------------------------------------- | ---------------------------------------- |
| `let B:! Z where B == ...`               | must have a dot on left of `==`          |
| `let B:! Z where B.X.Y == ...`           | must have a single dot on left of `==`   |
| `let B:! Z where A.X == ...`             | `A` ≠ `B` on left of `==`                |
| `let B:! Z where B.X == ..., B.X == ...` | no two constraints on same member        |
| `let B:! Z where B.X == B.Y`             | right side can't refer to members of `B` |
| `let B:! Z where B.X == C`               | no forward reference                     |

There is some room to rewrite other `where` expressions into allowed argument
passing forms. One simple example is allowing the two sides of the `==` in one
of the allowed forms to be swapped, but more complicated rewrites may be
possible. For example,

```
let B:! Z where B.X == B.Y;
```

might be rewritten to:

```
[let XY:! ...];
let B:! Z(.X = XY, .Y = XY);
```

except it may be tricky in general to find a type for `XY` that satisfies the
constraints on both `B.X` and `B.Y`. Similarly,

```
let A:! ...;
let B:! Z where B == A.T.U
```

might be rewritten as:

```
let A:! ...;
alias B = A.T.U;
```

unless the type bounds on `A.T.U` do not match the `Z` bound on `B`. In that
case, we need to find a type-of-type `Z2` that represents the intersection of
the two type constraints and a different rewrite:

```
let Z2:! B
[let AT:! ...(.U = B)];
let A:! ...(.T = AT);
```

**Note:** It would be great if the
['&' operator for type-of-types](#combining-interfaces-by-anding-type-of-types)
was all we needed to define the intersection of two type constraints, but it
isn't yet defined for two type-of-types that have the same interface but with
different constraints. And that requires being able to automatically combine
constraints of the form `B.X == Foo` and `B.X == Bar`.

**Open question:** How much rewriting can be done automatically?

**Open question:** Is there a simple set of rules explaining which `where`
clauses are allowed that we could explain to users?

#### Manual type equality

TODO

### Deleted content

The restrictions arise from the the algorithm used to answer type questions. It
works by first rewriting `where` operations to put a declaration, like a
function signature or interface definition, into a normalized form. FIXME:
triggered by type checking. This normalized form can then be lazily evaluated to
answer queries. Queries take a dotted name and return an archetype that has a
canonical type name and a type-of-type.

A more complete description of the normalization rewrite and querying algorithms
can be found in [this appendix](appendix-archetype-algorithm.md).

#### Normalized form

The normalized form for a function declaration includes generic type parameters
and any associated types mentioned in a `where` constraint.

```
fn Sort[C:! Container where .Elt is Comparable](c: C*)
```

normalizes to:

```
* $2 :! Comparable
  - C.Elt as Comparable
* $1 :! Container{.Elt = $2}
  - C as Container
```

The normalized form for an interface includes the associated types as well as
dotted names mentioned in `where` constraints. It includes the interface's
`Self` type and interface name to support recursive references. Given these
interface definitions,

```
interface P {
  let T:! F;
}

interface Q {
  let Y:! H;
}

interface R {
  let X:! Q;
}

interface S {
  let A:! P;
  let B:! R where .X.Y == A.T;
}
```

the interface `S` normalizes to:

```
S
* $4 :! F & H
  - A.T as F
  - B.X.Y as F & H
* $3 :! Q{.Y = $4}
  - B.X as Q
* $2 :! P{.T = $4}
  - A as P
* $1 :! R{.X = $3}
  - B as R
* $0 :! S{.A = $2, .B = $1}
  - Self as S
```

Note that `A.T` and `B.X.Y` both correspond to `$4`, but their types are
slightly different, reflecting the fact that the `where` clause modifies the API
of ` B.X.Y` but not `A.T`. The type of `A.T` is `$4 as F`, while the type of
`B.X.Y` is just `$4` or equivalently `$4 as F & H`.

Also note that `B.X` gets its own entry, as a result of `B.X.Y` being mentioned
in a `where` constraint.

#### Recursive constraints

FIXME: Cases which might be rejected due to recursion

#### Terminating recursion

This restriction comes from the query algorithm. It imposes a condition on
recursive references to the same interface. The rewrite to normalized form tries
to avoid triggering this condition, so the only known examples hitting this
restriction require mutually recursive interfaces. That would require forward
declaration of interfaces, which is not permitted at this time, but we may add
in the future.

The query algorithm allows us to determine if two dotted names represent equal
types by querying both and comparing canonical type names. It establishes what
type should be used for a dotted name after constraints are taken into
consideration. For example, in the `Sort` function declaration above, this would
determine that `C.Elt`, `C.Iter.Elt`, `C.Slice.Elt`, and so on are all equal and
implement `Comparable`. This is despite `Elt` as being declared `let Elt:! Type`
in the definition of `Container`.

### Options

TODO: Add third manual option

There is one big choice here, whether we want the fully general expressive
semantics of `where` clauses and the difficult compilation that comes with it,
or the parameter passing approach with enough restrictions to make type equality
straightforward and efficient.

With the general `where` clause approach we may decide, as Rust did, to include
some parameter passing alternatives to the `where` syntax, for convenience.
Similarly we could adopt the parameter passing model, but allow the `where`
syntax in some cases where we could rewrite it automatically to fit.

TODO: fold in content from
[this appendix arguing against `requires` clauses](appendix-requires-constraints.md)

## Parameterized impls

Also known as "blanket `impl`s", these are when you have an `impl` definition
that is parameterized so it applies to more than a single type and interface
combination. These are in many ways similar to implementations with
[conditional conformance](#conditional-conformance), with two differences:

-   Since they apply to more than one type they must always be defined as
    [external impls](#external-impl).
-   Since multiple blanket `impl`s could apply to a particular type and
    interface combination, we need rules to resolve those cases of overlap.

Example use cases:

-   Declare an out-of-line external impl for a parameterized type.
-   If `T` implements `As(U)`, then `Optional(T)` should implement
    `As(Optional(U))`.
-   If `T` implements `ComparableWith(U)`, then `U` should implement
    `ComparableWith(T)`.
-   Any type implementing `Ordered` should get an implementation of
    `PartiallyOrdered`. Question: do we want to guarantee those two must be
    consistent by forbidding any overriding of the `PartiallyOrdered`
    implementation? In other cases, we will want to support overriding for
    efficiency, such as an implementation of `+=` in terms of `+` and `=`.
-   `T` should implement `CommonType(T)` for all `T`

FIXME: This section should be rewritten to be about parameterized `impl` in
general, not just templated. For example, the
["lookup resolution and specialization" section](#lookup-resolution-and-specialization)
is applicable broadly.

TODO: Clarify the difference between a `structural interface SI` that requires
an interface `I` and a nominal `interface NI` that has a blanket implementation
for any type implementing `I`:

-   You can implement `NI` without implementing `I`, but you can't implement
    `SI` without implementing `I`.
-   You can provide a more specialized implementation of `NI` that overrides the
    blanket implementation.

Some things going on here:

-   Our syntax for `external impl` statements already allows you to have a
    (possibly templated) type parameter. This can be used to provide a general
    `impl` that depends on templated access to the type, even though the
    interface itself is defined generically.
-   We very likely will want to restrict the `impl` in some ways.
    -   Easy case: An `impl` for a family of parameterized types.
    -   Trickier is "structural conformance": we might want to say "here is an
        `impl` for interface `Foo` for any class implementing a method `Bar`".
        This is particularly for C++ types, and for
        [functions that are transitioning from templates into generics](goals.md#upgrade-path-from-templates).

The approach here is a generalization of the approach for
[conditional conformance](#conditional-conformance). Recall that conditional
conformance is about restricting an `impl` to apply only to those types whose
parameters meet a criteria. In this section, we extend the ways of
parameterizing a single `impl` declaration so that it applies to multiple types
or multiple interface arguments.

FIXME: Open question whether a blanket impl saying anything implementing `I1`
also implements `I2` is sufficient for a function `F` generically accepting a
type `T` implementing `I1` to call a function `G` that requires it to implement
`I2`. Two concerns: since `F` likely doesn't import the library defining the
actual `T`, it won't see all relevant `impl` declarations so it won't itself be
able to determine which `impl` will be used, that resolution will have to be
deferred to code generation when the generic is instantiated for the right type.
This is a problem for implementing dynamic dispatch cases, since it isn't
obvious that the compiler will need to package a witness table for `I2` along
with a witness table for `I1`. The other concern is whether it will in fact be
guaranteed to be able to implement `I2`, if there are concerns that the
implementation of `I2` by `T` might be blocked by having multiple
implementations to choose from without a clear rule of how to pick a winner.

### Conditional conformance

[The problem](terminology.md#conditional-conformance) we are trying to solve
here is expressing that we have an `impl` of some interface for some type, but
only if some additional type restrictions are met. We can represent these type
constraints in a couple of different ways:

-   We can provide the same `impl` argument in two places to constrain them to
    be the same.
-   We can declare the `impl` argument with a more-restrictive type, to for
    example say this `impl` can only be used if that type satisfies an
    interface.

This follows the approach for expressing
[constraints in other contexts](#constraints).

**Example:** [Interface constraint] Here we implement the `Printable` interface
for arrays of `N` elements of `Printable` type `T`, generically for `N`.

First, showing this with an [`external impl`](#external-impl):

```
interface Printable {
  fn Print[addr me: Self*]() -> String;
}
class FixedArray(T:! Type, N:! Int) { ... }

// By saying "T:! Printable" instead of "T:! Type" here, we constrain
// T to be Printable for this impl.
external impl FixedArray(T:! Printable, N:! Int) as Printable {
  fn Print[addr me: Self*]() -> String {
    var first: Bool = False;
    var ret: String = "";
    for (a: auto) in *this {
      if (!first) {
        ret += ", ";
      }
      ret += a.Print();
    }
    return ret;
  }
}
```

To define these `impl`s inline in a `class` definition, include a more-specific
type between the `impl` and `as` keywords.

```
class FixedArray(T:! Type, N:! Int) {
  // A few different syntax possibilities here:
  impl FixedArray(P:! Printable, N2:! Int) as Printable { ... }
  impl FixedArray(P:! Printable, N) as Printable { ... }
  impl [P:! Printable] FixedArray(P, N) as Printable { ... }
  impl FixedArray[P:! Printable](P, N) as Printable { ... }
}
```

As noted, there are still some open questions about the syntax, and the rules to
prevent inconsistent reuse of the names from the outer scope. This proposal has
the desirable property that the syntax for internal versus external conditional
conformance is very similar. This makes it straightforward to refactor between
those two choices (add or remove `external` and any type parameters from the
context), and is easier to learn.

**Example:** [Same-type constraint] We implement interface `Foo(T)` for
`Pair(T, U)` when `T` and `U` are the same. Using external impl we can write
this either directly, or using deduced parameters in square brackets.

```
interface Foo(T:! Type) { ... }
class Pair(T:! Type, U:! Type) { ... }
external impl Pair(T:! Type, T) as Foo(T) { ... }

// Alternatively:
external impl Pair[T:! Type](T, T) as Foo(T) { ... }
```

You may also define the `impl` inline:

```
class Pair(T:! Type, U:! Type) {
  impl Pair(T, T) as Foo(T) { ... }
}
```

**Proposal:** [Other boolean condition constraints] Just like we support
conditions when pattern matching (for example in overload resolution), we should
also allow them when defining an `impl`:

```
external impl [template T:! Type] T as Foo if (sizeof(T) <= 16) {
  ...
}
```

**Concern:** The conditional conformance feature makes the question "is this
interface implemented for this type" undecidable in general.
[This feature in Rust has been shown to allow implementing a Turing machine](https://sdleffler.github.io/RustTypeSystemTuringComplete/).
This means we will likely need some heuristic like a limit on how many steps of
recursion are allowed.

**Future work:**
[Rust uses this mechanism](https://doc.rust-lang.org/book/ch10-02-traits.html#using-trait-bounds-to-conditionally-implement-methods)
to also allow conditionally defining methods on types, independent of
interfaces. We will likely instead just use a more specific type in place of
`Self` in the method declaration, rather than be consistent across these two use
cases.

```
class FixedArray(T:! Type, N:! Int) {
  // ...
  fn Print[me: FixedArray(P:! Printable, N)]() { ... }
}

// FixedArray(T, N) has a `Print()` method if `T` is `Printable`.
```

### Bridge for C++ templates

#### Calling C++ template code from Carbon

Let's say we want to call some templated C++ code from generic Carbon code.

```
// In C++
template<class T>
struct S {
  void F(T* t);
};
```

We first define the common API for the template:

```
// In Carbon
interface SInterface(T:! Type) {
  fn F[addr me: Self*](t: T*);
}
```

and once we implement that interface for the C++ type `S`:

```
// Note: T has to be a templated argument to be usable with the
// C++ template `S`. There is no problem passing a template
// argument `T` to the generic argument of `SInterface`.
external impl C++::S(template T:! Type) as SInterface(T) {
  fn F[addr me: Self*](t: T*) { this->F(t); }
}
```

we can then call it from a generic Carbon function:

```
fn G[T:! Type, SType:! SInterface(T)](s: SType*, t: T*) {
  s->F(t);
}
var x: C++::S(Int);
var y: Int = 3;
// C++::S(Int) implements SInterface(Int) by way of templated impl
G(&x, &y);
```

#### Moving a C++ template to Carbon

Imagine we have a C++ templated type with (possibly templated) consumers in C++,
and we want to migrate that type to Carbon. For example, say we have a template
`C++::Foo` in C++, and are moving it to Carbon generic `Foo`. Let's say the
`C++::Foo` template takes optional parameters, `C++::std::optional<T>` for any
`T`, but of course the way template code is typically written is to make it work
with anything that has the `C++::std::optional<T>` API. When we move it to
generic `Foo` in Carbon, we need both the `T` argument, and a
[higher-ranked](#higher-ranked-types) type parameter to represent the optional
type. Some C++ users will continue to use this type with C++'s
`std::optional<T>`, which by virtue of being a C++ template, can't take generic
arguments. We still can make a templated implementation of a generic interface
for it:

```
interface Optional(T:! Type) { ... }
external impl C++::std::optional(template T:! Type) as Optional(T) {
  ...
}
```

### Subtlety around interfaces with parameters

Since interfaces with parameters can have multiple implementations for a single
type, it opens the question of how they work when implementing one interface in
terms of another.

**Open question:** We could allow parameterized `impl`s to take each of these
multiple implementations for one interface and manufacture an `impl` for another
interface, as in this example:

```
// Some interfaces with type parameters.
interface Equals(T:! Type) { ... }
// Types can implement parameterized interfaces more than once as long as the
// templated arguments differ.
class Complex {
  var r: Float64;
  var i: Float64;
  impl as Equals(Complex) { ... }
  impl as Equals(Float64) { ... }
}
// Some other interface with a type parameter.
interface PartiallyEquals(T:! Type) { ... }
// This provides an impl of PartiallyEquals(T) for U if U
// is Equals(T). In the case of Complex, this provides two
// impls, one for T == Complex, and one for T == Float64.
external impl [T:! Type, U:! Equals(T)]
    U as PartiallyEquals(T) {
  ...
}
```

In short, this is saying that deduced parameters in `external impl` declarations
are treated as meaning "for all" instead of "for some" as in other contexts.

One tricky part of this is that you may not have visibility into all the impls
of an interface for a type since they may be
[defined with one of the other types involved](#impl-lookup). Hopefully this
isn't a problem -- you will always be able to see the _relevant_ impls given the
types that have been imported / have visible definitions.

### Lookup resolution and specialization

Allowed patterns and their prioritization, where `LocalType` and `LocalTrait`
are defined in this library, `ForeignType` and `ForeignTrait` are defined in
something this library depends on, `T` and `U` are type variables:

-   Highest priority: implementation for a type expression involving a local
    type without any type variables
-   `impl LocalType as AnyTrait(...)`
-   `impl LocalType(ForeignType) as AnyTrait(...)`
-   `impl ForeignType(LocalType) as AnyTrait(...)`
-   High priority: implementation for a local type or a foreign type with a
    local type parameter
-   `impl ForeignType(LocalType(T)) as AnyTrait(...)` or
    `impl LocalType(ForeignType(T)) as AnyTrait(...)`
-   `impl ForeignType(LocalType, T) as AnyTrait(...)` or
    `impl LocalType(ForeignType, T) as AnyTrait(...)`
-   `impl ForeignType(T, LocalType) as AnyTrait(...)` or
    `impl LocalType(T, ForeignType) as AnyTrait(...)`
-   `impl LocalType(T, T) as AnyTrait(...)`
-   `impl LocalType(T, U) as AnyTrait(...)`
-   Medium priority: implementation for a foreign trait with a local type
    parameter
-   `impl ... as ForeignTrait(LocalType)`
    -   `impl ForeignType(T) as ...` prioritized higher than `impl T as ...`
-   `impl ... as ForeignTrait(LocalType(T))`
    -   `impl T as ...` prioritized higher than `impl U as ...`
-   `impl ... as ForeignTrait(LocalType, T)`
-   `impl ... as ForeignTrait(T, LocalType)`
-   Low priority: implementation for a local trait
-   `impl ForeignType as LocalTrait(...)`
-   `impl ForeignType(T) as LocalTrait(...)`
-   `impl T as LocalTrait(...)`

Constraints are only considered as a tie-breaker after all type structure rules
are applied.

**Orphan rule:** An `impl` may only be defined in a library that must be a
dependency of any query where that `impl` could apply. This is accomplished by
only allowing `impl` signatures involving a _local_ type or interface outside of
constraints. Here "local" is a type or interface defined in the same library as
the `impl`.

To see that the local type or interface can't just be in a constraint, imagine
we have three libraries, with one a common library imported by the other two:

```
// library Common
interface ICommon { ... }
struct S { ... }
```

```
// library A
import Common
interface IA { ... }
// Local interface only used as a constraint
impl [T:! IA] T as ICommon { ... }
// Fine: implementation of a local interface
impl S as IA { ... }
```

```
// library B
import Common
interface IB { ... }
// Local interface only used as a constraint
impl [T:! IB] T as ICommon { ... }
// Fine: implementation of a local interface
impl S as IB { ... }
```

Now imagine another library imports the `Common` library. Inside this new
library:

-   Does `S` implement `ICommon`? If you just import `ICommon`, no
    implementations are visible
-   Does the answer change if you import libraries `A` or `B`?
-   Which implementation of `ICommon` should `S` use if you import both?

We avoid these problems by requiring the use of a local type or interface
outside of constraints.

**Overlap rule:** Can have multiple `impl` definitions that match, as long as
there is a single best match. Best is defined using the "more specific" partial
ordering:

-   A more specific constraint on the `Self` type is more specific than
    constraints on the interface parameters.
-   A more specific constraint on earlier type or interface parameters is more
    specific than constraints on later parameters.
-   Matching an exact type (`Foo` or `Foo(Bar)`) is more specific than a
    parameterized family of types (`Foo(T)` for any type `T`) is more specific
    than a generic type (`T` for any type `T`).
-   If one type family is contained within another, the smaller type family is
    more specific, so `Foo(T, T)` is more specific than `Foo(T, U)`.
-   Matching a derived interface is more specific and therefore a closer match
    than the interface being extended.
-   We may want to do the same thing for a derived type, but we might also want
    to forbid that as a conflict to improve substitutability. Instead, we would
    allow the interface to be implemented with virtual functions.
-   A more restrictive constraint is more specific. In particular, a
    type-of-type `T` is more restrictive than a type-of-type `U` if the set of
    restrictions for `T` is a superset of those for `U`. So `Foo(T)` for
    `T:! Comparable & Printable` is more specific than `T:! Comparable`, which
    is more specific than `T:! Type`.
-   TODO: others?

The ability to have a more specific implementation used in place of a more
general is commonly called _[specialization](terminology.md#specialization)_.

**CONCERN:** We should restrict what is allowed in a library to prevent the case
where two impls from different libraries can match without one of them being
considered more specific.

**CONCERN:** We should forbid two overlapping `impl`s being defined in the same
library unless it also defines an `impl` matching the exact overlap.

Example: Two impls of the same local interface:

```
interface LocalInterface { ... }
impl [T:! A] T as LocalInterface { ... }
impl [T:! B] T as LocalInterface { ... }
```

Either one of `A` or `B` has to require/imply the other, or the library has to
define another blanket `impl`:

```
impl [T:! A & B] T as LocalInterface { ... }
```

**Open question:** Should we have a
["child trumps parent" rule](http://aturon.github.io/tech/2017/02/06/specialization-and-coherence/)
where we say library `Child` importing library `Parent` is enough to prioritize
`impl` definitions in `Child` over `Parent` when they would otherwise overlap?
How should we prioritize this rule? Perhaps it is more important than order, so
given an implementation for `MyInterface(ParentType, T)` and one for
`MyInterface(T, ChildType)`, we would choose the latter?

TODO: Examples

**Implication:** Can't do `impl` lookup with generic arguments, even if you can
see a matching parameterized definition, since there may be a more-specific
match and we want to be assured that we always get the same result any time we
do an `impl` lookup.

TODO: Example

**Open question:** We need a story for resolving situations where multiple "good
enough" implementations apply. Examples:
[adding a vector to something iterable](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/#in-defense-of-ordering),
[strong and partial ordering with mixed types](https://docs.google.com/presentation/d/1EQHNy1dMlSNu7dDgQmP6LLNtaEV5ZLuXPJxfZR-Zz5U/edit?resourcekey=0-L9Vv0RxrveukomtzmzQ8tQ#slide=id.gd858c35064_0_49),
[ordering implying an equality relation](https://docs.google.com/presentation/d/1EQHNy1dMlSNu7dDgQmP6LLNtaEV5ZLuXPJxfZR-Zz5U/edit?resourcekey=0-L9Vv0RxrveukomtzmzQ8tQ#slide=id.g7b0b30288d_0_0).

**Open question:** The specialization rules can lead to conflicts between
implementations in completely independent libraries, as seen in
["Little Orphan Impls" on the "Baby Steps" blog](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/).
Is this acceptable? Do we want to add some restrictions to avoid this problem?

**Open question:** Rust
[doesn't allow specialization of a general implementation unless its items are marked `default`](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#the-default-keyword).
Is that something we want to require? This would allow us to relax the
restriction of the previous implication for items that may not be specialized.

**Open question:** Rust doesn't let you define an impl for type `Bar(Baz)` in
the library with `Baz` unless the library defining `Bar(T)` marks the type as
"fundamental", in effect promising not to introduce a blanket impl of `Bar(T)`
that could conflicts. What restrictions should we have to balance the
flexibility to use different blanket impls with allowing evolution?

**Future Work:** Rust's
[specialization rules](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#the-default-keyword)
allow you to omit definitions in the more specific implementation, using the
definition in the more general implementation as a default. This requires that
implementations are completely ordered, not just a best match, which is
[a restriction in practice](https://github.com/rust-lang/rfcs/issues/1856).

**Future Work:** Rust furthermore supports defining a general
[`default impl`](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#default-impls)
that is incomplete (in fact this feature was originally called "partial impl")
and _only_ used to provide default implementations for more specialized
implementations.

**Comparison with other languages:** See
[Rust's rules for deciding which impl is more specific](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#defining-the-precedence-rules).

Note that specialization is both tricky and important. Rust started out without
specialization rules and now is taking a long time to stabilize the feature.

-   The
    [Rust impl specialization proposal](https://rust-lang.github.io/rfcs/1210-impl-specialization.html)
    was accepted in June 2015
-   The
    [first PR implementing specialization in Rust](https://github.com/rust-lang/rust/pull/30652)
    was proposed in December 2015 and merged in March 2016.
-   As of July 2021,
    [specialization is still not in stable Rust](https://github.com/rust-lang/rust/pull/30652),
    and lots of work and open questions remain.
-   Using the newtype pattern in Rust can cause the compiler to not use an
    efficient specialization,
    [hurting performance](https://blog.polybdenum.com/2021/08/09/when-zero-cost-abstractions-aren-t-zero-cost.html).

Some of this might be simplified in Carbon if we can avoid compatibility
concerns, but it is still a quite difficult problem.

## Other constraints as type-of-types

There are some constraints that we will naturally represent as named
type-of-types that the user can specify.

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
                          V:! CompatibleWith(T) & Comparable) -> Bool {
  match ((a as U).Compare(b)) {
    case CompareResult.Less => { return True; }
    case CompareResult.Greater => { return False; }
    case CompareResult.Equal => {
      return (a as V).Compare(b) == CompareResult.Less;
    }
  }
}
```

Used as:

```
class Song { ... }
adapter SongByArtist for Song { impl as Comparable { ... } }
adapter SongByTitle for Song { impl as Comparable { ... } }
assert(CombinedLess(Song(...), Song(...), SongByArtist, SongByTitle) == True);
```

We might generalize this to a list of implementations:

```
fn CombinedCompare[T:! Type]
    (a: T, b: T, CompareList:! List(CompatibleWith(T) & Comparable))
    -> CompareResult {
  for (U: auto) in CompareList {
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

#### Example: Creating an impl out of other impls

And then to package this functionality as an implementation of `Comparable`, we
combine `CompatibleWith` with [type adaptation](#adapting-types):

```
adapter ThenCompare(T:! Type,
                    CompareList:! List(CompatibleWith(T) & Comparable)) for T {
  impl as Comparable {
    fn Compare[me: Self](that: Self) -> CompareResult {
      for (U: auto) in CompareList {
        var result: CompareResult = (this as U).Compare(that);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

let SongByArtistThenTitle = ThenCompare(Song, (SongByArtist, SongByTitle));
var song: Song = ...;
var song2: SongByArtistThenTitle = Song(...) as SongByArtistThenTitle;
assert((song as SongByArtistThenTitle).Compare(song2) == CaompareResult.Less);
```

### Type facet of another type

Similar to `CompatibleWith(T)`, `FacetOf(T)` introduces an equivalence
relationship between types. `T1 is FacetOf(T2)` if both `T1` and `T2` are facets
of the same type.

### Sized types and type-of-types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`Int32`, `Float64`, etc.) most other concrete types (for
    example most FIXME
    [classes](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md)).
-   It could be known generically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a FIXME
    [dynamic type](#dynamic-pointer-type) such as `Dynamic(TT)`, a FIXME
    [variable-sized type](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#control-over-allocation),
    or you could dereference a pointer to a base type that could actually point
    to a FIXME
    [descendant](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#question-extension--inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

I'm going to call a type "sized" if it is in the first two categories, and
"unsized" otherwise. (Note: something with size 0 is still considered "sized".)
The type-of-type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring (member/local) variables
of that type, taking values of that type as parameters, returning values of that
type, and defining arrays of that type. There will generally be additional
requirements to initialize, move, or destroy values of that type as needed.

Example:

```
interface Foo {
  impl as DefaultConstructible;  // See "interface requiring other interfaces".
}
class Bar {  // Classes are "sized" by default.
  impl as Foo;
}
fn F[T: Foo](x: T*) {  // T is unsized.
  var y: T;  // Illegal: T is unsized.
}
// T is sized, but its size is only known generically.
fn G[T: Foo & Sized](x: T*) {
  var y: T = *x;  // Allowed: T is sized and default constructible.
}
var z: Bar;
G(&z);  // Allowed: Bar is sized and implements Foo.
```

**Note:** The compiler will determine which types are "sized", this is not
something types will implement explicitly like ordinary interfaces.

**Open question:** Even if the size is fixed, it won't be known at the time of
compiling the generic function if we are using the dynamic strategy. Should we
automatically [box](#boxed) local variables when using the dynamic strategy? Or
should we only allow `MaybeBox` values to be instantiated locally?

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

In particular, we should in general avoid monomorphizing to generate multiple
instantiations of the function in this case.

**Open question:** Should `TypeId` be implemented externally for types to avoid
name pollution (`.TypeName`, `.TypeHash`, etc.) unless the function specifically
requests those capabilities?

## Dynamic types

Generics provide enough structure to support runtime dispatch for values with
types that vary at runtime, without giving up type safety. Both Rust and Swift
have demonstrated the value of this feature.

There are two different goals here:

-   Reducing code size at the expense of more runtime dispatch.
-   Increasing expressivity by allowing types to vary at runtime, AKA
    "[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch)".

We address these two different use cases with two different mechanisms. What
they have in common is using a runtime/dynamic type value (using
`type_name: InterfaceName`, no `!`) instead of a generic type value (using
`type_name:! InterfaceName`, with a `!`). In the first case,
[we make the type parameter to a function dynamic](#runtime-type-parameters). In
the second case,
[we use a dynamic type value as a field in a class](#runtime-type-fields). In
both cases, we have a name bound to a runtime type value, which is modeled by a
[dynamic-dispatch witness table](terminology.md#dynamic-dispatch-witness-table)
instead of the
[static-dispatch witness table](terminology.md#static-dispatch-witness-table)
used with generic type values.

### Runtime type parameters

This feature is about allowing a function's type parameter to be passed in as a
dynamic (non-generic) parameter. All values of that type would still be required
to have the same type.

If we pass in a type as an ordinary parameter (using `:` instead of `:!`), this
means passing the witness table as an ordinary parameter -- that is a
[dynamic-dispatch witness table](terminology.md#dynamic-dispatch-witness-table)
-- to the function. This means that there will be a single copy of the generated
code for this parameter.

**Restriction:** The type's size will only be known at runtime, so patterns that
use a type's size such as declaring local variables of that type or passing
values of that type by value are forbidden. Essentially the type is considered
[unsized](#sized-types-and-type-of-types), even if the type-of-type is `Sized`.

**Note:** In principle you could imagine supporting values with a dynamic size,
but it would add a large amount of implementation complexity and would not have
the same runtime performance in a way that would likely be surprising to users.
Without a clear value proposition, it seems better just to ask the user to
allocate anything with a dynamic size on the heap using something like
<code>[Boxed](#boxed)</code> below.

**Question:** Should we prevent interfaces that have functions that accept
`Self` parameters or return `Self` values (and therefore violate the unsized
restriction) from being used as the type of runtime type parameters, as
[Rust requires](https://doc.rust-lang.org/book/ch17-02-trait-objects.html#object-safety-is-required-for-trait-objects),
or should just those functions be blocked?

Note that Swift is switching its approach from blocking interfaces to blocking
individual methods
([1](https://forums.swift.org/t/accepted-se-0309-unlock-existentials-for-all-protocols/47902),
[2](https://github.com/apple/swift-evolution/blob/main/proposals/0309-unlock-existential-types-for-all-protocols.md)).
Rust made the transition in the other direction
[in 2014](https://rust-lang.github.io/rfcs/0255-object-safety.html).

**Restriction:**
[Rust requires](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types)
all of the type-of-type's parameters and associated types be specified, since
they can not vary at runtime.

TODO examples

### Runtime type fields

Instead of
[passing in a single type parameter to a function](#runtime-type-parameters), we
could store a type per value. This changes the data layout of the value, and so
is a somewhat more invasive change. It also means that when a function operates
on multiple values they could have different real types, and so
[there are additional restrictions on what functions are supported, like no binary operations](#restrictions).

**Terminology:** Not quite
["Late binding" on Wikipedia](https://en.wikipedia.org/wiki/Late_binding), since
this isn't about looking up names dynamically. It could be called
"[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch)", but that
does not distinguish it from [runtime type parameter](#runtime-type-parameters)
(both use
[dynamic-dispatch witness tables](terminology.md#dynamic-dispatch-witness-table))
or normal
[virtual method dispatch](https://en.wikipedia.org/wiki/Virtual_function).

This is called
["protocols as types" or "existential types" in Swift](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID275)
and "trait objects" in Rust
([1](https://doc.rust-lang.org/book/ch17-02-trait-objects.html),
[2](https://doc.rust-lang.org/reference/types/trait-object.html)).

Note that Swift's approach of directly using the interface name as the type of a
value is incompatible with Carbon's approach as using interface names as the
type of generic type parameters. As a result, Carbon's approach will have to be
more similar to Rust's approach of using a keyword to distinguish this case.
Swift has found its approach
[has been a common source of confusion](https://forums.swift.org/t/improving-the-ui-of-generics/22814),
and has considered switching to using the keyword "`any`"
([1](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--clarifying-existentials),
[2](https://forums.swift.org/t/pitch-introduce-existential-any/53520)) analogous
to Rust's "`dyn`".

#### Dynamic pointer type

Given a type-of-type `TT` (with some restrictions described below), define
`DynPtr(TT)` as a type that can hold a pointer to any value `x` with type `T`
satisfying `TT`. Variables of type `DynPtr(TT)` act like pointers:

-   They do not own what they point to.
-   They have an assignment operator which allows them to point to new values
    (with potentially different types as long as they all satisfy `TT`).
-   They may be copied or moved.
-   They have a fixed size (unlike the values they point to), though that size
    is larger than a regular pointer.

Example:

```
interface Printable {
  fn Print[addr me: Self*]();
}
class AnInt {
  var x: Int;
  impl as Printable { fn Print[addr me: Self*]() { PrintInt(this->x); } }
}
class AString {
  var x: String;
  impl as Printable { fn Print[addr me: Self*]() { PrintString(this->x); } }
}

var i: AnInt = (.x = 3);
var s: AString = (.x = "Hello");

var i_dynamic: DynPtr(Printable) = &i;
i_dynamic->Print();  // Prints "3".
var s_dynamic: DynPtr(Printable) = &s;
s_dynamic->Print();  // Prints "Hello".

var dynamic: DynPtr(Printable)[2] = (&i, &s);
for (iter: DynPtr(Printable)) in dynamic {
  // Prints "3" and then "Hello".
  iter->Print();
}
```

This corresponds to
[a trait object reference in Rust](https://doc.rust-lang.org/book/ch17-02-trait-objects.html).

##### Restrictions

The member functions in the `TT` interface must only have `Self` in the
"receiver" or `this` position.

This is similar to
[the "object safe" restriction in Rust](https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md)
and for the same reasons. Consider an interface that takes `Self` as an
argument:

```
interface EqualCompare {
  fn IsEqual[addr me: Self*](that: Self*) -> Bool;
}
```

and implementations of this interface for our two types:

```
external impl AnInt as EqualCompare {
  fn IsEqual[me: AnInt*](that: AnInt*) -> Bool {
    return this->x == that->x;
  }
}
external impl AString as EqualCompare {
  fn IsEqual[me: AString*](that: AString*) -> Bool {
    return this->x == that->x;
  }
}
```

Now given two values of type `Dynamic(EqualCompare)`, what happens if we try and
call `IsEqual`?

```
var i_dyn_eq: DynPtr(EqualCompare) = &i;
var s_dyn_eq: DynPtr(EqualCompare) = &s;
i_dyn_eq->IsEqual(&*s_dyn_eq);  // Unsound: runtime type confusion
s_dyn_eq->IsEqual(&*i_dyn_eq);  // Unsound: runtime type confusion
```

For `*i_dyn_eq` to implement `EqualCompare.IsEqual`, it needs to accept any
`DynPtr(EqualCompare).T*` value for `that`, including `&*s_dyn_eq`. But
`i_dyn_eq->IsEquals(...)` is going to call `AnInt.EqualCompare.IsEqual` which
can only deal with values of type `AnInt`. So this construction is unsound.

Similarly, we can't generally convert a return value using a specific type (like
`AnInt`) into a value using the dynamic type, that has a different
representation.

**Concern:** "object safe" removes one of the expressivity benefits of generics
over inheritance, the ability to use `Self` arguments and return values like
`Compare(Self, Self)` and `Clone(Self) -> Self`. Is this going to be too
restrictive or prevent too many interfaces from being object safe?

Rust has a way of defining some methods only being present in an interface if
there is also a "sized" restriction on the type. Since Rust's trait objects are
not sized, this provides a mechanism for having some methods in the interface
only in situations where you know the type isn't dynamic.

**Context:** See
["Proposal: Fat pointers" section of Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.t96sqqyjs8xu -->,
[Existential types](https://en.wikipedia.org/wiki/Type_system#Existential_types)
or
[Dependent pair types](https://en.wikipedia.org/wiki/Dependent_type'%22%60UNIQ--postMath-00000012-QINU%60%22_type).
Also see
[this discussion](https://forums.swift.org/t/unlock-existential-types-for-all-protocols/40665)
about reducing restrictions in Swift.

##### Model

TODO

```
// Note: ConstraintType is essentially "TypeOfTypeOfType".
// It allows `TT` to be any interface or type-of-type.
class DynPtr(template TT:! ConstraintType) {
  class DynPtrImpl {
    private t: TT;
    // The type of `p` is really `t*` instead of `Void*`.
    private p: Void*;
    impl as TT {
      // Defined using meta-programming.
      // Forwards this->F(...) to (this->p as (this->t)*)->F(...)
      // or equivalently, this->t.F(this->p as (this->t)*, ...).
    }
  }
  let T:! TT = (DynPtrImpl as TT);
  private impl_: DynPtrImpl;
  impl as Deref(T) {
    fn Deref[me: Self]() -> T* { return &this->impl_; }
  }
  impl as Assignable[U:! TT](U) {
    fn Assign[addr me: Self*](p: U*) { this->impl_ = (.t = U, .p = p); }
  }
}
```

#### Deref

To make a function work on either regular or dynamic pointers, we define an
interface `Deref` that both `DynPtr` and `T*` implement:

```
// Types implementing `Deref` act like a pointer to associated type `DerefT`.
interface Deref {
  let DerefT:! Type;
  // This is used for the `->` and `*` dereferencing operators.
  fn Deref[me: Self]() -> DerefT*;
}

// Implementation of Deref() for DynPtr(TT).
external impl DynPtr(template TT:! ConstraintType) as Deref {
  let DerefT = DynPtr(TT).DynPtrImpl as TT;
  // or equivalently:
  let DerefT = DynPtr(TT).T;
  ...
}

// Implementation of Deref for T*.
external impl (T:! Type)* as Deref {
  let DerefT = T;
  fn Deref[me: T*]() -> T* { return this; }
}
```

Now we can implement a function that takes either a regular pointer to a type
implementing `Printable` or a `DynPtr(Printable)`:

```
// This is equivalent to `fn PrintIt[T:! Printable](x: T*) ...`,
// except it also accepts `DynPtr(Printable)` arguments.
fn PrintIt[T:! Printable, PtrT:! Deref(.DerefT=T) & Copyable](x: PtrT) {
  x->Print();
}

// T == (AnInt as Printable), PtrT == T*
PrintIt(&i);
// Prints "3"

// T == (AString as Printable), PtrT == T*
PrintIt(&s);
// Prints "Hello"

// T == DynPtr(Printable).T, PtrT == DynPtr(.DerefT=Printable)
PrintIt(dynamic[0]);
// Prints "3"

// T == DynPtr(Printable).T, PtrT == DynPtr(.DerefT=Printable)
PrintIt(dynamic[1]);
// Prints "Hello"
```

#### Boxed

One way of dealing with unsized types is by way of a pointer, as with `T*` and
`DynPtr` above. Sometimes, though, you would like to work with something closer
to value semantics. For example, the `Deref` interface and `DynPtr` type
captures nothing about ownership of the pointed-to value, or how to destroy it.

So we are looking for the equivalent of C++'s `unique_ptr<T>`, that will handle
unsized types, and then later we will add a variation that supports dynamic
types. `Boxed(T)` is like `unique_ptr<T>`:

-   it has a fixed size
-   is movable even if `T` is not
-   will destroy what it points to when it goes out of scope.

It differs, though, in that `Boxed(T)` has an allocator and so can support
copying if `T` does.

```
// Boxed is sized and movable even if T is not.
class Boxed(T:! Type,
             // May be able to add more constraints on AllocatorType (like
             // sized & movable) so we could make it a generic argument?
             template AllocatorType:! AllocatorInterface = DefaultAllocatorType) {
  private var p: T*;
  private var allocator: AllocatorType;
  operator create(p: T*, allocator: AllocatorType = DefaultAllocator) { ... }
  impl as Movable { ... }
  impl as Deref { ... }
}

// TODO: Should these just be constructors defined within Boxed(T)?
// If T is constructible from X, then Boxed(T) is constructible from X ...
external impl Boxed(T:! ConstructibleFrom(template Args:! ...))
    as ConstructibleFrom(Args) {
  ...
}
// ... and Boxed(X) as well.
external impl Boxed(T:! ConstructibleFrom(template Args:! ...))
    as ConstructibleFrom(Boxed(Args)) {
  ...
}

// This allows you to create a Boxed(T) value inferring T so you don't have to
// say it explicitly.
fn Box[T:! Type](x: T*) -> Boxed(T) { return Boxed(T)(x); }
fn Box[T:! Type, template AllocatorType:! AllocatorInterface]
    (x: T*, allocator: AllocatorType) -> Boxed(T, AllocatorType) {
  return Boxed(T, AllocatorType)(x, allocator);
}
```

NOTE: Chandler requests that boxing be explicit so that the cost of indirection
is visible in the source (and in fact visible wherever the dereference happens).
This solution also accomplishes that but may not address all use cases for
boxing.

#### DynBoxed

`DynBoxed(TT)` is to `Boxed(T)` as `DynPtr(TT)` is to `T*`. Like `DynPtr(TT)`,
it holds a pointer to a value of any type `T` that satisfies the interface `TT`.
Like `Boxed(T)`, it owns that pointer.

TODO

```
class DynBoxed(template TT:! ConstraintType,
                template AllocatorInterface:! AllocatorType = DefaultAllocatorType) {
  private var p: DynTT*;
  private var allocator: AllocatorType;
  ...  // Constructors, etc.
  // Destructor deallocates this->p.
  impl as Movable { ... }
  impl as Deref { ... }
}
```

**Question:** Should there be some mechanism to have values be dynboxed in
fast-compile builds, but not boxed in release builds?

**Answer:** Right now we are going with the static strategy for both, and are
just going to focus on making that fast.

#### MaybeBoxed

We have a few different ways of making types with value semantics:

-   `Boxed(T)`: Works with sized and unsized concrete types, `T` need not be
    movable. Even if `T` is movable, it may be large or expensive to move so you
    rather used `Boxed(T)` instead.
-   `DynBoxed(TT)`: Can store values of any type satisfying the interface (so
    definitely no fixed size). Performs
    [dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch).
-   `T`: Regular values that are sized and movable. The extra
    indirection/pointer and heap allocation for putting `T` into a box would
    introduce too much overhead / cost.

In all cases we end up with a sized, movable value that is not very large. Just
like we did with [`Deref` above](#deref), we can create an interface to abstract
over the differences, called `MaybeBoxed`:

```
interface MaybeBoxed {
  // Smart pointer operators
  impl as Deref;
  alias T = Deref.DerefT;
  impl as Movable;
  // We require that MaybeBoxed should be sized, to avoid all
  // users having to say so
  impl as Sized;
}

// Blanket implementations
external impl Boxed(T:! Type) as MaybeBoxed { }
external impl DynBoxed(template TT:! ConstraintType) as MaybeBoxed { }
```

For the case of values that we can efficiently move without boxing, we implement
a new type `NotBoxed(T)` that adapts `T` and so has the same representation and
supports zero-runtime-cost casting.

```
// Can pass a T to a function accepting a MaybeBoxed(T) value without boxing by
// first casting it to NotBoxed(T), as long as T is sized and movable.
adapter NotBoxed(T:! Movable & Sized) for T {  // `template` here?
  impl as Movable = T as Movable;
  impl as Deref {
    fn Deref[addr me: Self*]() -> T* { return this as T*; }
  }
  impl as MaybeBoxed { }
}
// TODO: Should this just be a constructor defined within NotBoxed(T)?
// Says NotBoxed(T) is constructible from a value of type Args if T is.
external impl NotBoxed(T:! ConstructibleFrom(template Args:! ...))
    as ConstructibleFrom(Args) {
  ...
}

// This allows you to create a NotBoxed(T) value inferring T so you don't have
// to say it explicitly. TODO: Could probably replace "template T:! Type" with
// "T:! Movable & Sized", here.
fn DontBox[template T:! Type](x: T) -> NotBoxed(T) inline {
  return x as NotBoxed(T);
}
// Use NotBoxed as the default implementation of MaybeBoxed for small & movable
// types. TODO: Not sure how to write a size <= 16 bytes constraint here.
external impl [template T:! Movable & Sized] T as MaybeBoxed
    if (sizeof(T) <= 16) = NotBoxed(T);
```

This allows us to write a single generic function using that interface and have
the caller decide which of these mechanisms is the best fit for the specific
types being used.

```
interface Foo { fn F[addr me: Self*](); }
fn UseBoxed[T:! Foo, BoxType:! MaybeBoxed(.T=T)](x: BoxType) {
  x->F();  // Possible indirection is visible
}
class Bar { impl as Foo { ... } }
var y: DynBoxed(Foo) = new Bar(...);
UseBoxed(y);
// DontBox might not be needed, if Bar meets the requirements to use the
// default NotBoxed impl of MaybeBox.
UseBoxed(DontBox(Bar()));
```

## Compiler-controlled dispatch strategy

Generally speaking, the differences between the
[static and dynamic dispatch strategies](goals.md#dispatch-control) for
functions with generic parameters should not be visible to end-users. This will
allow the compiler to change strategies as an implementation detail. For
example, it would be legal to use dynamic dispatch to compile a generic function
to reduce code size and build time in development build modes. In a release
build mode it might default to static dispatch, but could decide to use dynamic
dispatch for rarely executed functions based on runtime profiles.

### No address of generic functions

Since a function with a generic parameter can have many different addresses, we
have this rule:

**Rule:** It is illegal to take the address of any function with generic
parameters (similarly template parameters).

This rule also makes the difference between the compiler generating separate
static specializations or using a single generated function with runtime dynamic
dispatch harder to observe, enabling the compiler to switch between those
strategies without danger of accidentally changing the semantics of the program.

### Static local variables

If we support
[static local function variables](https://en.wikipedia.org/wiki/Local_variable#Static_local_variables),
we will need to define how many instances of those variables are created
independent of how the function is instantiated. If we want more than one
instance, we will need some way to explicitly define how those instances relate
to the generic parameters.

## Future work

### Abstract return types

This lets you return an anonymous type implementing an interface from a
function. In Rust this is
[return type of "`impl Trait`"](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).
Also see:

-   [https://rust-lang.github.io/rfcs/1951-expand-impl-trait.html]
-   [https://rust-lang.github.io/rfcs/2071-impl-trait-existential-types.html]
-   [https://rust-lang.github.io/rfcs/2515-type_alias_impl_trait.html]
-   [https://www.ncameron.org/blog/abstract-return-types-aka--impl-trait-/]

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
user's desire. As an example, in Rust the
[iterator trait](https://doc.rust-lang.org/std/iter/trait.Iterator.html) only
has one required method but dozens of "provided methods" with defaults.

In fact, defaults are a generalization of
[specialization](terminology.md#specialization), as observed
[here](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#default-impls),
as long as we allow more specific implementations to be incomplete and reuse
more general implementations for anything unspecified.

More background in
[this "Specialize to reuse" blog post](http://aturon.github.io/tech/2015/09/18/reuse/).

One variation on this may be default implementations of entire interfaces. For
example, `RandomAccessContainer` extends `Container` with an `IteratorType`
satisfying `RandomAccessIterator`. That is sufficient to provide a default
implementation of the indexing operator (operator `[]`), by way of
[implementing an interface](#operator-overloading).

```
interface RandomAccessContainer {
  // Refinement of the associated type `IteratorType` from `Container`.
  extends Container where .IteratorType is RandomAccessIterator;

  // Either `impl` or `extends` here, depending if you want
  // `RandomAccessContainer`'s API to include these names.
  impl as OperatorIndex(Int) {
    // Default implementation of interface.
    fn Get[me: Self](i: Int) -> ElementType {
      return (this.Begin() + i).Get();
    }
    fn Set[addr me: Self*](i: Int, value: ElementType) {
      (this->Begin() + i).Set(value);
    }
  }
}
```

Note that there is a difference between extending an interface with a `where`
constraint specifying a requirement on an associated type, as is done for
`Container.IteratorType` above, and providing a default value for an associated
type, which can be overridden.

### Interface evolution

There are a collection of use cases for making different changes to interfaces
that are already in use. These should be addressed either by describing how they
can be accomplished with existing generics features, or by adding features.

Being able to decorate associated items with `upcoming`, `deprecated`, etc. to
allow for transition periods when items are being added or removed.

As an alternative, users could version their interfaces explicitly. For example,
if we had an interface `Foo` with a method `F`:

```
interface Foo {
  fn F[me: Self]() -> Int;
}
```

and we want to add a method `G` with a default implementation, we do so in an
interface with a new name:

```
interface Foo2 {
  fn F[me: Self]() -> Int;
  fn G[me: Self]() -> Int { return this.F() + 1; }
}

structural interface Foo {
  impl as Foo2;
  alias F = Foo2.F;
}
```

Since `Foo` is now a structural interface, implementing `Foo2` means you
automatically also implement `Foo`. Further, implementing `Foo` implements
`Foo2` with the default implementation of `G`. So, any function requiring an
implementation of either `Foo` or `Foo2` is satisfied by any type implementing
either of those. This would allow an incremental transition from `Foo` to
`Foo2`.

### Evolution from templates to generics

There are several components to supporting a function's transition from template
to generic, described in
[the generic goals](goals.md#upgrade-path-from-templates). In short, we want
both safety and incrementality.

Imagine that we have a function that we want to transition between having a
template type parameter `T` to a generic parameter with a constraint that the
type implement a specific interface, say `Printable`:

```
// Before transition:
fn PrintIt[template T:! Type](p: T*) {
  p->Print();
}

// After transition:
fn PrintIt[T:! Printable](p: T*) {
  p->Print();
}
```

The problem is that the name lookup rules might pick different a different
`Print` function after the transition, in the case when the calling type has an
[external implementation](#external-impl) of `Printable`. For example, imagine
we had a `Song` type with two slightly different `Print` definitions:

```
class Song {
  // ...
  impl as ConvertibleToString {
    fn ToString[me: Self]() -> String { ... }
  }
  fn Print[me: Self]() {
    StdOut.Print("Song: ", me.ToString());
  }
}

// This could be in a different file from the definition
// of `Song` above, in the same file as the `Printable`
// interface definition.
external impl (T:! ConvertibleToString) as Printable {
  fn Print[me: Self]() {
    StdOut.Print(me.ToString());
  }
}
```

In this case, the `Printable.Print` definition would come from a
[blanket implementation](#parameterized-impls) of `Printable` for any type
implementing `ConvertibleToString`. For safety, we need an intermediate step
where the compiler would detect the change in semantics so the user could update
their code. A library might have a release with this intermediate step so their
clients could detect problems before a later release when the transition
completes. Our best idea for this intermediate step is to combine the template
with [the interface bound on the type](#type-bounds).

```
// Middle of transition:
fn PrintIt[template T:! Printable](p: T*) {
  p->Print();
}
```

The compiler would report an error for any caller where `T.Print` was different
from `T.(Printable.Print)`. In effect, the `T` template parameter would have the
API of `T` plus the API of `Printable`, except for any names where those two
APIs conflict, that is have different definitions for the same name.

These sorts of transitions will be significantly easier for users if Carbon has
support for
[calling template code from generic functions](generic-to-template.md)
([TODO: fix link](#broken-links-footnote)).

### Testing

The idea is that you would write tests alongside an interface that validate the
expected behavior of any type implementing that interface. These tests would
only have to be written once, and would serve both a documentation function and
give assurance that types implementing an interface satisfy expected properites.

Inevitably we will need some mechanism for generating values of the type. So an
interface's test suite would actually be parameterized by the `&` of two
interfaces: the interface being tested and a testing-specific interface for
generating values. In some cases, the test suite might be able to use a standard
generate-values-for-tests interface (of which there may be only a few). In
others, a particular interface's test suite might have specific requirements and
so require a custom testing interface.

### Operator overloading

We will need a story for defining how an operation is overloaded for a type by
implementing an interface for that type.

TODO: Basically same approach as Rust, implement specific standardized
interfaces for each operator, or family of operators. See
[Rust operator overloading](https://doc.rust-lang.org/rust-by-example/trait/ops.html),
the [Rust ops module](https://doc.rust-lang.org/std/ops/index.html), and the
list of
[Rust operators and corresponding traits](https://doc.rust-lang.org/book/appendix-02-operators.html#operators).

TODO: We absolutely want to support mixing types when overloading operators. For
example, we should support things like `Point + Vector = Point`, and
`Point - Point = Vector`.

TODO: should implement one function to get both `==` and `!=`, or another
function to get those plus all comparison operators (`<`, `<=`, `>=`, `>`, and
maybe `<=>`).

TODO: Concern about handling `a + b` and `b + a` in both mixed and same types
cases. See
[the discussion in this doc](https://docs.google.com/document/d/1YjF1jcXCSb4zQ4kCcZFAK5jtbIJh9IdaaTbzmQEwSNg/edit#heading=h.unhb6k4e2tps).

Rust added defaults for trait parameters for this use case, see
[Default Generic Type Parameters and Operator Overloading](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading)

**Rejected alternative:** How do we represent binary operations like `Addable`?
Could the interface be defined on the pair of types of the two arguments
somehow?

```
fn F(a: A, b: B, ..., T:! Addable(A, B)) where (A, B) is Addable(A, B) {
  ((A, B) as T).DoTheAdd(x, y)
}
```

There are a couple of problems with the idea of an interface implemented for the
`(LeftType, RightType)` tuple. How does the `impl` get passed in? How do you say
that an interface is only for pairs? These problems suggest it is not worth
trying to do anything special for this edge case. Rust considered this approach
and instead decided that the left-hand type implements the interface and the
right-hand type is a parameter that defaults to being equal to the left-hand
type.

```
interface Addable(Right:! Type = Self) {
  // Assuming we allow defaults for associated types.
  let AddResult:! Type = Self;
  fn Add(lhs: Self, rhs: Right) -> AddResult;
}
```

### Impls with state

`Impl`s where the `impl` itself has state. (from @zygoloid). Use case:
implementing interfaces for a flyweight in a Flyweight pattern where the `impl`
needs a reference to a key -> info map.

### Generic associated types

TODO: Used for "property maps" in the Boost.Graph library.

Also called "associated type constructors."

See
[Carbon generics use case: graph library](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?usp=sharing&resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA)
for context.

Rust has been working toward adding this feature
([1](https://github.com/rust-lang/rust/issues/44265),
[2](https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md),
[3](https://github.com/rust-lang/rfcs/pull/1598),
[4](https://www.fpcomplete.com/blog/monads-gats-nightly-rust/),
[5](https://smallcultfollowing.com/babysteps//blog/2016/11/02/associated-type-constructors-part-1-basic-concepts-and-introduction/),
[6](https://smallcultfollowing.com/babysteps//blog/2016/11/03/associated-type-constructors-part-2-family-traits/)).
It has been proposed for Swift as well
([1](https://forums.swift.org/t/idea-generic-associated-types/5422),
[2](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#generic-associatedtypes),
[3](https://forums.swift.org/t/generic-associated-type/17831)). This corresponds
roughly to
[member templates in C++](https://en.cppreference.com/w/cpp/language/member_template).

### Higher-ranked types

This would be some solution for when a function signature specifies that there
is a way to go from a type to an implementation of an interface parameterized by
that type. This problem was posed
[here (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.qvhzlz54obmt -->.
Examples of things we might want to express:

-   This priority queue's second argument (`QueueLike`) is a function that takes
    a type `U` and returns a type that implements `QueueInterface(U)`:

```
class PriorityQueue(
    Type:! T, QueueLike:! fnty (U:! Type)->QueueInterface(U)) {
  ...
}
```

-   Map takes a container of type `T` and function from `T` to `V` into a
    container of type `V`:

```
fn Map[T:! Type,
       StackLike:! fnty (U:! Type)->StackInterface(U),
       Type:! V]
    (x: StackLike(T)*, f: fnty (T)->V) -> StackLike(V) { ... }
```

TODO: Challenging! Probably needs something like
[Dependent function types](https://en.wikipedia.org/wiki/Dependent_type#Pi_type)

Generic associated types and higher-ranked (or is it higher-kinded?) types solve
the same problem in two different contexts. Generic associated types are about
members of interfaces and higher-ranked types are about function parameters.
There are use cases for higher-ranked types that can be solved by using generic
associated types creatively
([1](https://smallcultfollowing.com/babysteps//blog/2016/11/03/associated-type-constructors-part-2-family-traits/),
[2](https://smallcultfollowing.com/babysteps//blog/2016/11/04/associated-type-constructors-part-3-what-higher-kinded-types-might-look-like/)).

Swift proposals:
[1](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#higher-kinded-types),
[2](https://forums.swift.org/t/higher-kinded-types-monads-functors-etc/4691),
[3](https://forums.swift.org/t/proposal-higher-kinded-types-monads-functors-etc/559),
[4](https://github.com/typelift/swift/issues/1). Rust proposals:
[1](https://smallcultfollowing.com/babysteps//blog/2016/11/04/associated-type-constructors-part-3-what-higher-kinded-types-might-look-like/),
[2](https://smallcultfollowing.com/babysteps//blog/2016/11/09/associated-type-constructors-part-4-unifying-atc-and-hkt/).
These correspond roughly to
[C++ template template parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Template_template_parameter).

### Field requirements

We might want to allow interfaces to express the requirement that any
implementing type has a particular field. This would be to match the
expressivity of inheritance, which can express "all subtypes start with this
list of fields." We might want to restrict what can be done with that field,
using capabilities like "read", "write", and "address of" (which implies read
and write). Swift also has a "modify" capability implemented using coroutines,
without requiring there be a value of the right type we can take the address of.
If we do expose an "address of" capability, it will have to be a real address
since we don't expect any sort of proxy to be able to be used instead.

**Question:** C++ maybe gets wrong that you can take address of any member. If
you couldn't it would:

-   Greatly simplify sanitizers.
-   Make reasoning about what side effects can affect members easier.

Maybe being able to take the address of a member is an opt-in feature? Similarly
for local variables. Maybe can call function taking a pointer from a member
function as long as it doesn't capture? We need to firm up design for example
fields before interfaces for example fields.

### Generic type specialization

[Generic specialization](terminology.md#generic-specialization)

TODO: Main idea is that given `MyType(T)` we should be able to derive
`MyTypeInterface(T)` that captures the interface of `MyType(T)` without its
implementation. Only that interface should be used to typecheck uses of
`MyType(T)` in generic functions, so that we can support specializations. For
example, we could have a specific optimized implementation of `MyType(Bool)` as
long as it conformed to `MyTypeInteface(Bool)` or some extension. Similarly we
should support partial specializations like `MyType(T*)`. Main problem is
supporting this with the dynamic strategy.

### Bridge for C++ customization points

See details in [the goals document](goals.md#bridge-for-c-customization-points).

### Variadic arguments

Background:

-   C++ [variadic templates](http://www.jot.fm/issues/issue_2008_02/article2/)
    and
    [parameter packs](https://en.cppreference.com/w/cpp/language/parameter_pack)
-   Swift "variadic generics"
    [manifesto](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#variadic-generics)
    and pitches [1](https://forums.swift.org/t/variadic-generics/20320), 2
    [a](https://forums.swift.org/t/pitching-the-start-of-variadic-generics/51467)
    [b](https://gist.github.com/CodaFi/a461aca155b16cd4d05a2635e7d7a361)
-   [C# `params` keyword](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/params)
-   Rust
    [supports variadics by way of macros](https://doc.rust-lang.org/rust-by-example/macros/variadics.html),
    [but has thought about supporting them through generics](https://gist.github.com/PoignardAzur/aea33f28e2c58ffe1a93b8f8d3c58667)
-   [D variadic templates](https://dlang.org/articles/variadic-function-templates.html)

We have so far been talking about a syntax using `...` to indicate a parameter
could match multiple arguments, like:

```
fn StrCat(args: SOMETHING...) -> String { ... }
```

There are four use cases to support:

-   `SumInts`: All matching arguments are a specific, concrete type like `Int`.
-   `Min`: All matching arguments are the same type, but that type is a generic
    type parameter.
-   `StaticStrCat`: All matching arguments have a generic type satisfying a
    type-of-type, but may all be different.
-   `DynamicStrCat`: All matching arguments have types satisfying a
    type-of-type, those types may be different, and we use dynamic dispatch to
    access the methods of those types.

Examples:

```
fn SumInts(args: Span(Int)...) -> Int {
  var sum: Int = 0;
  for (var i: Int in args) {
    sum += i;
  }
  return sum;
}

// Concern: Can't deduce `T` unless there is at least one argument.
// fn Min[T:! Comparable & Value](args: Span(T)...) -> Optional(T)
fn Min[T:! Comparable & Value](args: NonEmptySpan(T)...) -> T {
  // Safe since args non-empty
  var result: T = args[0];
  for (var x: T in args[1..]) {
    if (x < result) {
      result = x;
    }
  }
  return result;
}

fn StaticStrCat[T:! GenericArray(ConvertibleToString)](args: T...) -> String {
  // Instantiated once per tuple of argument types.
  var len: Int = 0;
  for (var gr: auto in args) {
    len += gr->Length();
  }
  var result: String = "";
  result.Reserve(len);
  // Loop is unrolled, `gr` has a different type for each iteration.
  for (var gr: auto in args) {
    result += gr->ToString();
  }
  return result;
}

fn DynamicStrCat(args: Array(DynPtr(ConvertibleToString))...) -> String {
  // Same body as above, but only instantiated once and no loop unrolling.
}
```

### Interaction with inheritance

Would like to make object-safe interfaces and abstract base classes (ABCs)
without data members interchangeable to a degree. This is particularly important
for C++ interop.

-   Should be able to inherit from an object-safe interface, and the result
    should be considered to implement that interface.
-   An ABC without data-members should be considered an object-safe interface.
-   You should be able to construct an object from a "pointer to type `T`
    implementing `MyInterface`" that inherits from `MyInterface` as an ABC.
    Ideally, this object would have type
    [`DynPtr(MyInterface)`](#dynamic-pointer-type).
-   Should be able to declare a method of an interface with an implementation as
    `final`, to match the behavior and performance of a non-virtual method in a
    base class.

## Notes

These are notes from discussions after this document was first written that have
not yet been incorporated into the main text above.

-   Can use IDE tooling to show all methods including `external impl`,
    automatically switching to [qualified member names](#qualified-member-names)
    where needed to get that method.
-   Address use cases in [the motivation document](motivation.md).
-   Want inheritance with virtual functions to be modeled by interface
    extension. Example showing the interaction between Dynamic pointer types and
    interface extension.

### Other dynamic types

There are additional use cases for dynamic types beyond
[`DynPtr`](#dynamic-pointer-type) and [`DynBoxed`](#dynboxed). Particularly we
want tools for specific situations where greater performance or type safety is
possible. One example is
[DynStack](https://guiand.xyz/blog-posts/unboxed-trait-objects.html) which holds
a sequence of items with dynamic types more efficiently than a container holding
`DynBoxed` values. We may also want to provide unsafe building blocks for other
types holding values with dynamic types, like
[Rust's unstable CoerceUnsized](https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html).

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
