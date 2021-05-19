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
        -   [Covariant refinement](#covariant-refinement)
        -   [Diamond dependency issue](#diamond-dependency-issue)
    -   [Use case: overload resolution](#use-case-overload-resolution)
-   [Type compatibility](#type-compatibility)
-   [Adapting types](#adapting-types)
    -   [Use case: Using independent libraries together](#use-case-using-independent-libraries-together)
    -   [Example: Defining an impl for use by other types](#example-defining-an-impl-for-use-by-other-types)
-   [Associated constants](#associated-constants)
-   [Associated types](#associated-types)
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
    -   [Generic type equality](#generic-type-equality)
        -   [Type equality with where clauses](#type-equality-with-where-clauses)
        -   [Type equality with argument passing](#type-equality-with-argument-passing)
            -   [Canonical types and type checking](#canonical-types-and-type-checking)
        -   [Restricted where clauses](#restricted-where-clauses)
        -   [Manual type equality](#manual-type-equality)
    -   [Options](#options)
-   [Conditional conformance](#conditional-conformance)
-   [Templated impls for generic interfaces](#templated-impls-for-generic-interfaces)
    -   [Structural conformance](#structural-conformance)
    -   [Bridge for C++ templates](#bridge-for-c-templates)
        -   [Calling C++ template code from Carbon](#calling-c-template-code-from-carbon)
        -   [Moving a C++ template to Carbon](#moving-a-c-template-to-carbon)
    -   [Subtlety around interfaces with parameters](#subtlety-around-interfaces-with-parameters)
    -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
-   [Other constraints as type-types](#other-constraints-as-type-types)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other impls](#example-creating-an-impl-out-of-other-impls)
    -   [Sized types and type-types](#sized-types-and-type-types)
        -   [Model](#model-2)
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
-   [Future work](#future-work)
    -   [Abstract return types](#abstract-return-types)
    -   [Interface defaults](#interface-defaults)
    -   [Evolution](#evolution)
    -   [Testing](#testing)
    -   [Operator overloading](#operator-overloading)
    -   [Impls with state](#impls-with-state)
    -   [Generic associated types](#generic-associated-types)
    -   [Higher-ranked types](#higher-ranked-types)
    -   [Inferring associated types](#inferring-associated-types)
    -   [Field requirements](#field-requirements)
    -   [Generic type specialization](#generic-type-specialization)
    -   [Bridge for C++ customization points](#bridge-for-c-customization-points)
    -   [Reverse generics for return types](#reverse-generics-for-return-types)
-   [Notes](#notes)
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
implemented, we provide an addition operator on type-types. This operator gives
the type-type with the union of all the requirements and the union of the names
minus any conflicts.

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
  x.RenderableDraw();
  x.TieGame();
}
```

Reserving the name when there is a conflict is part of resolving what happens
when you add more than two type-types. If `x` is forbidden in `A`, it is
forbidden in `A & B`, whether or not `B` defines the name `x`. This makes `&`
associative and commutative, and so it is well defined on sets of interfaces, or
other type-types, independent of order.

Note that we do _not_ consider two type-types using the same name to mean the
same thing to be a conflict. For example, the adding a type-type to itself gives
itself, `MyTypeType & MyTypeType == MyTypeType`. Also, given two
[interface extensions](#interface-extension) of a common base interface, the sum
should not conflict on any names in the common base.

**Open syntax question:** Instead of using `&` as the combining operator, we
could use `+`. I'm using `&` in this proposal since it is
[consistent with Swift](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282),
but
[Rust uses `+`](https://rust-lang.github.io/rfcs/0087-trait-bounds-with-plus.html).
See [#531](https://github.com/carbon-language/carbon-lang/issues/531).

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
interface B { method (Self: this) BMethod(); }

interface A {
  method (Self: this) AMethod();
  impl B;
}

def F[A:$ T](T: x) {
  // `x` has type `T` that implements `A`, and so has `AMethod`.
  x.AMethod();
  // `A` requires an implementation of `B`, so `T` also implements `B`.
  x.(B.BMethod)();
}

struct S {
  impl A { method (Self: this) AMethod() { ... } }
  impl B { method (Self: this) BMethod() { ... } }
}
var S: x;
F(x);
```

Like with structural interfaces, an interface implementation requirement doesn't
by itself add any names to the interface, but again those can be added with
`alias` declarations:

```
interface D {
  method (Self: this) DMethod();
  impl B;
  alias BMethod = B.BMethod;
}

def G[D:$ T](T: x) {
  // Now both `DMethod` and `BMethod` are available directly:
  x.DMethod();
  x.BMethod();
}
```

**Comparison with other languages:**
[This feature is called "Supertraits" in Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait).

### Interface extension

When implementing an interface, we should allow implementing the aliased names
as well. In the case of `D` above, this includes all the members of `B`,
obviating the need to implement `B` itself:

```
struct T {
  impl D {
    method (Self: this) DMethod() { ... }
    method (Self: this) BMethod() { ... }
  }
}
var T: y;
G(y);
```

This allows us to say that `D`
["extends" or "refines"](terminology.md#extendingrefining-an-interface) `B`,
with some benefits:

-   This allows `B` to be an implementation detail of `D`.
-   This allows types implementing `D` to implement all of its API in one place.
-   This reduces the boilerplate for types implementing `D`.

We expect this concept to be common enough to warrant dedicated syntax:

```
interface B { method (Self: this) BMethod(); }

interface D {
  extends B;
  method (Self: this) DMethod();
}
// is equivalent to the definition of D from before:
// interface D {
//   impl B;
//   alias BMethod = B.BMethod;
//   method (Self: this) DMethod();
// }
```

No names in `D` are allowed to conflict with names in `B` (unless those names
are marked as `upcoming` or `deprecated` as in
[evolution future work](#evolution)). Hopefully this won't be a problem in
practice, since interface extension is a very closely coupled relationship, but
this may be something we will have to revisit in the future.

**Note:** This feature should be generalized to support implementing a
`structural interface`. The `impl` block would include definitions for any names
defined by the structural interface, and the result would be that the type
implements any interfaces that the structural interface requires (assuming this
doesn't leave any of those interface's requirements unimplemented). This
provides a tool useful for [evolution](#evolution).

**Concern:** Having both `extends` and [`extend`](#external-impl) with different
meanings is going to be confusing. One should be renamed.

TODO Use cases:
[Boost.Graph](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/) concepts,
see , C++ iterator concepts. See
[Carbon generics use case: graph library](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?usp=sharing&resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA).

To write an interface extending multiple interfaces, use multiple `extends`
declarations:

```
interface RefinesTwo {
  extends B1;
  extends B2;
}
```

The `extends` declarations are in the body of the `interface` definition instead
of the header so we can use
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

#### Covariant refinement

**Open question:** Can we redefine associated types in the refined interface as
long as the new definition is compatible but more specific ("covariance")? Here,
more specific means that the requirements and the name-to-binding map are
supersets (extending/refining the interface is sufficient).

```
interface ForwardIterator { ... }
interface BidirectionalIterator {
  extends ForwardIterator;
}
interface ForwardContainer {
  var ForwardIterator:$ IteratorType;
  method (Ptr(Self): this) Begin() -> IteratorType;
  method (Ptr(Self): this) End() -> IteratorType;
}
// Note: This should probably give a compile error complaining
// about an `IteratorType` name collision.
interface BidirectionalContainer {
  // Redeclaration of `IteratorType` with a more specific bound.
  var BidirectionalIterator:$ IteratorType;

  // Question: does this cause any weird shadowing?
  // Question: do we have to have a constraint equating
  // `IteratorType` and `ForwardContainer.IteratorType`?
  extends ForwardContainer;
}
```

One possible syntax would be to allow a block of these kinds of refinements in
place of a terminating semicolon (`;`) for `impl` and `extends` declarations in
an interface, as in:

```
interface BidirectionalContainer {
  extends ForwardContainer {
    // Redeclaration of `IteratorType` with a more specific bound.
    var BidirectionalIterator:$ IteratorType;
  }
}
```

another uses a [`where` clause](#where-clauses):

```
interface BidirectionalContainer {
  extends ForwardContainer
      where ForwardContainer.IteratorType is BidirectionalIterator;
}
```

or the argument passing approach would use an inferred variable:

```
interface BidirectionalContainer {
  // `Refined` is some new name so we don't collide with
  // `IteratorType`. The `[...]` mean this new name is
  // only used as a constraint, and is not part of the
  // `BidirectionalContainer` API.
  [var BidirectionalIterator:$ Refined];
  extends ForwardContainer(.IteratorType = Refined);
}
```

**Open question:** We may want to support refinement of other items as well,
such as methods. This would be part of matching the features of C++ `class`
inheritance.

#### Diamond dependency issue

Since types can implement interfaces at most once, we need to specify what
happens in when a type implements interfaces `D1` and `D2` both of which extend
`B`.

```
interface B {
  method (Self: this) B1();
  method (Self: this) B2();
}
interface D1 { extends B; ... }
interface D2 { extends B; ... }
struct U {
  impl D1 { ... }
  impl D2 { ... }
}
```

We can only have one definition of each method of `B`. Each method though could
be defined in `D1`, `D2`, or `B`. These would all be valid:

-   `D1` implements all methods of `B`, `D2` implements none of them.

```
struct U {
  impl D1 {
    method (Self: this) B1() { ... }
    method (Self: this) B2() { ... }
  }
  impl D2 { ... }
}
```

-   `D1` and `D2` implement all methods of `B` between them, but with no
    overlap.

```
struct U {
  impl D1 {
    method (Self: this) B1() { ... }
  }
  impl D2 {
    method (Self: this) B2() { ... }
  }
}
```

-   We explicitly implement `B`.

```
struct U {
  impl B {
    method (Self: this) B1() { ... }
    method (Self: this) B2() { ... }
  }
  impl D1 { ... }
  impl D2 { ... }
}
```

The `extends` declaration makes sense with the same meaning inside a
[`structural interface`](#structural-interfaces), and should also be supported.

### Use case: overload resolution

Implementing an extended interface is an example of a more specific match for
[lookup resolution](#lookup-resolution-and-specialization). For example, this
could be used to provide different implementations of an algorithm depending on
the capabilities of the iterator being passed in:

```
interface ForwardIterator(Type:$ T) { ... }
interface BidirectionalIterator(Type:$ T) {
  extends ForwardIterator(T);  ...
}
interface RandomAccessIterator(Type:$ T) {
  extends BidirectionalIterator(T); ...
}

fn SearchInSortedList[Comparable:$ T, ForwardIterator(T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  // does linear search
}
// Will prefer the following overload when it matches since it is more specific.
fn SearchInSortedList[Comparable:$ T, RandomAccessIterator(T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  // does binary search
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
fn PrintValue[
    Printable & Hashable:$ KeyT,
    Printable:$ ValueT](HashMap(KeyT, ValueT): map, KeyT: key) { ... }

var HashMap(String, Int): m;
PrintValue(m, "key");
```

## Adapting types

We also provide a way to create new types
[compatible with](terminology.md#compatible-types) existing types with different
APIs, in particular with different interface implementations, by
[adapting](terminology.md#adapting-a-type) them:

```
interface A { method (Self: this) F(); }
interface B { method (Self: this) G(); }
struct C {
  impl A { method (Self: this) F() { Print("CA"); } }
}
adapter D for C {
  impl B { method (Self: this) G() { Print("DB"); } }
}
adapter E for C {
  impl A { method (Self: this) F() { Print("EA"); } }
}
adapter F for C {
  impl A = E as A;  // Possibly we'd allow "impl A = E;" here.
  impl B = D as B;
}
```

This allows us to provide implementations of new interfaces (as in `D`), provide
different implementations of the same interface (as in `E`), or mix and match
implementations from other compatible types (as in `F`). The rules are:

-   You may only add APIs, not change the representation of the type, unlike
    extending a type where you may add fields.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `C`, `D`, `E`, and `F` end up compatible
    with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit casting between these
    types (unlike between a type and one of its facet types / impls).

The framework from the previous
[type compatibility section](#type-compatibility) allows us to evaluate when we
can cast between two different arguments to a parameterized type. Consider three
compatible types, all of which implement `Hashable`:

```
struct X {
  impl Hashable { ... }
  impl A { ... }
}
adapter Y for X {
  impl Hashable { ... }
}
adapter Z for X {
  impl Hashable = X as Hashable;
  impl B { ... }
}
```

Observe that `X as Hashable` is different from `Y as Hashable`, since they have
different definitions of the `Hashable` interface even though they are
compatible types since they use the same data representation. However
`X as Hashable` and `Z as Hashable` are almost the same. In addition to using
the same data representation, they both implement one interface, `Hashable`, and
use the same implementation for that interface. The one difference between them
is that `X as Hashable` may be implicitly cast to `X`, which implements
interface `A`, and `Z as Hashable` may be implicilty cast to `Z`, which
implements interface `B`. This means that it is safe to cast between
`HashMap(X, Int)` and `HashMap(Z, Int)` (though maybe only with an explicit
cast) but `HashMap(Y, Int)` is incompatible. This is a relief, because we know
that in practice the invariants of a `HashMap` implementation rely on the
hashing function staying the same.

**Comparison with other languages:** This matches the Rust construct called
`newtype`, which is used to implement traits on types while avoiding coherence
problems, see
[here](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-the-newtype-pattern-to-implement-external-traits-on-external-types)
and
[here](https://github.com/Ixrec/rust-orphan-rules#user-content-why-are-the-orphan-rules-controversial).

### Use case: Using independent libraries together

Imagine we have two packages that are developed independently. Package `A`
defines an interface `A.I` and a generic algorithm `A.F` that operates on types
that implement `A.I`. Package `B` defines a type `B.T`. Neither has a dependency
on the other, so neither package defines an implementation for `A.I` for type
`B.T`. A user that wants to pass a value of type `B.T` to `A.F` has to define an
adapter that provides an implementation of `A.I` for `B.T`:

```
import A;
import B;

adapter T for B.T {
  impl A.I { ... }
}
// Or, to keep the names from A.I out of T's API:
adapter T for B.T { }
extend T {
  impl A.I { ... }
}
```

The caller can either cast `B.T` values to `T` when calling `A.F` or just start
with `T` values in the first place.

```
var B.T: bt = ...;
A.F(bt as T);

var T: t = ...;
A.F(t);
```

**Open question:** This case is expected to be common, and to be convenient we
want it to be easy to retain the type-being-adapted's API, as much as possible.
We need some syntax for doing this, along with a way to make incremental
changes, such as to resolve conflicts. One idea is to use the `extends` keyword
instead of `for`:

```
adapter Foo extends Bar {
  // Include all of Bar's API and interface implementations.
}

adapter Foo2 extends Bar {
  // As above, but override implementation of `Baz` interface.
  impl Baz { ... }
}
```

**Open question:** Rust also uses the newtype idiom to create types with
additional invariants or other information encoded in the type
([1](https://doc.rust-lang.org/rust-by-example/generics/new_types.html),
[2](https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction),
[3](https://www.worthe-it.co.za/blog/2020-10-31-newtype-pattern-in-rust.html)).
This is used to record in the type system that some data has passed validation
checks, like `ValidDate` with the same data layout as `Date`. Or to record the
units associated with a value, such as `Seconds` versus `Milliseconds` or `Feet`
versus `Meters`. Perhaps only adapters that use `extends` should support
convenient casting?

### Example: Defining an impl for use by other types

Let's say we want to provide a possible implementation of an interface for use
by types for which that implementation would be appropriate. We can do that by
defining an adapter implementing the interface that is parameterized on the type
it is adapting. That impl may then be pulled in using the `"impl ... = ...;"`
syntax.

```
interface Comparable {
  fn operator<(Self: this, Self: that) -> Bool;
  ... // And also for >, <=, etc.
}
adapter ComparableFromDifferenceFn(Type:$ T,
                                   fnty(T, T)->Int:$ Difference)
    for T {
  impl Comparable {
    fn operator<(Self: this, Self: that) -> Bool {
      return Difference(this, that) < 0;
    }
    ... // And also for >, <=, etc.
  }
}
struct MyType {
  var Int: x;
  fn Difference(Self: this, Self: that) { return that.x - this.x; }
  impl Comparable = ComparableFromDifferenceFn(MyType, Difference) as Comparable;
}
```

## Associated constants

In addition to associated methods, we allow other kinds of associated items. For
consistency, we use the same syntax to describe a constant in an interface as in
a type without assigning a value. Since these are compile-time constants with
unknown value until compile time, they use the generic `:$` syntax. For example,
a fixed-dimensional point type could have the dimension as an associated
constant.

```
interface NSpacePoint {
  var Int:$ N;
  // The following require: 0 <= i < N.
  method (Ptr(Self): this) Get(Int: i) -> Float64;
  method (Ptr(Self): this) Set(Int: i, Float64 : value);
}
```

Implementations of `NSpacePoint` for different types might have different values
for `N`:

```
struct Point2D {
  impl NSpacePoint {
    var Int:$ N = 2;
    method (Ptr(Self): this) Get(Int: i) -> Float64 { ... }
    method (Ptr(Self): this) Set(Int: i, Float64: value) { ... }
  }
}

struct Point3D {
  impl NSpacePoint {
    var Int:$ N = 3;
    method (Ptr(Self): this) Get(Int: i) -> Float64 { ... }
    method (Ptr(Self): this) Set(Int: i, Float64: value) { ... }
  }
}
```

And these values may be accessed as follows:

```
Assert(Point2D.N == 2);
Assert(Point3D.N == 3);

fn PrintPoint[NSpacePoint:$ PointT](PointT: p) {
  for (var Int: i = 0; i < PointT.N; ++i) {
    if (i > 0) { Print(", "); }
    Print(p.Get(i));
  }
}

fn ExtractPoint[NSpacePoint:$ PointT](
    PointT: p,
    Ptr(Array(Float64, PointT.N)): dest) {
  for (var Int: i = 0; i < PointT.N; ++i) {
    (*dest)[i] = p.Get(i);
  }
}
```

To be consistent with normal function declaration syntax, function constants are
written:

```
interface DeserializeFromString {
  fn Deserialize(String: serialized) -> Self;
}

struct MySerializableType {
  var Int: i;

  impl DeserializeFromString {
    fn Deserialize(String: serialized) -> Self {
      return (.i = StringToInt(serialized));
    }
  }
}

var MySerializableType: x = MySerializableType.Deserialize("3");

fn Deserialize(DeserializeFromString:$ T, String: serialized) -> T {
  return T.Deserialize(serialized);
}
var MySerializableType: y = Deserialize(MySerializableType, "4");
```

**Aside:** In general, any field declared as "generic" (using the `:$` syntax),
will only have compile-time and not runtime storage associated with it.

**Comparison with other languages:** This feature is also called
[associated constants in Rust](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants).

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
  var Type:$ ElementType;
  method (Ptr(Self): this) Push(ElementType: value);
  method (Ptr(Self): this) Pop() -> ElementType;
  method (Ptr(Self): this) IsEmpty() -> Bool;
}
```

Here we have an interface called `StackAssociatedType` which defines two
methods, `Push` and `Pop`. The signatures of those two methods declared as
accepting or returning values with the type `ElementType`, which any implementer
of `StackAssociatedType` must also define. For example, maybe `DynamicArray`
implements `StackAssociatedType`:

```
struct DynamicArray(Type:$ T) {
  method (Ptr(Self): this) Begin() -> IteratorType;
  method (Ptr(Self): this) End() -> IteratorType;
  method (Ptr(Self): this) Insert(IteratorType: pos, T: value);
  method (Ptr(Self): this) Remove(IteratorType: pos);

  impl StackAssociatedType {
    var Type:$ ElementType = T;
    // `Self` and `DynamicArray(T)` are still equivalent here.
    method (Ptr(Self): this) Push(ElementType: value) {
      this->Insert(this->End(), value);
    }
    method (Ptr(Self): this) Pop() -> ElementType {
      var IteratorType: pos = this->End();
      Assert(pos != this->Begin());
      --pos;
      var ElementType: ret = *pos;
      this->Remove(pos);
      return ret;
    }
    method (Ptr(Self): this) IsEmpty() -> Bool {
      return this->Begin() == this->End();
    }
  }
}
```

Now we can write a generic function that operates on anything implementing the
`StackAssociatedType` interface, for example:

```
fn PeekAtTopOfStack[StackAssociatedType:$ StackType](Ptr(StackType): s)
    -> StackType.ElementType {
  var StackType.ElementType: top = s->Pop();
  s->Push(top);
  return top;
}

var DynamicArray(Int): my_array = (1, 2, 3);
// PeekAtTopOfStack's `StackType` is set to
// `DynamicArray(Int) as StackAssociatedType`.
// `StackType.ElementType` becomes `Int`.
Assert(PeekAtTopOfStack(my_array) == 3);
```

For context, see
["Interface type parameters versus associated types" in the Carbon: Generics Terminology doc](terminology.md#interface-type-parameters-versus-associated-types).

**Comparison with other languages:** Both
[Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189)
support associated types.

**Open question:**
[Swift allows the value of an associated type to be omitted when it can be determined from the method signatures in the implementation](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID190).
For the above example, this would mean figuring out `ElementType == T` from
context:

```
struct DynamicArray(Type:$ T) {
  // ...

  impl StackAssociatedType {
    // Not needed: var Type:$ ElementType = T;
    method (Ptr(Self): this) Push(T: value) { ... }
    method (Ptr(Self): this) Pop() -> T { ... }
    method (Ptr(Self): this) IsEmpty() -> Bool { ... }
  }
}
```

Should we do the same thing in Carbon? One concern is this might be a little
more complicated in the presence of method overloads with
[default implementations](interface-defaults), since it might not be clear how
they should match up, as in this example:

```
interface Has2OverloadsWithDefaults {
  var StackAssociatedType:$ T;
  method (Self: this) F(DynamicArray(T): x, T: y) { ... }
  method (Self: this) F(T: x, T.ElementType: y) { ... }
}

struct S {
  impl Has2OverloadsWithDefaults {
     // Unclear if T == DynamicArray(Int) or
     // T == DynamicArray(DynamicArray(Int)).
     method (Self: this) F(
         DynamicArray(DynamicArray(Int)): x,
         DynamicArray(Int): y) { ... }
  }
}
```

Not to say this can't be resolved, but it does add complexity.
[Swift considered](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#associated-type-inference)
removing this feature because it was the one thing in Swift that required global
type inference, which they otherwise avoided. They
[ultimately decided to keep the feature](https://github.com/apple/swift-evolution/blob/main/proposals/0108-remove-assoctype-inference.md).

### Model

The associated type is modeled by a witness table field in the interface.

```
interface Iterator {
  method (Ptr(Self): this) Advance();
}

interface Container {
  var Iterator:$ IteratorType;
  method (Ptr(Self): this) Begin() -> IteratorType;
}
```

is represented by:

```
struct Iterator(Type:$ Self) {
  var fnty(Ptr(Self): this): Advance;
  ...
}
struct Container(Type:$ Self) {
  // Representation type for the iterator.
  var Type:$ IteratorType;
  // Witness that IteratorType implements Iterator.
  var Ptr(Iterator(IteratorType)): iterator_impl;

  // Method
  var fnty (Ptr(Self): this) -> IteratorType: begin;
  ...
}
```

## Parameterized interfaces

Associated types don't change the fact that a type can only implement an
interface at most once. If instead you want a family of related interfaces, each
of which could be implemented for a given type, you could use parameterized
interfaces instead. To parameterized the stack interface instead of using
associated types, write a parameter list after the name of the interface:

```
interface StackParameterized(Type:$ ElementType) {
  method (Ptr(Self): this) Push(ElementType: value);
  method (Ptr(Self): this) Pop() -> ElementType;
  method (Ptr(Self): this) IsEmpty() -> Bool;
}
```

Then `StackParameterized(Fruit)` and `StackParameterized(Veggie)` would be
considered different interfaces, with distinct implementations.

```
struct Produce {
  var DynamicArray(Fruit): fruit;
  var DynamicArray(Veggie): veggie;
  impl StackParameterized(Fruit) {
    method (Ptr(Self): this) Push(Fruit: value) {
      this->fruit.Push(value);
    }
    method (Ptr(Self): this) Pop() -> Fruit {
      return this->fruit.Pop();
    }
    method (Ptr(Self): this) IsEmpty() -> Bool {
      return this->fruit.IsEmpty();
    }
  }
  impl StackParameterized(Veggie) {
    method (Ptr(Self): this) Push(Veggie: value) {
      this->veggie.Push(value);
    }
    method (Ptr(Self): this) Pop() -> Veggie {
      return this->veggie.Pop();
    }
    method (Ptr(Self): this) IsEmpty() -> Bool {
      return this->veggie.IsEmpty();
    }
  }
}
```

This approach is useful for the `ComparableTo(T)` interface, where a type might
be comparable with multiple other types, and in fact interfaces for
[operator overloads](#operator-overloading) more generally. Example:

```
interface EqualityComparableTo(Type:$ T) {
  fn operator==(Self: this, T: that) -> Bool;
  ...
}
struct Complex {
  var Float64: real;
  var Float64: imag;
  // Can implement this interface more than once as long as it has different
  // arguments.
  impl EqualityComparableTo(Complex) { ... }
  impl EqualityComparableTo(Float64) { ... }
}
```

All interface parameters must be marked as "generic", using the `:$` syntax.
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
interface Map(Type:$ FromType, Type:$ ToType) {
  method (Ptr(Self): this) Map(FromType: needle) -> Optional(ToType);
}
struct Bijection(Type:$ FromType, Type:$ ToType) {
  impl Map(FromType, ToType) { ... }
  impl Map(ToType, FromType) { ... }
}
// Error: Bijection has two impls of interface Map(String, String)
var Bijection(String, String): oops = ...;
```

In this case, it would be better to have an [adapting type](#adapting-types) to
contain the impl for the reverse map lookup, instead of implementing the `Map`
interface twice:

```
struct Bijection(Type:$ FromType, Type:$ ToType) {
  impl Map(FromType, ToType) { ... }
}
adapter ReverseLookup(Type:$ FromType, Type:$ ToType)
    for Bijection(FromType, ToType) {
  impl Map(ToType, FromType) { ... }
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
parameters could be inferred like associated types are. For example, we could
make a `Stack` interface that took a deducible `ElementType` parameter. You
would only be able to implement that interface once for a type, which would
allow you to infer the `ElementType` parameter like so:

```
fn PeekAtTopOfStack[Type:$ ElementType, Stack(ElementType):$ StackType]
    (Ptr(StackType): s) -> ElementType { ... }
```

This can result in more concise code for interfaces where you generally need to
talk about some parameter anytime you use that interface. For example,
`NTuple(N, type)` is much shorter without having to specify names with the
arguments.

**Rationale for the rejection:**

-   Having only one type of parameter simplifies the language.
-   Multi parameters express something we need, while deducible parameters can
    always be changed to associated types.
-   One implementation per interface & type parameter combination is more
    consistent with other parameterized constructs in Carbon. For example,
    parameterized types `Foo(A)` and `Foo(B)` are distinct, unconnected types.
-   It would be hard to give clear guidance on when to use associated types
    versus deducible type parameters, since which is best for a particular use
    is more of a subtle judgement call.
-   Deducible parameters
    [complicate the lookup rules for impls](appendix-interface-param-impl.md).
-   Deducible parameters in structural interfaces require additional rules to
    ensure they can be deduced unambiguously.

### Impl lookup

Let's say you have some interface `I(T, U(V))` being implemented for some type
`A(B(C(D), E))`. That impl must be defined in the same library that defines the
interface or one of the names needed by the type. That is, the impl must be
defined with one of `I`, `T`, `U`, `V`, `A`, `B`, `C`, `D`, or `E`. We further
require anything looking up this impl to import the _definitions_ of all of
those names. Seeing a forward declaration of these names is insufficient, since
you can presumably see forward declarations without seeing an impl with the
definition. This accomplishes a few goals:

-   The compiler can check that there is only one definition of any impl that is
    actually used, avoiding
    [One Definition Rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule)
    problems.
-   Every attempt to use an impl will see the exact same impl, making the
    interpretation and semantics of code consistent no matter its context, in
    accordance with the FIXME
    [Refactoring principle](https://github.com/josh11b/carbon-lang/blob/principle-refactoring/docs/project/principles/principle-refactoring.md).
-   Allowing the impl to be defined with either the interface or the type
    addresses the
    [expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

Note that [the rules for specialization](#lookup-resolution-and-specialization)
do allow there to be more than one impl to be defined for a type, as long as one
can unambiguously be picked as most specific.

### Parameterized structural interfaces

We should also allow the [structural interface](#structural-interfaces)
construct to support parameters. Parameters would work the same way as for
regular (nominal/non-structural) interfaces.

## Constraints

TODO: Fix this up a lot

### Contexts where you might need constraints

-   In a declaration of a function, type, interface, or impl.
-   Within the body of an interface definition.
-   Naming a new type-type that represents the constraint (typically an `alias`
    or `structural interface` definition).

To handle this last use case, we expand the kinds of requirements that
type-types can have from just interface requirements to also include the various
kinds of constraints discussed later in this section.

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
// Constraints on a function:
fn F[D:$ V](V: v) where ... { ... }

// Constraints on a type parameter:
struct S(B:$ T) where ... { ... }

// Constraints on an interface parameter:
interface A(B:$ T) where ... {
  // Constraints on an associated type or constant:
  var C:$ U where ...;
  // Constraints on a method:
  method (Self: this) G[D:$ V](V: v) where ...;
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
-   Awkward to produce type-types that have specified values for all associated
    types for use with `DynPtr` and `DynBox`.
-   Can introduce some inconsistency/redundancy with how interface parameters
    are specified.
-   Adds a redundant way of expressing some constraints.

#### Argument passing

This approach is to constrain the inputs to the thing being constrained. It
requires that anything that might be constrained be able to be specified as an
input. In particular, this means the associated types of an interface can be
specified as optional named arguments.

In this approach, the constraints come before the thing being constrainted.

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
    some" `[...]` inferred variables
    ([1](#type-bounds-on-associated-types-in-interfaces),
    [2](#same-type-constraints)).
-   Type-types would become callable, and when called would return a type-type
    that could then be called again.

### Constraint use cases

#### Set to a specific value

Useful for associated constants and associated types, with little difference
between them.

##### Associated constants

For [associated constants](#associated-constants), we might need to write a
function that only works with a specific value of `N`. We can solve this using
argument passing, as in:

```
fn PrintPoint2D[NSpacePoint(.N = 2):$ PointT](PointT: p) {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Or `where` clauses:

```
fn PrintPoint2D[NSpacePoint:$ PointT](PointT: p) where PointT.N == 2 {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Similarly in an interface definition:

```
interface {
  // Argument passing:
  var NSpacePoint(.N = 2): PointT;
  // versus `where` clause:
  var NSpacePoint: PointT where PointT.N == 2;
}
```

To name such a constraint:

```
// Argument passing:
alias Point2DInterface = NSpacePoint(.N = 2);
structural interface Point2DInteface {
  extends NSpacePoint(.N = 2);
}

// versus `where` clause:
alias Point2DInterface = NSpacePoint where Point2D.N == 2;
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
  var Type:$ ElementType;
  ...
}
interface Container {
  var Type:$ ElementType;
  // Argument passing:
  var Iterator(.ElementType = ElementType):$ IteratorType;
  // versus `where` clause:
  var Iterator:$ IteratorType where IteratorType.ElementType == ElementType;
  ...
}
```

Functions accepting a generic type might also want to constrain an associated
type. For example, we might want to have a function only accept stacks
containing integers:

```
// Argument passing:
fn SumIntStack[Stack(.ElementType = Int):$ T](Ptr(T): s) -> Int {
// versus `where` clause:
fn SumIntStack[Stack:$ T](Ptr(T): s) -> Int where T.ElementType == Int {

// Same implementation in either case:
  var Int: sum = 0;
  while (!s->IsEmpty()) {
    sum += s->Pop();
  }
  return sum;
}
```

To name these sorts of constraints, we could use `alias` statements or
`structural interface` definitions.

```
// Argument passing:
alias IntStack = Stack(.ElementType = Int);
structural interface IntStack {
  extends Stack(.ElementType = Int);
}

// versus `where` clause:
alias IntStack = Stack where IntStack.ElementType == Int;
structural interface IntStack {
  extends Stack where Stack.ElementType == Int;
}
```

[Rust uses trait aliases](https://rust-lang.github.io/rfcs/1733-trait-alias.html)
for this case.

##### Concern

Sometimes we may need a single type-type without any parameters or unspecified
associated constants/types, such as to define a `DynPtr(TT)` (as
[described in the following dynamic pointer type section](#dynamic-pointer-type)).
To do this with a `where` clause approach you would have to name the constraint
before using it, an annoying extra step. For Rust, it is part of
[the motivation of supporting argument passing to set values of associated types](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types).

#### Range constraints on associated constants

**Concern:** It is difficult to express mathematical constraints on values in
the argument passing framework. For example, the constraint "`NTuple` where `N`
is at least 2" naturally translates into a `where` clause:

```
fn TakesAtLeastAPair[Int:$ N](NTuple(N, Int): x) where N >= 2 { ... }
```

Similarly for now we only have a `where` clause formulation for constraining the
`N` member of `NSpacePoint` from
[the "associated constants" section](#associated-constants)

```
fn PrintPoint2Or3[NSpacePoint:$ PointT](PointT: p)
  where 2 <= PointT.N, PointT.N <= 3 { ... }
```

The same syntax would be used in an interface definition:

```
interface HyperPointInterface {
  var Int:$ N where N > 3;
  method (Ptr(Self): this) Get(Int: i) -> Float64;
}
```

or naming this kind of constraint:

```
alias HyperPoint = NSpacePoint where Point2Or3.N > 3;
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
fn SortContainer[Comparable:$ ElementType,
                 Container(.ElementType = ElementType):$ ContainerType]
    (Ptr(ContainerType): container_to_sort);
```

You might read this as "for some `ElementType` of type `Comparable`, ...".

To do this with a `where` clause, we need some way of saying a type bound, which
unfortunately is likely to be redundant and inconsistent with how it is said
outside of a `where` clause.

**Open question:** How do you spell that? This proposal provisionally uses `is`,
which matches Swift, but maybe we should have another operator that more clearly
returns a boolean like `has_type`?

```
fn SortContainer[Container:$ ContainerType]
    (Ptr(ContainerType): container_to_sort)
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
[other type-type](#adapting-types)). For example, we might say interface
`Container` has a `Begin` method returning values with type satisfying the
`Iterator` interface:

```
interface Iterator {
  method (Ptr(Self): this) Advance();
  ...
}
interface Container {
  var Iterator:$ IteratorType;
  method (Ptr(Self): this) Begin() -> IteratorType;
  ...
}
```

With this additional information, a function can now call `Iterator` methods on
the return value of `Begin`:

```
fn OneAfterBegin[Container:$ T](Ptr(T): c) -> T.IteratorType {
  var T.IteratorType: iter = c->Begin();
  iter.Advance();
  return iter;
}
```

##### Naming type bound constraints

Given these definitions (omitting `ElementType` for brevity):

```
interface IteratorInterface { ... }
interface ContainerInterface {
  var IteratorInterface:$ IteratorType;
  ...
}
interface RandomAccessIterator {
  extends IteratorInterface;
  ...
}
```

We would like to be able to define a `RandomAccessContainer` to be a type-type
whose types satisfy `ContainerInterface` with an `IteratorType` satisfying
`RandomAccessIterator`.

**Concern:** We would need to introduce some sort of "for some" operator to
support this with argument passing. We might use a `[...]` to indicate that the
introduced parameter is inferred.

```
// Argument passing:
fn F[RandomAccessIterator:$ IterType,
     ContainerInterface(.IteratorType=IterType):$ ContainerType]
    (ContainerType: c);
// versus `where` clause:
fn F[ContainerInterface:$ ContainerType](ContainerType: c)
    where ContainerType.IteratorType is RandomAccessIterator;

// WANT a definition of RandomAccessContainer such that the above
// is equivalent to:
fn F[RandomAccessContainer:$ ContainerType](ContainerType: c);

// Argument passing:
alias RandomAccessContainer =
    [RandomAccessIterator:$ IterType]
    ContainerInterface(.IteratorType=IterType);
// versus `where` clause:
alias RandomAccessContainer = ContainerInterface
    where RandomAccessContainer.IteratorType is RandomAccessIterator;
```

##### Type bound with interface argument

Use case: we want a function that can take two values `x` and `y`, with
potentially different types, and multiply them. So `x` implements the
`MultipliesBy(R)` interface for `R`, the type of `y`.

```
fn F[Type:$ R, MutipliesBy(R):$ L](L: x, R: y) {
  x * y;
}
```

#### Same type constraints

```
interface PairInterface {
  var Type:$ Left;
  var Type:$ Right;
}

// Argument passing:
fn F[Type:$ T, PairInterface(.Left = T, .Right = T):$ MatchedPairType]
    (Ptr(MatchedPairType): x);

// versus `where` clause:
fn F[PairInterface:$ MatchedPairType](Ptr(MatchedPairType): x)
    where MatchedPairType.Left == MatchedPairType.Right;
```

Constraint in an interface definition:

Argument passing approach needs the "for some" `[...]` syntax for inferred
associated types that don't introduce new names into the interface just to
represent the constraint.

```
// Argument passing:
interface HasEqualPair {
  [var Type:$ T];
  var PairInterface(.Left = T, .Right = T):$ P;
}

// versus `where` clause:
interface HasEqualPair {
  var PairInterface:$ P where P.Left == P.Right;
}
```

##### Naming same type constraints

Again, the argument passing approach also needs the "for some" `[...]` syntax
for inferred associated types. Otherwise this first `EqualPair` interface would
only match types that had a type member named `T`.

```
// Argument passing:
alias EqualPair = [var Type:$ T]
    PairInterface(.Left = T, .Right = T);
structural interface EqualPair {
  [var Type:$ T];
  extends PairInterface(.Left = T, .Right = T);
}

// versus `where` clause:
alias EqualPair = PairInterface
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
fn EqualContainers[HasEquality:$ ET,
                   Container(.ElementType = ET):$ CT1,
                   Container(.ElementType = ET):$ CT2]
    (Ptr(CT1): c1, Ptr(CT2): c2) -> Bool;

interface HasEqualContainers {
  [var HasEquality:$ ET];
  var Container(.ElementType = ET):$ CT1;
  var Container(.ElementType = ET):$ CT2;
}

// versus `where` clause:
fn EqualContainers[Container:$ CT1, Container:$ CT2]
    (Ptr(CT1): c1, Ptr(CT2): c2) -> Bool
    where CT1.ElementType == CT2.ElementType,
          CT1.ElementType as HasEquality;

interface HasEqualContainers {
  var Container:$ CT1 where CT1.ElementType as HasEquality;
  var Container:$ CT2 where CT1.ElementType == CT2.ElementType,
}
```

#### Rejected alternative: `ForSome(F)`

Another way to solve the [type bounds](#type-bounds) and
[same type](#same-type-constraints) constraint use cases using argument passing
without the "for some" `[...]` operator would be to have a `ForSome(F)`
construct, where `F` is a function from types to type-types.

> `ForSome(F)`, where `F` is a function from type `T` to type-type `TT`, is a
> type whose values are types `U` with type `TT=F(T)` for some type `T`.

**Example:** Pairs of values where both values have the same type might be
written as

```
fn F[ForSome(lambda (Type:$ T) =>
        PairInterface(.Left = T, .Right = T)):$ MatchedPairType]
    (Ptr(MatchedPairType): x) { ... }
```

This would be equivalent to:

```
fn F[Type:$ T, PairInterface(T, T):$ MatchedPairType]
    (Ptr(MatchedPairType): x) { ... }
```

**Example:** Containers where the elements implement the `HasEquality` interface
might be written as:

```
fn F[ForSome(lambda (HasEquality:$ T) =>
        Container(.ElementType = T)):$ ContainerType]
    (Ptr(ContainerType): x) { ... }
```

This would be equivalent to:

```
fn F[HasEquality:$ T, Container(T):$ ContainerType]
  (Ptr(ContainerType): x) { ... }
```

#### Is a subtype

**Concern:** We need to add some operator to express this, with either argument
passing or where clauses.

For `where` clause we could represent this by a binary `extends` operator
returning a boolean. For argument passing, we'd introduce an `Extends(T)`
type-type, whose values are types that extend `T`, that is types `U` that are
subtypes of `T`.

```
// Argument passing:
fn F[Extends(BaseType):$ T](Ptr(T): p);
fn UpCast[Type:$ U, Extends(U):$ T](Ptr(T): p, U) -> Ptr(U);
fn DownCast[Type:$ T](Ptr(T): p, Extends(T):$ U) -> Ptr(U);

// versus `where` clause:
fn F[Type:$ T](Ptr(T): p) where T extends BaseType;
fn UpCast[Type:$ T](Ptr(T): p, Type:$ U) -> Ptr(U) where T extends U;
fn DownCast[Type:$ T](Ptr(T): p, Type:$ U) -> Ptr(U) where U extends T;
```

In Swift, you can
[add a required superclass to a type bound using `&`](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282).

#### Parameterized type implements interface

TODO: This use case was part of the
[Rust rationale for adding support for `where` clauses](https://rust-lang.github.io/rfcs/0135-where.html#motivation).

**Concern:** Right now this is only easily expressed using `where` clauses.

```
// Some parametized type.
struct Vector(Type:$ T) { ... }

// Parameterized type implements interface only for some arguments.
extend Vector(String) {
  impl Printable { ... }
}

// Constraint: `T` such that `Vector(T)` implements `Printable`
fn PrintThree[Type:$ T](T: a, T: b, T: c) where Vector(T) is Printable {
  var Vector(T): v = (a, b, c);
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
  var Numeric:$ MagnitudeType;
  method (Self: this) Abs() -> MagnitudeType;
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
fn Relu[HasAbs(.MagnitudeType = .Self):$ T](T: x) {
  // T.MagnitudeType == T so the following is allowed:
  return (x.Abs() + x) / 2;
}
fn UseContainer[Container(.SliceType = .Self):$ T](T: c) -> Bool {
  // T.SliceType == T so `c` and `c.Slice(...)` can be compared:
  return c == c.Slice(...);
}

// versus `where` clause
fn Relu[HasAbs:$ T](T: x) where T.MagnitudeType == T {
  return (x.Abs() + x) / 2;
}
fn UseContainer[Container:$ T](T: c) -> Bool where T.SliceType == T {
  return c == c.Slice(...);
}
```

Interface definition:

```
interface Container {
  var Type:$ ElementType;

  // Argument passing:
  var Container(.ElementType = ElementType, .SliceType = .Self):$ SliceType;
  // versus `where` clause
  var Container:$ SliceType where SliceType.ElementType == ElementType,
                                  Slicetype.SliceType == SliceType;

  method (Ptr(Self): this) GetSlice(IteratorType: start,
                                    IteratorType: end) -> SliceType;
}
```

Naming these constraints:

```
// Argument passing
alias RealAbs = HasAbs(.MagnitudeType = .Self);
structural interface RealAbs {
  extends HasAbs(.MagnitudeType = Self);
}
alias ContainerIsSlice = Container(.SliceType = .Self);
structural interface ContainerIsSlice {
  extends Container(.SliceType = Self);
}

// versus `where` clause
alias RealAbs = HasAbs where RealAbs.MagnitudeType == RealAbs;
structural interface RealAbs {
  extends HasAbs where HasAbs.MagnitudeType == Self;
}
alias ContainerIsSlice = Container
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
`Container`, you'd have to allow the it do be defined inline in the `Container`
definition:

```
interface Container {
  var Type:$ ElementType;

  structural interface ContainerIsSlice {
    extends Container where Container.SliceType == Self;
  }
  var ContainerIsSlice(.ElementType = ElementType):$ SliceType;

  method (Ptr(Self): this) GetSlice(IteratorType: start,
                                    IteratorType: end) -> SliceType;
}
```

**Rejected alternative:** If we were to write variable declarations with the
name first instead of the type, we could use that name inside the type
declaration, as in `T:$ HasAbs(.MagnitudeType = T)`.

#### Type inequality

TODO: inequality type constraints (for example "type is not `Bool`").

You might need an inequality type constraint, for example, to control overload
resolution:

```
fn F[Type:$ T](T: x) -> T { return x; }
fn F(Bool: x) -> String {
  if (x) return "True"; else return "False";
}

fn G[Type:$ T](T: x) -> T {
  // We need a T != Bool constraint for this to type check.
  return F(x);
}
```

Another use case for inequality type constraints would be to say something like
"define `ComparableTo(T1)` for `T2` if `ComparableTo(T2)` is defined for `T1`
and `T1 != T2`".

**Concern:** Right now this is only easily expressed using `where` clauses.

```
fn G[Type:$ T](T: x) -> T where T != Bool { return F(x); }
```

### Implicit constraints

Imagine we have a generic function that accepts a arbitrary `HashMap`:

```
fn LookUp[Type:$ KeyType](Ptr(HashMap(KeyType, Int)): hm,
                          KeyType: k) -> Int;

fn PrintValueOrDefault[Printable:$ KeyType,
                       Printable & HasDefault:$ ValueT]
    (HashMap(KeyType, ValueT): map, KeyT: key);
```

The `KeyType` in these declarations does not satisfy the requirements of
`HashMap`, which requires the type to at least implement `Hashable` and probably
others like `Sized`, `EqualityComparable`, `Movable`, and so on.

```
struct HashMap(
    Hashable & Sized & EqualityComparable & Movable:$ KeyType,
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

**Caveat:** These constraints can be obscured:

```
interface I(Type:$ A, Type:$ B, Type:$ C, Type:$ D, Type:$ E) {
  var I(B, A, C, D, E):$ SwapType;
  var I(B, C, D, E, A):$ CycleType;
  fn LookUp(Ptr(HashMap(D, E)): hm) -> E;
  fn Foo(Bar(A, B): x);
}
```

All type arguments to "I" must actually implement `Hashable` (since
[an adjacent swap and a cycle generate the full symmetry group on 5 elements](https://www.mathcounterexamples.net/generating-the-symmetric-group-with-a-transposition-and-a-maximal-length-cycle/)).
And additional restrictions on those types depend on the definition of `Bar`.
For example, this definition

```
struct Bar(Type:$ A, ComparableWith(A):$ B) { ... }
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

### Generic type equality

Imagine we have some function with generic parameters:

```
fn F1[SomeInterface:$ T](T: x) {
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
fn F2[A:$ Z, B(.Y = Z, .X = Z.W):$ V](...) { ... }
```

We require the following rules to be enforced by the language definition and
compiler:

-   No forward references in a function declaration.
-   No forward references between items in an `interface` definition.
-   No implicit type equality constraints.

From these rules, we derive rules about which type expressions are canonical.
The first rule is:

> For purposes of type checking a function, the names of types declared in the
> function declaration to the right of the `:$` are all canonical.

This is because these can all be given distinct types freely, only their
associated types can be constrained to be equal to some other type. In this
example, this means that the types `Z` and `V` are both canonical. The second
rule comes from there being no forward references in declarations, and no
impliict type equality constraints:

> No declaration can affect type equality for any declaration to its left.

This means that the canonical types for type expressions starting with `Z.` are
completely determined by the declaration `A:$ Z`. Furthermore, since the set of
type expressions starting with `Z.` might be infinite, we adopt the lazy
strategy of only evaluating expressions that are needed for something explicitly
mentioned.

We do need to evaluate `Z.W` though for the `B(.Y = Z, .X = Z.W):$ V`
expression. This is an easy case, though since `A:$ Z` doesn't include any
assignments to any associated types. In this case, the associated types of `A`
are all canonical. An alias defined in `A` would of course not be, it would be
set to the canonical type for whatever it is an alias for. For example:

```
interface A {
  // `W` is canonical.
  var A:$ W;
  // `U` is not canonical, is equal to `W.W`.
  alias U = W.W;
  // `T` is canonical, but `T.Y` is not.
  var B(.Y = Self):$ T;
}
```

Next lets examine the definition of `B` so we can resolve expressions starting
with `V.`.

```
interface B {
  var A:$ S;
  var A(.W = S):$ Y;
  var A:$ X;
  var B(.X = S):$ R;
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
  var Broken:$ Q;
  var Broken(.R = Q.R.R):$ R;
}

fn F[Broken:$ T](T: x) {
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

The fix for this situation is to introduce new inferred associated types:

```
interface Fixed {
  [var Fixed:$ RR];
  [var Fixed(.R = RR):$ QR];
  var Fixed(.R = QR):$ Q;
  var Fixed(.R = RR):$ R;
}

fn F[Fixed:$ T](T: x) {
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
fn F3[A:$ N, A:$ P, B(.S = N, .Y = P):$ Q](...) { ... }
```

The compiler is required to report an error rejecting this declaration. This is
because the constraints declared in `B` require that `Q.Y.W == Q.S == N` so
`P.Y == N`. This violates the "no implicit type equality constraint" rule since
`P` is not declared with any constraint forcing that to hold. We can't let `Q`'s
declaration affect earlier declarations, otherwise our algorithm would
potentially have to resolve cycles. The compiler should recommend the user
rewrite their code to:

```
fn F3[A:$ N, A(.Y = N):$ P, B(.S = N, .Y = P):$ Q](...) { ... }
```

This resolves the issue, and with this change the compiler can now correctly
determine canonical types.

**Note:** This algorithm still works with the `.Self` feature from the
["recursive constraints" section](#recursive-constraints). For example, the
expression `var A(.X = .Self):$ Y` means `Y.X == Y` and so the `.Self` on the
right-side represents a shorter and earlier type expression. This precludes
introducing a loop and so is safe.

**Open question:** Can we relax any of the restrictions? For example, perhaps we
would like to allow items in an interface to reference each other, as in:

```
interface D {
  var A(.W = V):$ E;
  var A(.W = E):$ V;
}
```

In this case `D.E.W == D.F` and `D.F.W == D.E` and we would need some way of
deciding which were canonical (probably `D.E` and `D.F`). This would have to be
restricted to cases where the expression on the right has no `.` to avoid cycles
or type expression that grow without bound. Another concern is if there are type
constructors involved:

```
interface Graph {
  var A(.W = Vector(Verts)):$ Edges;
  var A(.W = Vector(Edges)):$ Verts;
}
```

**Open question:** Is this expressive enough to represent the equality
constraints needed by users in practice?

##### Canonical types and type checking

TODO: For assignment to type check, argument has to have the same or a more
restrictive type-type than the parameter. This means that the canonical type
expression would have the right (most restrictive) type-type to use for all
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
  var ...:$ A;
  var Z:$ B where B.X == ..., B.Y == ...;
  var ...:$ C
}
```

These forms of `where` clauses are allowed because we can rewrite them into the
argument passing form:

| `where` form                   | argument passing form   |
| ------------------------------ | ----------------------- |
| `var Z:$ B where B.X == A`     | `var Z(.X = A):$ B`     |
| `var Z:$ B where B.X == A.T.U` | `var Z(.X = A.T.U):$ B` |
| `var Z:$ B where B.X == Self`  | `var Z(.X = Self):$ B`  |
| `var Z:$ B where B.X == B`     | `var Z(.X = .Self):$ B` |

Note that the second example would not be allowed if `A.T.U` had type `Foo`, to
avoid non-terminating recursion.

These forms of `where` clauses are forbidden:

| Example forbidden `where` form           | Rule                                     |
| ---------------------------------------- | ---------------------------------------- |
| `var Z:$ B where B == ...`               | must have a dot on left of `==`          |
| `var Z:$ B where B.X.Y == ...`           | must have a single dot on left of `==`   |
| `var Z:$ B where A.X == ...`             | `A` ≠ `B` on left of `==`                |
| `var Z:$ B where B.X == ..., B.X == ...` | no two constraints on same member        |
| `var Z:$ B where B.X == B.Y`             | right side can't refer to members of `B` |
| `var Z:$ B where B.X == C`               | no forward reference                     |

There is some room to rewrite other `where` expressions into allowed argument
passing forms. One simple example is allowing the two sides of the `==` in one
of the allowed forms to be swapped, but more complicated rewrites may be
possible. For example,

```
var Z:$ B where B.X == B.Y;
```

might be rewritten to:

```
[var ...:$ XY];
var Z(.X = XY, .Y = XY):$ B;
```

except it may be tricky in general to find a type for `XY` that satisfies the
constraints on both `B.X` and `B.Y`. Similarly,

```
var ...:$ A;
var Z:$ B where B == A.T.U
```

might be rewritten as:

```
var ...:$ A;
alias B = A.T.U;
```

unless the type bounds on `A.T.U` do not match the `Z` bound on `B`. In that
case, we need to find a type-type `Z2` that represents the intersection of the
two type constraints and a different rewrite:

```
var Z2:$ B
[var ...(.U = B):$ AT];
var ...(.T = AT):$ A;
```

**Note:** It would be great if the
['&' operator for type-types](#combining-interfaces-by-anding-type-types) was
all we needed to define the intersection of two type constraints, but it isn't
yet defined for two type-types that have the same interface but with different
constraints. And that requires being able to automatically combine constraints
of the form `B.X == Foo` and `B.X == Bar`.

**Open question:** How much rewriting can be done automatically?

**Open question:** Is there a simple set of rules explaining which `where`
clauses are allowed that we could explain to users?

#### Manual type equality

TODO

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

## Conditional conformance

[The problem](terminology.md#conditional-conformance) we are trying to solve
here is expressing that we have an impl of some interface for some type, but
only if some additional type restrictions are met. To do this, we leverage
[external impl](#external-impl):

-   We can provide the same impl argument in two places to constrain them to be
    the same.
-   We can declare the impl argument with a more-restrictive type, to for
    example say this impl can only be used if that type satisfies an interface.

**Example:** [Interface constraint] Here we implement the `Printable` interface
for arrays of `N` elements of `Printable` type `T`, generically for `N`.

```
interface Printable {
  method (Ptr(Self): this) Print() -> String;
}
struct FixedArray(Type:$ T, Int:$ N) { ... }

// By saying "Printable:$ T" instead of "Type:$ T" here, we constrain
// T to be Printable for this impl.
extend FixedArray(Printable:$ T, Int:$ N) {
  impl Printable {
    method (Ptr(Self): this) Print() -> String {
      var Bool: first = False;
      var String: ret = "";
      for (auto: a) in *this {
        if (!first) {
          ret += ", ";
        }
        ret += a.Print();
      }
      return ret;
    }
  }
}
```

**Example:** [Same-type constraint] We implement interface `Foo(T)` for
`Pair(T, U)` when `T` and `U` are the same.

```
interface Foo(Type:$ T) { ... }
struct Pair(Type:$ T, Type:$ U) { ... }
extend Pair(Type:$ T, T) {
  impl Foo(T) { ... }
}
// Alternatively:
extend Pair[Type:$ T](T, T) {
  impl Foo(T) { ... }
}
```

**Proposal:** [Other boolean condition constraints] Just like we support
conditions when pattern matching (for example in overload resolution), we should
also allow them when defining an impl:

```
extend[Type:$$ T] T if (sizeof(T) <= 16) {
  impl Foo { ... }
}
```

**Concern:** The conditional conformance feature makes the question "is this
interface implemented for this type" undecidable in general.
[This feature in Rust has been shown to allow implementing a Turing machine](https://sdleffler.github.io/RustTypeSystemTuringComplete/).
This means we will likely need some heuristic like a limit on how many steps of
recursion are allowed.

**Open question:** We could also have a syntax for defining these impls inline
in the struct definition, but I haven't found a syntax that works as well as the
`extend` statement constructs shown above. We have a choice between two evils:
either the meaning of the names defined in the outer struct change inside the
conditional impl, or we end up having to give new names to the same values. This
was discussed in
[Carbon meeting Nov 27, 2019 on Generics & Interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.gebr4cdi0y8o -->.

The best idea I've had so far is to include an `extend` statement inside the
`struct` definition to make it inline:

```
struct FixedArray(Type:$ T, Int:$ N) {
  // A few different syntax possibilities here:
  extend FixedArray(Printable:$ P, Int:$ N2) { impl Printable { ... } }
  extend FixedArray(Printable:$ P, N) { impl Printable { ... } }
  extend[Printable:$ P] FixedArray(P, N) { impl Printable { ... } }
}

struct Pair(Type:$ T, Type:$ U) {
  extend Pair(T, T) { impl Foo(T) { ... } }
}
```

We would need rules to prevent inconsistent reuse of the names from the outer
scope. This proposal has the desirable property that the syntax for internal
versus external conditional conformance matches. This makes it straightforward
to refactor between those two choices, and is easier to learn.

Some other ideas we have considered lack this consistency:

-   One approach would be to use implicit arguments in square brackets after the
    `impl` keyword, and an `if` clause to add constraints:

```
struct FixedArray(Type:$ T, Int:$ N) {
  impl[Printable:$ U] Printable if T == U {
    // Here `T` and `U` have the same value and so you can freely
    // cast between them. The difference is that you can call the
    // `Print` method on values of type `U`.
  }
}

struct Pair(Type:$ T, Type:$ U) {
  impl Foo(T) if T == U {
    // Can cast between `Pair(T, U)` and `Pair(T, T)` since `T == U`.
  }
}
```

-   Another approach is to use pattern matching instead of boolean conditions.
    This might look like (though it introduces another level of indentation):

```
struct FixedArray(Type:$ T, Int:$ N) {
  @if let Printable:$ P = T {
    impl Printable { ... }
  }
}

interface Foo(Type:$ T) { ... }
struct Pair(Type:$ T, Type:$ U) {
  @if let Pair(T, T):$ P = Self {
    impl Foo(T) { ... }
  }
}
```

We can have this consistency, but lose the property that all unqualified names
for a type come from its `struct definition:

-   We could keep `extend` statements outside of the struct block to avoid
    sharing names between scopes, but allow them to have an `internal` keyword
    (as long as it is in the same library).

```
struct FixedArray(Type:$ T, Int:$ N) { ... }
extend FixedArray(Printable:$ P, Int:$ N) internal { impl Printable { ... } }

struct Pair(Type:$ T, Type:$ U) { ... }
extend Pair(Type:$ T, T) internal { ... }
```

Lastly, we could adopt a "flow sensitive" approach, where the meaning of names
can change in an inner scope. This would allow the `if` conditions that govern
when an impl is used to affect the types in that impl's definition:

```
struct FixedArray(Type:$ T, Int:$ N) {
  impl Printable if (T implements Printable) {
    // Inside this scope, `T` has type `Printable` instead of `Type`.
  }
}
```

This would require mechanisms to both describe these conditions and determine
how they affect types.

**Future work:**
[Rust uses this mechanism](https://doc.rust-lang.org/book/ch10-02-traits.html#using-trait-bounds-to-conditionally-implement-methods)
to also allow conditionally defining methods on types, independent of
interfaces. This is not specific to generics, but we would like to have a syntax
that is consistent across these two use cases. The above "inline `extend`"
syntax does naturally generalize to this case:

```
struct FixedArray(Type:$ T, Int:$ N) {
  // ...
  extend[Printable:$ P] FixedArray(P, N) {
    method (Ptr(Self): this) Print() { ... }
  }
}

// FixedArray(T, N) has a `Print()` method if `T` is `Printable`.
```

## Templated impls for generic interfaces

TODO: This section should be rewritten to be about parameterized `impl` in
general, not just templated. For example, the
["lookup resolution and specialization" section](#lookup-resolution-and-specialization)
is applicable broadly.

Some things going on here:

-   Our syntax for `extend` statements already allows you to have a templated
    type parameter. This can be used to provide a general impl that depends on
    templated access to the type, even though the interface itself is defined
    generically.
-   We very likely will want to restrict the impl in some ways.
    -   Easy case: An impl for a family of parameterized types.
    -   Trickier is "structural conformance": we might want to say "here is an
        impl for interface `Foo` for any class implementing a method `Bar`".
        This is particularly for C++ types.

### Structural conformance

**Question:** How do you say: "restrict this impl to types that have a member
function with a specific name & signature"?

An important use case is to restrict templated definitions to an appropriate set
of types.

**Rejected alternative:** We don't want to support the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
of C++ because it does not let the user clearly express the intent of which
substitution failures are meant to be constraints and which are bugs.
Furthermore, the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
leads to problems where the constraints can change accidentally as part of
modifications to the body that were not intended to affect the constraints at
all. As such, constraints should only be in the impl signature rather than be
determined by anything in the body.

**Rejected alternative:** We don't want anything like `LegalExpression(...)` for
turning substitution success/failure into True/False at this time, since we
believe that it introduces a lot of complexity, and we would rather lean on
conforming to an interface or the reflection APIs. However, we feel less
strongly about this position than the previous position and we may revisit (say
because of needing a bridge for C++ users). One nice property of the
`LegalExpression(...)` paradigm for expressing a constraint is that it would be
easy for the constraint to mirror code from the body of the function.

**Additional concern:** We want to be able to express "method has a signature"
in terms of the types involved, without necessarily any legal example values for
those types. For example, we want to be able to express that "`T` is
constructible from `U` if it has a `operator create` method that takes a `U`
value", without any way to write down an particular value for `U` in general:

```
interface ConstructibleFrom(...:$ Args) { ... }
extend[Type:$$ T, Type:$$ U] T if (LegalExpression(T.operator create(???))) {
  impl ConstructibleFrom(U) { ... }
}
```

This is a problem for the `LegalExpression(...)` model, another reason to avoid
it.

**Answer:** We will use the planned
[method constraints extension](#future-work-method-constraints) to
[structural interfaces](#structural-interfaces).

This is similar to the structural interface matching used in the Go language. It
isn't clear how much we want to encourage it, but it does have some advantages
with respect to decoupling dependencies, breaking cycles, cleaning up layering.
Example: two libraries can be combined without knowing about each other as long
as they use methods with the same names and signatures.

```
structural interface HasFooAndBar {
  method (Self: this) Foo(Int: _) -> String;
  method (Self: this) Bar(String: _) -> Bool;
}

fn CallsFooAndBar[HasFooAndBar:$$ T]
    (T: x, Int: y) -> Bool {
  return x.Bar(x.Foo(y));
}
```

One downside of this approach is that it nails down the type of `this`, even
though multiple options would work in a template. We might need to introduce
additional optiion in the syntax only for use with templates:

```
structural interface HasFooAndBar {
  method (_) Foo(Int: _) -> String;
  method (_) Bar(String: _) -> Bool;
}
```

Note that this would be awkward for generics to support the dynamic compilation
strategy, and we don't expect to want to hide the difference between read-only
and mutable in a generic context.

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
interface SInterface(Type:$ T) {
  method (Ptr(Self): this) F(Ptr(T): t);
}
```

and once we implement that interface for the C++ type `S`:

```
// Note: T has to be a templated argument to be usable with the
// C++ template `S`. There is no problem passing a template
// argument `T` to the generic argument of `SInterface`.
extend C++::S(Type:$$ T) {
  impl SInterface(T) {
    method (Ptr(Self): this) F(Ptr(T): t) { this->F(t); }
  }
}
```

we can then call it from a generic Carbon function:

```
fn G[Type:$ T, SInterface(T):$ SType](Ptr(SType): s, Ptr(T): t) {
  s->F(t);
}
var C++::S(Int) : x;
var Int : y = 3;
G(&x, &y);  // C++::S(Int) implements SInterface(Int) by way of templated impl
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
interface Optional(Type:$ T) { ... }
extend C++::std::optional(Type:$$ T) {
  impl Optional(T) { ... }
}
```

### Subtlety around interfaces with parameters

Since interfaces with parameters can have multiple implementations for a single
type, it opens the question of how they work when implementing one interface in
terms of another.

**Open question:** We could allow templated impls to take each of these multiple
implementations for one interface and manufacture an impl for another interface,
as in this example:

```
// Some interfaces with type parameters.
interface EqualityComparableTo(Type:$ T) { ... }
// Types can implement templated interfaces more than once as long as the
// templated arguments differ.
struct Complex {
  var Float64: r;
  var Float64: i;
  impl EqualityComparableTo(Complex) { ... }
  impl EqualityComparableTo(Float64) { ... }
}
// Some other interface with a type parameter.
interface Foo(Type:$ T) { ... }
// This provides an impl of Foo(T) for U if U is EqualityComparableTo(T).
// In the case of Complex, this provides two impls, one for T == Complex,
// and one for T == Float64.
extend [EqualityComparableTo(Type:$$ T):$ U] U {
  impl Foo(T) { ... }
}
```

One tricky part of this is that you may not have visibility into all the impls
of an interface for a type since they may be
[defined with one of the other types involved](#impl-lookup). Hopefully this
isn't a problem -- you will always be able to see the _relevant_ impls given the
types that have been imported / have visible definitions.

### Lookup resolution and specialization

**Rule:** Can have multiple impl definitions that match, as long as there is a
single best match. Best is defined using the "more specific" partial ordering:

-   Matching a descendant type or descendant interface is more specific and
    therefore a closer match than a parent type or interface.
-   Matching an exact type (`Foo` or `Foo(Bar)`) is more specific than a
    parameterized family of types (`Foo(T)` for any type `T`) is more specific
    than a generic type (`T` for any type `T`).
-   A more restrictive constraint is more specific. In particular, a type-type
    `T` is more restrictive than a type-type `U` if the set of restrictions for
    `T` is a superset of those for `U`. So `Foo(T)` for
    `Comparable & Printable:$ T` is more specific than `Comparable:$ T`, which
    is more specific than `Type:$ T`.
-   TODO: others?

The ability to have a more specific implementation used in place of a more
general is commonly called _[specialization](terminology.md#specialization)_.

TODO: Examples

**Implication:** Can't do impl lookup with generic arguments, even if you can
see a matching templated definition, since there may be a more-specific match
and we want to be assured that we always get the same result any time we do an
impl lookup.

TODO: Example

**Open question:** Rust
[doesn't allow specialization of a general implementation unless its items are marked `default`](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#the-default-keyword).
Is that something we want to require? This would allow us to relax the
restriction of the previous implication for items that may not be specialized.

**Future Work:** Rust's
[specialization rules](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#the-default-keyword)
allow you to omit definitions in the more specific implementation, using the
definition in the more general implementation as a default. It furthermore
supports defining a general
[`default impl`](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#default-impls)
that is incomplete (in fact this feature was originally called "partial impl")
and _only_ used to provide default implementations for more specialized
implementations.

**Comparison with other languages:** See
[Rust's rules for deciding which impl is more specific](https://rust-lang.github.io/rfcs/1210-impl-specialization.html#defining-the-precedence-rules).

## Other constraints as type-types

### Type compatible with another type

Given a type `U`, define the type-type `CompatibleWith(U)` as follows:

> `CompatibleWith(U)` is a type whose values are types `T` such that `T` and `U`
> are [compatible](terminology.md#compatible-types). That is values of types `T`
> and `U` can be cast back and forth without any change in representation (for
> example `T` is an [adapter](#adapting-types) for `U`).

To support this, we extend the requirements that type-types are allowed to have
to include a "data representation requirement" option.

**Note:** Just like interface parameters, we require the user to supply `U`,
they may not be inferred. Specifically, this code would be illegal:

```
fn Illegal[Type:$ U, CompatibleWith(U):$ T](Ptr(T): x) ...
```

In general there would be multiple choices for `U` given a specific `T` here,
and no good way of picking one. However, similar code is allowed if there is
another way of determining `U`:

```
fn Allowed[Type:$ U, CompatibleWith(U):$ T](Ptr(U): x, Ptr(T): y) ...
```

#### Example: Multiple implementations of the same interface

This allows us to represent functions that accept multiple implementations of
the same interface for a type.

```
enum CompareResult { Less, Equal, Greater }
interface Comparable {
  method (Self: this) Compare(Self: that) -> CompareResult;
}
fn CombinedLess[Type:$ T](T: a, T: b,
                          CompatibleWith(T) & Comparable:$ U,
                          CompatibleWith(T) & Comparable:$ V) -> Bool {
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
struct Song { ... }
adapter SongByArtist for Song { impl Comparable { ... } }
adapter SongByTitle for Song { impl Comparable { ... } }
assert(CombinedLess(Song(...), Song(...), SongByArtist, SongByTitle) == True);
```

We might generalize this to a list of implementations:

```
fn CombinedCompare[Type:$ T]
    (T: a, T: b, List(CompatibleWith(T) & Comparable):$ CompareList)
    -> CompareResult {
  for (auto: U) in CompareList {
    var CompareResult: result = (a as U).Compare(b);
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
adapter ThenCompare(Type:$ T,
                    List(CompatibleWith(T) & Comparable):$ CompareList) for T {
  impl Comparable {
    method (Self: this) Compare(Self: that) -> CompareResult {
      for (auto : U) in CompareList {
        var CompareResult: result = (this as U).Compare(that);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

alias SongByArtistThenTitle = ThenCompare(Song, (SongByArtist, SongByTitle));
var Song: song = ...;
var SongByArtistThenTitle: song2 = Song(...) as SongByArtistThenTitle;
assert((song as SongByArtistThenTitle).Compare(song2) == CaompareResult.Less);
```

### Sized types and type-types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`Int32`, `Float64`, etc.) most other concrete types (for
    example most FIXME
    [structs](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md)).
-   It could be known generically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a
    [dynamic type](#dynamic-pointer-type) such as `Dynamic(TT)`, a FIXME
    [variable-sized type](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#control-over-allocation),
    or you could dereference a pointer to a base type that could actually point
    to a FIXME
    [descendant](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#question-extension--inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

I'm going to call a type "sized" if it is in the first two categories, and
"unsized" otherwise. (Note: something with size 0 is still considered "sized".)
The type-type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring (member/local) variables
of that type, taking values of that type as parameters, returning values of that
type, and defining arrays of that type. There will be other requirements as
well, such as being movable, copyable, or constructible from some types.

Example:

```
interface Foo {
  impl DefaultConstructible;  // See "interface requiring other interfaces".
}
struct Bar {  // Structs are "sized" by default.
  impl Foo;
}
fn F[Foo: T](Ptr(T): x) {  // T is unsized.
  var T: y;  // Illegal: T is unsized.
}
// T is sized, but its size is only known generically.
fn G[Foo & Sized: T](Ptr(T): x) {
  var T: y = *x;  // Allowed: T is sized and default constructible.
}
var Bar: z;
G(&z);  // Allowed: Bar is sized and implements Foo.
```

**Note:** The compiler will determine which types are "sized", this is not
something types will implement explicitly like ordinary interfaces.

**Open question:** Even if the size is fixed, it won't be known at the time of
compiling the generic function if we are using the dynamic strategy. Should we
automatically [box](#boxed) local variables when using the dynamic strategy? Or
should we only allow `MaybeBox` values to be instantiated locally?

**Open question:** Should the `Sized` type-type expose an associated constant
with the size? So you could say `T.ByteSize` in the above example to get a
generic int value with the size of `T`.

#### Model

This requires a special integer field be included in the witness table type to
hold the size of the type. This field will only be known generically, so if its
value is used for type checking, we need some way of evaluating those type tests
symbolically.

## Dynamic types

Two different goals here:

-   Reducing code size at the expense of more runtime dispatch.
-   Increasing expressivity by allowing types to vary at runtime, AKA
    "[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch)".

We address these two different use cases with two different mechanisms. What
they have in common is using a runtime/dynamic type value (using
`InterfaceName: type_name`, no `$`) instead of a generic type value (using
`InterfaceName:$ type_name`, with a `$`). In the first case,
[we make the type parameter to a function dynamic](#runtime-type-parameters). In
the second case,
[we use a dynamic type value as a field in a struct](#runtime-type-fields). In
both cases, we have a name bound to a runtime type value, which is modeled by a
[dynamic-dispatch witness table](terminology.md#dynamic-dispatch-witness-table)
instead of the
[static-dispatch witness table](terminology.md#static-dispatch-witness-table)
used with generic type values.

### Runtime type parameters

If we pass in a type as an ordinary parameter (using `:` instead of `:$`), this
means passing the witness table as an ordinary parameter -- that is a
[dynamic-dispatch witness table](terminology.md#dynamic-dispatch-witness-table)
-- to the function. This means that there will be a single copy of the generated
code for this parameter.

**Restriction:** The type's size will only be known at runtime, so patterns that
use a type's size such as declaring local variables of that type or passing
values of that type by value are forbidden. Essentially the type is considered
[unsized](#sized-types-and-type-types), even if the type-type is `Sized`.

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
all of the type-type's parameters and associated types be specified, since they
can not vary at runtime.

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
Swift has
[considered switching to this approach](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--clarifying-existentials),
using the keyword "`any`", instead of Rust's "`dyn`".

#### Dynamic pointer type

Given a type-type `TT` (with some restrictions described below), define
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
  method (Ptr(Self): this) Print();
}
struct AnInt {
  var Int: x;
  impl Printable { method (Ptr(Self): this) Print() { PrintInt(this->x); } }
}
struct AString {
  var String: x;
  impl Printable { method (Ptr(Self): this) Print() { PrintString(this->x); } }
}

var AnInt: i = (.x = 3);
var AString: s = (.x = "Hello");

var DynPtr(Printable): i_dynamic = &i;
i_dynamic->Print();  // Prints "3".
var DynPtr(Printable): s_dynamic = &s;
s_dynamic->Print();  // Prints "Hello".

var DynPtr(Printable)[2]: dynamic = (&i, &s);
for (DynPtr(Printable): iter) in dynamic {
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
  method (Ptr(Self): this) IsEqual(Ptr(Self): that) -> Bool;
}
```

and implementations of this interface for our two types:

```
impl EqualCompare for AnInt {
  method (Ptr(AnInt): this) IsEqual(Ptr(AnInt): that) -> Bool {
    return this->x == that->x;
  }
}
impl EqualCompare for AString {
  method (Ptr(AString): this) IsEqual(Ptr(AString): that) -> Bool {
    return this->x == that->x;
  }
}
```

Now given two values of type `Dynamic(EqualCompare)`, what happens if we try and
call `IsEqual`?

```
var DynPtr(EqualCompare): i_dyn_eq = &i;
var DynPtr(EqualCompare): s_dyn_eq = &s;
i_dyn_eq->IsEqual(&*s_dyn_eq);  // Unsound: runtime type confusion
s_dyn_eq->IsEqual(&*i_dyn_eq);  // Unsound: runtime type confusion
```

For `*i_dyn_eq` to implement `EqualCompare.IsEqual`, it needs to accept any
`Ptr(DynPtr(EqualCompare).T)` value for `that`, including `&*s_dyn_eq`. But
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

##### Model

TODO

```
// Note: InterfaceType is essentially "TypeTypeType".
struct DynPtr(InterfaceType:$$ TT) {  // TT is any interface
  struct DynPtrImpl {
    private TT: t;
    private Ptr(Void): p;  // Really Ptr(t) instead of Ptr(Void).
    impl TT {
      // Defined using meta-programming.
      // Forwards this->F(...) to (this->p as Ptr(this->t))->F(...)
      // or equivalently, this->t.F(this->p as Ptr(this->t), ...).
    }
  }
  var TT:$ T = (DynPtrImpl as TT);
  private DynPtrImpl: impl;
  fn operator->(Ptr(Self): this) -> Ptr(T) { return &this->impl; }
  fn operator=[TT:$ U](Ptr(Self): this, Ptr(U): p) { this->impl = (.t = U, .p = p); }
}
```

#### Deref

To make a function work on either regular or dynamic pointers, we define an
interface `Deref(T)` that both `DynPtr` and `Ptr(T)` implement:

```
// Types implementing `Deref(T)` act like a pointer to `T`.
interface Deref(Type:$ T) {
  // This is used for the `->` and `*` dereferencing operators.
  method (Self: this) Deref() -> Ptr(T);
  impl Copyable;
  impl Movable;
}

// Implementation of Deref() for DynPtr(TT).
extend DynPtr(InterfaceType:$$ TT) {
  impl Deref(DynPtr(TT).DynPtrImpl as TT) { ... }
}
// or equivalently:
extend DynPtr(InterfaceType:$$ TT) {
  impl Deref(DynPtr(TT).T) { ... }
}

// Implementation of Deref(T) for Ptr(T).
extend Ptr(Type:$ T) {
  impl Deref(T) {
    method (Ptr(T): this) Deref() -> Ptr(T) { return this; }
  }
}
```

Now we can implement a function that takes either a regular pointer to a type
implementing `Printable` or a `DynPtr(Printable)`:

```
// This is equivalent to `fn PrintIt[Printable:$ T](T*: x) ...`,
// except it also accepts `DynPtr(Printable)` arguments.
fn PrintIt[Printable:$ T, Deref(T) & Sized: PtrT](PtrT: x) {
  x->Print();
}
PrintIt(&i); // T == (AnInt as Printable), PtrT == T*
// Prints "3"
PrintIt(&s); // T == (AString as Printable), PtrT == T*
// Prints "Hello"
PrintIt(dynamic[0]);  // T == DynPtr(Printable).T, PtrT == DynPtr(Printable)
// Prints "3"
PrintIt(dynamic[1]);  // T == DynPtr(Printable).T, PtrT == DynPtr(Printable)
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
struct Boxed(Type:$ T,
             // May be able to add more constraints on AllocatorType (like
             // sized & movable) so we could make it a generic argument?
             AllocatorInterface:$$ AllocatorType = DefaultAllocatorType) {
  private var T*: p;
  private var AllocatorType: allocator;
  operator create(T*: p, AllocatorType: allocator = DefaultAllocator) { ... }
  impl Movable { ... }
}

// TODO: Should these just be constructors defined within Boxed(T)?
// If T is constructible from X, then Boxed(T) is constructible from X ...
extend Boxed(ConstructibleFrom(...:$$ Args): T) {
  impl ConstructibleFrom(Args) { ... }
}
// ... and Boxed(X) as well.
extend Boxed(ConstructibleFrom(...:$$ Args): T) {
  impl ConstructibleFrom(Boxed(Args)) { ... }
}

// This allows you to create a Boxed(T) value inferring T so you don't have to
// say it explicitly.
fn Box[Type:$ T](T*: x) -> Boxed(T) { return Boxed(T)(x); }
fn Box[Type:$ T, AllocatorInterface:$$ AllocatorType]
    (T*: x, AllocatorType: allocator) -> Boxed(T, AllocatorType) {
  return Boxed(T, AllocatorType)(x, allocator);
}
```

NOTE: Chandler requests that boxing be explicit so that the cost of indirection
is visible in the source (and in fact visible wherever the dereference happens).
This solution also accomplishes that but may not address all use cases for
boxing.

#### DynBoxed

`DynBoxed(TT)` is to `Boxed(T)` as `DynPtr(TT)` is to `Ptr(T)`. Like
`DynPtr(TT)`, it holds a pointer to a value of any type `T` that satisfies the
interface `TT`. Like `Boxed(T)`, it owns that pointer.

TODO

```
struct DynBoxed(InterfaceType:$$ TT,
                AllocatorInterface:$$ AllocatorType = DefaultAllocatorType) {
  private DynPtr(TT): p;
  private var AllocatorType: allocator;
  ...  // Constructors, etc.
  // Destructor deallocates this->p.
  impl Movable { ... }
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
    definitely unsized). Performs
    [dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch).
-   `T`: Regular values that are sized and movable. The extra
    indirection/pointer and heap allocation for putting `T` into a box would
    introduce too much overhead / cost.

In all cases we end up with a sized, movable value that is not very large. Just
like we did with <code>[Deref(T) above](#deref)</code>, we can create an
interface to abstract over the differences, called <code>MaybeBoxed(T)</code>:

```
interface MaybeBoxed(Type:$ T) {
  fn operator->(Ptr(Self): this) -> Ptr(T);
  // plus other smart pointer operators
  impl Movable;
  // We require that MaybeBoxed(T) should be sized, to avoid all
  // users having to say so
  impl Sized;
}

extend Boxed(Type:$ T) {
  impl MaybeBoxed(T {
    fn operator->(Ptr(Self): this) -> Ptr(T) { return this->p; }
  }
}

extend DynBoxed(InterfaceType:$$ TT) {
  impl MaybeBoxed(DynBoxed(TT).T) {
    ...  // TODO
  }
}
```

For the case of values that we can efficiently move without boxing, we implement
a new type `NotBoxed(T)` that adapts `T` and so has the same representation and
supports zero-runtime-cost casting.

```
// Can pass a T to a function accepting a MaybeBoxed(T) value without boxing by
// first casting it to NotBoxed(T), as long as T is sized and movable.
adapter NotBoxed(Movable & Sized:$ T) for T {  // :$ or :$$ here?
  impl Movable = T as Movable;
  impl MaybeBoxed(T) {
    fn operator->(Ptr(Self): this) -> Ptr(T) { return this as Ptr(T); }
  }
}
// TODO: Should this just be a constructor defined within NotBoxed(T)?
// Says NotBoxed(T) is constructible from a value of type Args if T is.
extend NotBoxed(ConstructibleFrom(...:$$ Args):$ T) {
  impl ConstructibleFrom(Args) { ... }
}

// This allows you to create a NotBoxed(T) value inferring T so you don't have
// to say it explicitly. TODO: Could probably replace "Type:$$ T" with
// "Movable & Sized:$ T", here.
fn DontBox[Type:$$ T](T: x) -> NotBoxed(T) inline { return x as NotBoxed(T); }
// Use NotBoxed as the default implementation of MaybeBoxed for small & movable
// types. TODO: Not sure how to write a size <= 16 bytes constraint here.
extend[Movable & Sized:$$ T] T if (sizeof(T) <= 16) {
  impl MaybBoxed(T) = NotBoxed(T);
}
```

This allows us to write a single generic function using that interface and have
the caller decide which of these mechanisms is the best fit for the specific
types being used.

```
interface Foo { method (Ptr(Self): this) F(); }
fn UseBoxed[Foo:$ T, MaybeBoxed(T):$ BoxType](BoxType: x) {
  x->F();  // Possible indirection is visible
}
struct Bar { impl Foo { ... } }
var DynBoxed(Foo): y = new Bar(...);
UseBoxed(y);
// DontBox might not be needed, if Bar meets the requirements to use the
// default NotBoxed impl of MaybeBox.
UseBoxed(DontBox(Bar()));
```

## Future work

### Abstract return types

This lets you return am anonymous type implementing an interface from a
function.
[Rust has this feature](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).
Also see:

-   [https://rust-lang.github.io/rfcs/1951-expand-impl-trait.html]
-   [https://rust-lang.github.io/rfcs/2071-impl-trait-existential-types.html]
-   [https://rust-lang.github.io/rfcs/2515-type_alias_impl_trait.html]

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

One variation on this may be default implementations of entire interfaces. For
example, `RandomAccessContainer` refines `Container` with an `IteratorType`
satisfying `RandomAccessIterator`. That is sufficient to provide a default
implementation of the indexing operator (operator `[]`), by way of
[implementing an interface](#operator-overloading).

```
interface RandomAccessContainer {
  extends Container {
    // Refinement of the associated type `IteratorType` from `Container`.
    var RandomAccessIterator:$ IteratorType;
  }
  // Either `impl` or `extends` here, depending if you want
  // `RandomAccessContainer`'s API to include these names.
  impl OperatorIndex(Int) {
    // Default implementation of interface.
    method (Self: this) Get(Int: i) -> ElementType {
      return (this.Begin() + i).Get();
    }
    method (Ptr(Self): this) Set(Int: i, ElementType: value) {
      (this->Begin() + i).Set(value);
    }
  }
}
```

### Evolution

Being able to decorate associated items with `upcoming`, `deprecated`, etc. to
allow for transition periods when items are being added or removed.

As an alternative, users could version their interfaces explicitly. For example,
if we had an interface `Foo` with a method `F`:

```
interface Foo {
  method (Self: this) F() -> Int;
}
```

and we want to add a method `G` with a default implementation, we do so in an
interface with a new name:

```
interface Foo2 {
  method (Self: this) F() -> Int;
  method (Self: this) G() -> Int { return this.F() + 1; }
}

structural interface Foo {
  impl Foo2;
  alias F = Foo2.F;
}
```

Since `Foo` is now a structural interface, implementing `Foo2` means you
automatically also implement `Foo`. Further, implementing `Foo` implements
`Foo2` with the default implementation of `G`. So, any function requiring an
implementation of either `Foo` or `Foo2` is satisfied by any type implementing
either of those. This would allow an incremental transition from `Foo` to
`Foo2`.

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
fn F(A: a, B: b, ..., Addable(A, B):$ T) where (A, B) is Addable(A, B) {
  ((A, B) as T).DoTheAdd(x, y)
}
```

There are a couple of problems with the idea of an interface implemented for the
`(LeftType, RightType)` tuple. How does the impl get passed in? How do you say
that an interface is only for pairs? These problems suggest it is not worth
trying to do anything special for this edge case. Rust considered this approach
and instead decided that the left-hand type implements the interface and the
right-hand type is a parameter that defaults to being equal to the left-hand
type.

```
interface Addable(Type:$ Right = Self) {
  // Assuming we allow defaults for associated types.
  var Type:$ AddResult = Self;
  fn Add(Self: lhs, Right: rhs) -> AddResult;
}
```

### Impls with state

Impls where the impl itself has state. (from @zygoloid). Use case: implementing
interfaces for a flyweight in a Flyweight pattern where the Impl needs a
reference to a key -> info map.

### Generic associated types

TODO: Used for "property maps" in the Boost.Graph library.

See
[Carbon generics use case: graph library](https://docs.google.com/document/d/1xk0GLtpBl2OOnf3F_6Z-A3DtTt-r7wdOZ5wPipYUSO0/edit?usp=sharing&resourcekey=0-mBSmwn6b6jwbLaQw2WG6OA)
for context.

Rust has been working toward adding this feature
([1](https://github.com/rust-lang/rust/issues/44265),
[2](https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md),
[3](https://github.com/rust-lang/rfcs/pull/1598),
[4](https://www.fpcomplete.com/blog/monads-gats-nightly-rust/)). It has been
proposed for Swift as well
([1](https://forums.swift.org/t/idea-generic-associated-types/5422),
[2](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#generic-associatedtypes),
[3](https://forums.swift.org/t/generic-associated-type/17831)). This corresponds
roughly to
[member templates in C++](https://en.cppreference.com/w/cpp/language/member_template).

### Higher-ranked types

A solution to the problem posed
[here (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.qvhzlz54obmt -->,
where we need a representation for a way to go from a type to an implementation
of an interface parameterized by that type. Examples of things we might want to
express:

-   This priority queue's second argument (`QueueLike`) is a function that takes
    a type `U` and returns a type that implements `QueueInterface(U)`:

```
struct PriorityQueue(
    Type:$ T, fn (Type:$ U)->QueueInterface(U):$ QueueLike) {
  ...
}
```

-   Map takes a container of type `T` and function from `T` to `V` into a
    container of type `V`:

```
fn Map[Type:$ T,
       fn (Type:$ U)->StackInterface(U):$ StackLike,
       Type:$ V]
    (Ptr(StackLike(T)): x, fn (T)->V: f) -> StackLike(V) { ... }
```

TODO: Challenging! Probably needs something like
[Dependent function types](https://en.wikipedia.org/wiki/Dependent_type#Pi_type)

Generic associated types and higher-ranked (or is it higher-kinded?) types solve
the same problem in two different contexts. Generic associated types are about
members of interfaces and higher-ranked types are about function parameters.

Swift proposals:
[1](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#higher-kinded-types),
[2](https://forums.swift.org/t/higher-kinded-types-monads-functors-etc/4691),
[3](https://forums.swift.org/t/proposal-higher-kinded-types-monads-functors-etc/559),
[4](https://github.com/typelift/swift/issues/1). These correspond roughly to
[C++ template template parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Template_template_parameter).

### Inferring associated types

Imagine we have an interface that has an associated type used in the signature
of one of its methods:

```
interface A {
  var Type:$ T;
  method (Self: this) F() -> T;
}
```

And we have a type implementing that interface:

```
struct S {
  impl A {
    // var Type:$ T = Int;
    method (Self: this) F() -> Int { return 3; }
  }
}
```

The compiler could infer the type to assign to the associated type by matching
the definition of `F`. This is
[supported in Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID190),
but may be tricky in the presence of function overloading or something where we
would prefer the user to be explicit (so, for example, the name `T` was visible
in the definition of `S`).

### Field requirements

To match the expressivity of inheritance, we might want to allow interfaces to
express the requirement that any implementing type has a particular field. We
might want to restrict what can be done with that field, using capabilities like
"read", "write", and "address of" (which implies read and write). Swift also has
a "modify" capability implemented using coroutines, without requiring there be a
value of the right type we can take the address of. If we do expose an "address
of" capability, it will have to be a real address since we don't expect any sort
of proxy to be able to be used instead.

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
should support partial specializations like `MyType(Ptr(T))`. Main problem is
supporting this with the dynamic strategy.

### Bridge for C++ customization points

See details in [the goals document](goals.md#bridge-for-c-customization-points).

### Reverse generics for return types

In Rust this is
[return type of "`impl Trait`"](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).
In Swift,
[this feature is in discussion](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--reverse-generics).
Swift is considering spelling this `<V: Collection> V` or `some Collection`.

## Notes

These are notes from discussions after this document was first written that have
not yet been incorporated into the main text above.

-   Can use IDE tooling to show all methods including external impl,
    automatically switching to [qualified member names](#qualified-member-names)
    where needed to get that method.
-   Address use cases in [the motivation document](motivation.md).
-   Want inheritance with virtual functions to be modeled by interface
    extension. Example showing the interaction between Dynamic pointer types and
    interface extension.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
