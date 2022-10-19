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
    -   [Implementing multiple interfaces](#implementing-multiple-interfaces)
    -   [External impl](#external-impl)
    -   [Qualified member names and compound member access](#qualified-member-names-and-compound-member-access)
    -   [Access](#access)
-   [Generics](#generics)
    -   [Return type](#return-type)
    -   [Implementation model](#implementation-model)
-   [Interfaces recap](#interfaces-recap)
-   [Type-of-types](#type-of-types)
-   [Named constraints](#named-constraints)
    -   [Subtyping between type-of-types](#subtyping-between-type-of-types)
-   [Combining interfaces by anding type-of-types](#combining-interfaces-by-anding-type-of-types)
-   [Interface requiring other interfaces](#interface-requiring-other-interfaces)
    -   [Interface extension](#interface-extension)
        -   [`extends` and `impl` with named constraints](#extends-and-impl-with-named-constraints)
        -   [Diamond dependency issue](#diamond-dependency-issue)
    -   [Use case: overload resolution](#use-case-overload-resolution)
-   [Adapting types](#adapting-types)
    -   [Adapter compatibility](#adapter-compatibility)
    -   [Extending adapter](#extending-adapter)
    -   [Use case: Using independent libraries together](#use-case-using-independent-libraries-together)
    -   [Use case: Defining an impl for use by other types](#use-case-defining-an-impl-for-use-by-other-types)
    -   [Use case: Private impl](#use-case-private-impl)
    -   [Use case: Accessing external names](#use-case-accessing-external-names)
    -   [Adapter with stricter invariants](#adapter-with-stricter-invariants)
-   [Associated constants](#associated-constants)
    -   [Associated class functions](#associated-class-functions)
-   [Associated types](#associated-types)
    -   [Implementation model](#implementation-model-1)
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
    -   [Referencing names in the interface being defined](#referencing-names-in-the-interface-being-defined)
    -   [Manual type equality](#manual-type-equality)
        -   [`observe` declarations](#observe-declarations)
-   [Other constraints as type-of-types](#other-constraints-as-type-of-types)
    -   [Is a derived class](#is-a-derived-class)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Same implementation restriction](#same-implementation-restriction)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other impls](#example-creating-an-impl-out-of-other-impls)
    -   [Sized types and type-of-types](#sized-types-and-type-of-types)
        -   [Implementation model](#implementation-model-2)
    -   [`TypeId`](#typeid)
    -   [Destructor constraints](#destructor-constraints)
-   [Generic `let`](#generic-let)
-   [Parameterized impls](#parameterized-impls)
    -   [Impl for a parameterized type](#impl-for-a-parameterized-type)
    -   [Conditional conformance](#conditional-conformance)
        -   [Conditional methods](#conditional-methods)
    -   [Blanket impls](#blanket-impls)
        -   [Difference between blanket impls and named constraints](#difference-between-blanket-impls-and-named-constraints)
    -   [Wildcard impls](#wildcard-impls)
    -   [Combinations](#combinations)
    -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
        -   [Type structure of an impl declaration](#type-structure-of-an-impl-declaration)
        -   [Orphan rule](#orphan-rule)
        -   [Overlap rule](#overlap-rule)
        -   [Prioritization rule](#prioritization-rule)
        -   [Acyclic rule](#acyclic-rule)
        -   [Termination rule](#termination-rule)
    -   [`final` impls](#final-impls)
        -   [Libraries that can contain `final` impls](#libraries-that-can-contain-final-impls)
    -   [Comparison to Rust](#comparison-to-rust)
-   [Forward declarations and cyclic references](#forward-declarations-and-cyclic-references)
    -   [Declaring interfaces and named constraints](#declaring-interfaces-and-named-constraints)
    -   [Declaring implementations](#declaring-implementations)
    -   [Matching and agreeing](#matching-and-agreeing)
    -   [Declaration examples](#declaration-examples)
    -   [Example of declaring interfaces with cyclic references](#example-of-declaring-interfaces-with-cyclic-references)
    -   [Interfaces with parameters constrained by the same interface](#interfaces-with-parameters-constrained-by-the-same-interface)
-   [Interface members with definitions](#interface-members-with-definitions)
    -   [Interface defaults](#interface-defaults)
    -   [`final` members](#final-members)
-   [Interface requiring other interfaces revisited](#interface-requiring-other-interfaces-revisited)
    -   [Requirements with `where` constraints](#requirements-with-where-constraints)
-   [Observing a type implements an interface](#observing-a-type-implements-an-interface)
    -   [Observing interface requirements](#observing-interface-requirements)
    -   [Observing blanket impls](#observing-blanket-impls)
-   [Operator overloading](#operator-overloading)
    -   [Binary operators](#binary-operators)
    -   [`like` operator for implicit conversions](#like-operator-for-implicit-conversions)
-   [Parameterized types](#parameterized-types)
    -   [Specialization](#specialization)
-   [Future work](#future-work)
    -   [Dynamic types](#dynamic-types)
        -   [Runtime type parameters](#runtime-type-parameters)
        -   [Runtime type fields](#runtime-type-fields)
    -   [Abstract return types](#abstract-return-types)
    -   [Evolution](#evolution)
    -   [Testing](#testing)
    -   [Impls with state](#impls-with-state)
    -   [Generic associated types and higher-ranked types](#generic-associated-types-and-higher-ranked-types)
        -   [Generic associated types](#generic-associated-types)
        -   [Higher-ranked types](#higher-ranked-types)
    -   [Field requirements](#field-requirements)
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
pass in `T` explicitly, so it can be a
[deduced parameter](terminology.md#deduced-parameter) (also see
[deduced parameters](overview.md#deduced-parameters) in the Generics overview
doc). Basically, the user passes in a value for `val`, and the type of `val`
determines `T`. `T` still gets passed into the function though, and it plays an
important role -- it defines the key used to look up interface implementations.

We can think of the interface as defining a struct type whose members are
function pointers, and an implementation of an interface as a value of that
struct with actual function pointer values. An implementation is a table mapping
the interface's functions to function pointers. For more on this, see
[the implementation model section](#implementation-model).

In addition to function pointer members, interfaces can include any constants
that belong to a type. For example, the
[type's size](#sized-types-and-type-of-types) (represented by an integer
constant member of the type) could be a member of an interface and its
implementation. There are a few cases why we would include another interface
implementation as a member:

-   [associated types](#associated-types)
-   [type parameters](#parameterized-interfaces)
-   [interface requirements](#interface-requiring-other-interfaces)

The function expresses that the type argument is passed in
[statically](terminology.md#static-dispatch-witness-table), basically generating
a separate function body for every different type passed in, by using the
"generic argument" syntax `:!`, see [the generics section](#generics) below. The
interface contains enough information to
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

The last piece of the puzzle is calling the function. For a value of type `Song`
to be printed using the `PrintToStdout` function, `Song` needs to implement the
`ConvertibleToString` interface. Interface implementations will usually be
defined either with the type or with the interface. They may also be defined
somewhere else as long as Carbon can be guaranteed to see the definition when
needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

Unless the implementation of `ConvertibleToString` for `Song` is defined as
`external`, every member of `ConvertibleToString` is also a member of `Song`.
This includes members of `ConvertibleToString` that are not explicitly named in
the `impl` definition but have defaults. Whether the implementation is defined
as [internal](terminology.md#internal-impl) or
[external](terminology.md#external-impl), you may access the `ToString` function
for a `Song` value `s` by a writing function call
[using a qualified member access expression](terminology.md#qualified-member-access-expression),
like `s.(ConvertibleToString.ToString)()`.

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
  // Here the `Self` keyword means
  // "the type implementing this interface".
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: f64) -> Self;
}
```

The syntax here is to match
[how the same members would be defined in a type](/docs/design/classes.md#methods).
Each declaration in the interface defines an
[associated entity](terminology.md#associated-entity). In this example, `Vector`
has two associated methods, `Add` and `Scale`.

An interface defines a type-of-type, that is a type whose values are types. The
values of an interface are any types implementing the interface, and so provide
definitions for all the functions (and other members) declared in the interface.

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
  var x: f64;
  var y: f64;
  impl as Vector {
    // In this scope, the `Self` keyword is an
    // alias for `Point`.
    fn Add[me: Self](b: Self) -> Self {
      return {.x = a.x + b.x, .y = a.y + b.y};
    }
    fn Scale[me: Self](v: f64) -> Self {
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

**Note:** A type may implement any number of different interfaces, but may
provide at most one implementation of any single interface. This makes the act
of selecting an implementation of an interface for a type unambiguous throughout
the whole program.

**Comparison with other languages:** Rust defines implementations lexically
outside of the `class` definition. This Carbon approach means that a type's API
is described by declarations inside the `class` definition and doesn't change
afterwards.

**References:** This interface implementation syntax was accepted in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553). In
particular, see
[the alternatives considered](/proposals/p0553.md#interface-implementation-syntax).

### Implementing multiple interfaces

To implement more than one interface when defining a type, simply include an
`impl` block per interface.

```
class Point {
  var x: f64;
  var y: f64;
  impl as Vector {
    fn Add[me: Self](b: Self) -> Self { ... }
    fn Scale[me: Self](v: f64) -> Self { ... }
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
    fn Name[me: Self]() -> String { return me.name; }
    // ...
  }
  impl as GameUnit {
    // Possible syntax options for defining
    // `GameUnit.Name` as the same as `Icon.Name`:
    alias Name = Icon.Name;
    fn Name[me: Self]() -> String = Icon.Name;
    // ...
  }
}
```

### External impl

Interfaces may also be implemented for a type
[externally](terminology.md#external-impl), by using the `external impl`
construct. An external impl does not add the interface's methods to the type.

```
class Point2 {
  var x: f64;
  var y: f64;

  external impl as Vector {
    // In this scope, the `Self` keyword is an
    // alias for `Point2`.
    fn Add[me: Self](b: Self) -> Self {
      return {.x = a.x + b.x, .y = a.y + b.y};
    }
    fn Scale[me: Self](v: f64) -> Self {
      return {.x = a.x * v, .y = a.y * v};
    }
  }
}

var a: Point2 = {.x = 1.0, .y = 2.0};
// `a` does *not* have `Add` and `Scale` methods:
// ❌ Error: a.Add(a.Scale(2.0));
```

An external impl may be defined out-of-line, by including the name of the
existing type before `as`, which is otherwise optional:

```
class Point3 {
  var x: f64;
  var y: f64;
}

external impl Point3 as Vector {
  // In this scope, the `Self` keyword is an
  // alias for `Point3`.
  fn Add[me: Self](b: Self) -> Self {
    return {.x = a.x + b.x, .y = a.y + b.y};
  }
  fn Scale[me: Self](v: f64) -> Self {
    return {.x = a.x * v, .y = a.y * v};
  }
}

var a: Point3 = {.x = 1.0, .y = 2.0};
// `a` does *not* have `Add` and `Scale` methods:
// ❌ Error: a.Add(a.Scale(2.0));
```

**References:** The external interface implementation syntax was decided in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553). In
particular, see
[the alternatives considered](/proposals/p0553.md#interface-implementation-syntax).

The `external impl` statement is allowed to be defined in a different library
from `Point3`, restricted by [the coherence/orphan rules](#impl-lookup) that
ensure that the implementation of an interface can't change based on imports. In
particular, the `external impl` statement is allowed in the library defining the
interface (`Vector` in this case) in addition to the library that defines the
type (`Point3` here). This (at least partially) addresses
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

Carbon requires `impl`s defined in a different library to be `external` so that
the API of `Point3` doesn't change based on what is imported. It would be
particularly bad if two different libraries implemented interfaces with
conflicting names that both affected the API of a single type. As a consequence
of this restriction, you can find all the names of direct members (those
available by [simple member access](terminology.md#simple-member-access)) of a
type in the definition of that type. The only thing that may be in another
library is an `impl` of an interface.

You might also use `external impl` to implement an interface for a type to avoid
cluttering the API of that type, for example to avoid a name collision. A syntax
for reusing method implementations allows us to do this selectively when needed.
In this case, the `external impl` may be declared lexically inside the class
scope.

```
class Point4a {
  var x: f64;
  var y: f64;
  fn Add[me: Self](b: Self) -> Self {
    return {.x = me.x + b.x, .y = me.y + b.y};
  }
  external impl as Vector {
    alias Add = Point4a.Add;  // Syntax TBD
    fn Scale[me: Self](v: f64) -> Self {
      return {.x = me.x * v, .y = me.y * v};
    }
  }
}

// OR:

class Point4b {
  var x: f64;
  var y: f64;
  external impl as Vector {
    fn Add[me: Self](b: Self) -> Self {
      return {.x = me.x + b.x, .y = me.y + b.y};
    }
    fn Scale[me: Self](v: f64) -> Self {
      return {.x = me.x * v, .y = me.y * v};
    }
  }
  alias Add = Vector.Add;
}

// OR:

class Point4c {
  var x: f64;
  var y: f64;
  fn Add[me: Self](b: Self) -> Self {
    return {.x = me.x + b.x, .y = me.y + b.y};
  }
}

external impl Point4c as Vector {
  alias Add = Point4c.Add;  // Syntax TBD
  fn Scale[me: Self](v: f64) -> Self {
    return {.x = me.x * v, .y = me.y * v};
  }
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
declaration, the method could be called using
[simple member access](terminology.md#simple-member-access). This avoids most
concerns arising from name collisions between interfaces. It has a few downsides
though:

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

### Qualified member names and compound member access

Given a value of type `Point3` and an interface `Vector` implemented for that
type, you can access the methods from that interface using a
[qualified member access expression](terminology.md#qualified-member-access-expression)
whether or not the implementation is done externally with an `external impl`
declaration. The qualified member access expression writes the member's
_qualified name_ in the parentheses of the
[compound member access syntax](/docs/design/expressions/member_access.md):

```
var p1: Point3 = {.x = 1.0, .y = 2.0};
var p2: Point3 = {.x = 2.0, .y = 4.0};
Assert(p1.(Vector.Scale)(2.0) == p2);
Assert(p1.(Vector.Add)(p1) == p2);
```

Note that the name in the parens is looked up in the containing scope, not in
the names of members of `Point3`. So if there was another interface `Drawable`
with method `Draw` defined in the `Plot` package also implemented for `Point3`,
as in:

```
package Plot;
import Points;

interface Drawable {
  fn Draw[me: Self]();
}

external impl Points.Point3 as Drawable { ... }
```

You could access `Draw` with a qualified name:

```
import Plot;
import Points;

var p: Points.Point3 = {.x = 1.0, .y = 2.0};
p.(Plot.Drawable.Draw)();
```

**Comparison with other languages:** This is intended to be analogous to, in
C++, adding `ClassName::` in front of a member name to disambiguate, such as
[names defined in both a parent and child class](https://stackoverflow.com/questions/357307/how-to-call-a-parent-class-function-from-derived-class-function).

### Access

An `impl` must be visible to all code that can see both the type and the
interface being implemented:

-   If either the type or interface is private to a single file, then since the
    only way to define the `impl` is to use that private name, the `impl` must
    be defined private to that file as well.
-   Otherwise, if the type or interface is private but declared in an API file,
    then the `impl` must be declared in the same file so the existence of that
    `impl` is visible to all files in that library.
-   Otherwise, the `impl` must be defined in the public API file of the library,
    so it is visible in all places that might use it.

No access control modifiers are allowed on `impl` declarations, an `impl` is
always visible to the intersection of the visibility of all names used in the
declaration of the `impl`.

## Generics

Here is a function that can accept values of any type that has implemented the
`Vector` interface:

```
fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: f64) -> T {
  return a.Add(b).Scale(s);
}
var v: Point = AddAndScaleGeneric(a, w, 2.5);
```

Here `T` is a type whose type is `Vector`. The `:!` syntax means that `T` is a
_[generic parameter](terminology.md#generic-versus-template-parameters)_. That
means it must be known to the caller, but we will only use the information
present in the signature of the function to type check the body of
`AddAndScaleGeneric`'s definition. In this case, we know that any value of type
`T` implements the `Vector` interface and so has an `Add` and a `Scale` method.

**References:** The `:!` syntax was accepted in
[proposal #676](https://github.com/carbon-language/carbon-lang/pull/676).

Names are looked up in the body of `AddAndScaleGeneric` for values of type `T`
in `Vector`. This means that `AddAndScaleGeneric` is interpreted as equivalent
to adding a `Vector`
[qualification](#qualified-member-names-and-compound-member-access) to replace
all simple member accesses of `T`:

```
fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: Double) -> T {
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

With these qualifications, the function can be type-checked for any `T`
implementing `Vector`. This type checking is equivalent to type checking the
function with `T` set to an [archetype](terminology.md#archetype) of `Vector`.
An archetype is a placeholder type considered to satisfy its constraint, which
is `Vector` in this case, and no more. It acts as the most general type
satisfying the interface. The effect of this is that an archetype of `Vector`
acts like a [supertype](https://en.wikipedia.org/wiki/Subtyping) of any `T`
implementing `Vector`.

For name lookup purposes, an archetype is considered to have
[implemented its constraint internally](terminology.md#internal-impl). The only
oddity is that the archetype may have different names for members than specific
types `T` that implement interfaces from the constraint
[externally](terminology.md#external-impl). This difference in names can also
occur for supertypes in C++, for example members in a derived class can hide
members in the base class with the same name, though it is not that common for
it to come up in practice.

The behavior of calling `AddAndScaleGeneric` with a value of a specific type
like `Point` is to set `T` to `Point` after all the names have been qualified.

```
// AddAndScaleGeneric with T = Point
fn AddAndScaleForPoint(a: Point, b: Point, s: Double) -> Point {
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

This qualification gives a consistent interpretation to the body of the function
even when the type supplied by the caller
[implements the interface externally](terminology.md#external-impl), as `Point2`
does:

```
// AddAndScaleGeneric with T = Point2
fn AddAndScaleForPoint2(a: Point2, b: Point2, s: Double) -> Point2 {
  // ✅ This works even though `a.Add(b).Scale(s)` wouldn't.
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

### Return type

From the caller's perspective, the return type is the result of substituting the
caller's values for the generic parameters into the return type expression. So
`AddAndScaleGeneric` called with `Point` values returns a `Point` and called
with `Point2` values returns a `Point2`. So looking up a member on the resulting
value will look in `Point` or `Point2` rather than `Vector`.

This is part of realizing
[the goal that generic functions can be used in place of regular functions without changing the return type that callers see](goals.md#path-from-regular-functions).
In this example, `AddAndScaleGeneric` can be substituted for
`AddAndScaleForPoint` and `AddAndScaleForPoint2` without affecting the return
types. This requires the return value to be converted to the type that the
caller expects instead of the erased type used inside the generic function.

A generic caller of a generic function performs the same substitution process to
determine the return type, but the result may be generic. In this example of
calling a generic from another generic,

```
fn DoubleThreeTimes[U:! Vector](a: U) -> U {
  return AddAndScaleGeneric(a, a, 2.0).Scale(2.0);
}
```

the return type of `AddAndScaleGeneric` is found by substituting in the `U` from
`DoubleThreeTimes` for the `T` from `AddAndScaleGeneric` in the return type
expression of `AddAndScaleGeneric`. `U` is an archetype of `Vector`, and so
implements `Vector` internally and therefore has a `Scale` method.

If `U` had a more specific type, the return value would have the additional
capabilities of `U`. For example, given a parameterized type `GeneralPoint`
implementing `Vector`, and a function that takes a `GeneralPoint` and calls
`AddAndScaleGeneric` with it:

```
class GeneralPoint(C:! Numeric) {
  external impl as Vector { ... }
  fn Get[me: Self](i: i32) -> C;
}

fn CallWithGeneralPoint[C:! Numeric](p: GeneralPoint(C)) -> C {
  // `AddAndScaleGeneric` returns `T` and in these calls `T` is
  // deduced to be `GeneralPoint(C)`.

  // ❌ Illegal: AddAndScaleGeneric(p, p, 2.0).Scale(2.0);
  //    `GeneralPoint(C)` implements `Vector` externally, and so
  //    does not have a `Scale` method.

  // ✅ Allowed: `GeneralPoint(C)` has a `Get` method
  AddAndScaleGeneric(p, p, 2.0).Get(0);

  // ✅ Allowed: `GeneralPoint(C)` implements `Vector`
  //    externally, and so has a `Vector.Scale` method.
  //    `Vector.Scale` returns `Self` which is `GeneralPoint(C)`
  //    again, and so has a `Get` method.
  return AddAndScaleGeneric(p, p, 2.0).(Vector.Scale)(2.0).Get(0);
}
```

The result of the call to `AddAndScaleGeneric` from `CallWithGeneralPoint` has
type `GeneralPoint(C)` and so has a `Get` method and a `Vector.Scale` method.
But, in contrast to how `DoubleThreeTimes` works, since `Vector` is implemented
externally the return value in this case does not directly have a `Scale`
method.

### Implementation model

A possible model for generating code for a generic function is to use a
[witness table](terminology.md#witness-tables) to represent how a type
implements an interface:

-   [Interfaces](#interfaces) are types of witness tables.
-   [Impls](#implementing-interfaces) are witness table values.

Type checking is done with just the interface. The impl is used during code
generation time, possibly using
[monomorphization](https://en.wikipedia.org/wiki/Monomorphization) to have a
separate instantiation of the function for each combination of the generic
argument values. The compiler is free to use other implementation strategies,
such as passing the witness table for any needed implementations, if that can be
predicted.

For the example above, [the Vector interface](#interfaces) could be thought of
defining a witness table type like:

```
class Vector {
  // `Self` is the representation type, which is only
  // known at compile time.
  var Self:! Type;
  // `fnty` is **placeholder** syntax for a "function type",
  // so `Add` is a function that takes two `Self` parameters
  // and returns a value of type `Self`.
  var Add: fnty(a: Self, b: Self) -> Self;
  var Scale: fnty(a: Self, v: f64) -> Self;
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
    .Scale = lambda(a: Point, v: f64) -> Point {
      return {.x = a.x * v, .y = a.y * v};
    },
};
```

Since generic arguments (where the parameter is declared using `:!`) are passed
at compile time, so the actual value of `VectorForPoint` can be used to generate
the code for functions using that impl. This is the
[static-dispatch witness table](terminology.md#static-dispatch-witness-table)
approach.

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
-   as a namespace name in
    [a qualified name](#qualified-member-names-and-compound-member-access), and
-   as a [type-of-type](terminology.md#type-of-type) for
    [a generic type parameter](#generics).

While interfaces are examples of type-of-types, type-of-types are a more general
concept, for which interfaces are a building block.

## Type-of-types

A [type-of-type](terminology.md#type-of-type) consists of a set of requirements
and a set of names. Requirements are typically a set of interfaces that a type
must satisfy, though other kinds of requirements are added below. The names are
aliases for qualified names in those interfaces.

An interface is one particularly simple example of a type-of-type. For example,
`Vector` as a type-of-type has a set of requirements consisting of the single
interface `Vector`. Its set of names consists of `Add` and `Scale` which are
aliases for the corresponding qualified names inside `Vector` as a namespace.

The requirements determine which types are values of a given type-of-type. The
set of names in a type-of-type determines the API of a generic type value and
define the result of [member access](/docs/design/expressions/member_access.md)
into the type-of-type.

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
  // Can accept values of any type. But, since we know nothing about the
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
inside a `template constraint`. Note that a generic constraint ignores the names
of members defined for a type, but a template constraint can depend on them.

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

There is a subtyping relationship between type-of-types that allows calls of one
generic function from another as long as it has a subset of the requirements.

Given a generic type variable `T` with type-of-type `I1`, it satisfies a
type-of-type `I2` as long as the requirements of `I1` are a superset of the
requirements of `I2`. This means a value `x` of type `T` may be passed to
functions requiring types to satisfy `I2`, as in this example:

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

Conflicts can be resolved at the call site using a
[qualified member access expression](#qualified-member-names-and-compound-member-access),
or by defining a named constraint explicitly and renaming the methods:

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
[interface extensions](#interface-extension) of a common base interface, the
combination should not conflict on any names in the common base.

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
[Carbon: Access to interface methods](https://docs.google.com/document/d/17IXDdu384x1t9RimQ01bhx4-nWzs4ZEeke4eO6ImQNc/edit?resourcekey=0-Fe44R-0DhQBlw0gs2ujNJA).

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
[type-of-types in general](#type-of-types). For consistency we will use the same
semantics and syntax as we do for [named constraints](#named-constraints):

```
interface Equatable { fn Equals[me: Self](rhs: Self) -> bool; }

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
  impl as Equatable { fn Equals[me: Self](rhs: Self) -> bool { ... } }
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

**Note:** The design for this feature is continued in
[a later section](#interface-requiring-other-interfaces-revisited).

### Interface extension

When implementing an interface, we should allow implementing the aliased names
as well. In the case of `Hashable` above, this includes all the members of
`Equatable`, obviating the need to implement `Equatable` itself:

```
class Song {
  impl as Hashable {
    fn Hash[me: Self]() -> u64 { ... }
    fn Equals[me: Self](rhs: Self) -> bool { ... }
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
interface Equatable { fn Equals[me: Self](rhs: Self) -> bool; }

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
    [Carbon generics use case: graph library](https://docs.google.com/document/d/15Brjv8NO_96jseSesqer5HbghqSTJICJ_fTaZOH0Mg4/edit?usp=sharing&resourcekey=0-CYSbd6-xF8vYHv9m1rolEQ)
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
The [`SetAlgebra` protocol](https://swiftdoc.org/v5.1/protocol/setalgebra/)
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

This is just like when you get an implementation of `Equatable` by implementing
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
[this example generic graph library doc](https://docs.google.com/document/d/15Brjv8NO_96jseSesqer5HbghqSTJICJ_fTaZOH0Mg4/edit?usp=sharing&resourcekey=0-CYSbd6-xF8vYHv9m1rolEQ):

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
    external impl MyEdgeListIncidenceGraph as Graph {
      fn Source[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
      fn Target[me: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    }
    ```

This last point means that there are situations where we can only detect a
missing method definition by the end of the file. This doesn't delay other
aspects of semantic checking, which will just assume that these methods will
eventually be provided.

**Open question:** We could require that the `external impl` of the required
interface be declared lexically in the class scope in this case. That would
allow earlier detection of missing definitions.

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
  fn Difference[me: Self](rhs: Self) -> i32;
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

## Adapting types

Since interfaces may only be implemented for a type once, and we limit where
implementations may be added to a type, there is a need to allow the user to
switch the type of a value to access different interface implementations. Carbon
therefore provides a way to create new types
[compatible with](terminology.md#compatible-types) existing types with different
APIs, in particular with different interface implementations, by
[adapting](terminology.md#adapting-a-type) them:

```
interface Printable {
  fn Print[me: Self]();
}
interface Comparable {
  fn Less[me: Self](rhs: Self) -> bool;
}
class Song {
  impl as Printable { fn Print[me: Self]() { ... } }
}
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](rhs: Self) -> bool { ... }
  }
}
adapter FormattedSong for Song {
  impl as Printable { fn Print[me: Self]() { ... } }
}
adapter FormattedSongByTitle for Song {
  impl as Printable = FormattedSong;
  impl as Comparable = SongByTitle;
}
```

This allows developers to provide implementations of new interfaces (as in
`SongByTitle`), provide different implementations of the same interface (as in
`FormattedSong`), or mix and match implementations from other compatible types
(as in `FormattedSongByTitle`). The rules are:

-   You can add any declaration that you could add to a class except for
    declarations that would change the representation of the type. This means
    you can add methods, functions, interface implementations, and aliases, but
    not fields, base classes, or virtual functions.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `Song`, `SongByTitle`, `FormattedSong`,
    and `FormattedSongByTitle` end up compatible with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit conversion between
    these types.

Inside an adapter, the `Self` type matches the adapter. Members of the original
type may be accessed either by a cast:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](rhs: Self) -> bool {
      return (me as Song).Title() < (rhs as Song).Title();
    }
  }
}
```

or using a qualified member access expression:

```
adapter SongByTitle for Song {
  impl as Comparable {
    fn Less[me: Self](rhs: Self) -> bool {
      return me.(Song.Title)() < rhs.(Song.Title)();
    }
  }
}
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

Consider a type with a generic type parameter, like a hash map:

```
interface Hashable { ... }
class HashMap(KeyT:! Hashable, ValueT:! Type) {
  fn Find[me:Self](key: KeyT) -> Optional(ValueT);
  // ...
}
```

A user of this type will provide specific values for the key and value types:

```
class Song {
  impl as Hashable { ... }
  // ...
}

var play_count: HashMap(Song, i32) = ...;
var thriller_count: Optional(i32) =
    play_count.Find(Song("Thriller"));
```

Since the `Find` function is generic, it can only use the capabilities that
`HashMap` requires of `KeyT` and `ValueT`. This allows us to evaluate when we
can convert between two different arguments to a parameterized type. Consider
two adapters of `Song` that implement `Hashable`:

```
adapter PlayableSong for Song {
  impl as Hashable = Song;
  impl as Media { ... }
}
adapter SongHashedByTitle for Song {
  impl as Hashable { ... }
}
```

`Song` and `PlayableSong` have the same implementation of `Hashable` in addition
to using the same data representation. This means that it is safe to convert
between `HashMap(Song, i32)` and `HashMap(PlayableSong, i32)`, because the
implementation of all the methods will use the same implementation of the
`Hashable` interface. Carbon permits this conversion with an explicit cast.

On the other hand, `SongHashedByTitle` has a different implementation of
`Hashable` than `Song`. So even though `Song` and `SongHashedByTitle` are
compatible types, `HashMap(Song, i32)` and `HashMap(SongHashedByTitle, i32)` are
incompatible. This is important because we know that in practice the invariants
of a `HashMap` implementation rely on the hashing function staying the same.

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

Unlike the similar `class B extends A` notation, `adapter B extends A` is
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
  external impl as Printable = Song;

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
// Or, equivalently:
adapter Song extends SongLib.Song {
  external impl as CompareLib.Comparable { ... }
}
```

The caller can either convert `SongLib.Song` values to `Song` when calling
`CompareLib.Sort` or just start with `Song` values in the first place.

```
var lib_song: SongLib.Song = ...;
CompareLib.Sort((lib_song as Song,));

var song: Song = ...;
CompareLib.Sort((song,));
```

### Use case: Defining an impl for use by other types

Let's say we want to provide a possible implementation of an interface for use
by types for which that implementation would be appropriate. We can do that by
defining an adapter implementing the interface that is parameterized on the type
it is adapting. That impl may then be pulled in using the `impl as ... = ...;`
syntax.

For example, given an interface `Comparable` for deciding which value is
smaller:

```
interface Comparable {
  fn Less[me: Self](rhs: Self) -> bool;
}
```

We might define an adapter that implements `Comparable` for types that define
another interface `Difference`:

```
interface Difference {
  fn Sub[me:Self](rhs: Self) -> i32;
}
adapter ComparableFromDifference(T:! Difference) for T {
  impl as Comparable {
    fn Less[me: Self](rhs: Self) -> bool {
      return (me as T).Sub(rhs) < 0;
    }
  }
}
class IntWrapper {
  var x: i32;
  impl as Difference {
    fn Sub[me: Self](rhs: Self) -> i32 {
      return left.x - right.x;
    }
  }
  impl as Comparable = ComparableFromDifferenceFn(IntWrapper);
}
```

**TODO:** If we support function types, we could potentially pass a function to
use to the adapter instead:

```
adapter ComparableFromDifferenceFn
    (T:! Type, Difference:! fnty(T, T)->i32) for T {
  impl as Comparable {
    fn Less[me: Self](rhs: Self) -> bool {
      return Difference(me, rhs) < 0;
    }
  }
}
class IntWrapper {
  var x: i32;
  fn Difference(left: Self, right: Self) {
    return left.x - right.x;
  }
  impl as Comparable =
      ComparableFromDifferenceFn(IntWrapper, Difference);
}
```

### Use case: Private impl

Adapter types can be used when a library publicly exposes a type, but only wants
to say that type implements an interface as a private detail internal to the
implementation of the type. In that case, instead of implementing the interface
for the public type, the library can create a private adapter for that type and
implement the interface on that instead. Any member of the class can cast its
`me` parameter to the adapter type when it wants to make use of the private
impl.

```
// Public, in API file
class Complex64 {
  // ...
  fn CloserToOrigin[me: Self](them: Self) -> bool;
}

// Private

adapter ByReal extends Complex64 {
  // Complex numbers are not generally comparable,
  // but this comparison function is useful for some
  // method implementations.
  impl as Comparable {
    fn Less[me: Self](that: Self) -> bool {
      return me.Real() < that.Real();
    }
  }
}

fn Complex64.CloserToOrigin[me: Self](them: Self) -> bool {
  var me_mag: ByReal = me * me.Conj() as ByReal;
  var them_mag: ByReal = them * them.Conj() as ByReal;
  return me_mag.Less(them_mag);
}
```

### Use case: Accessing external names

Consider a case where a function will call several functions from an interface
that is [implemented externally](terminology.md#external-impl) for a type.

```
interface DrawingContext {
  fn SetPen[me: Self](...);
  fn SetFill[me: Self](...);
  fn DrawRectangle[me: Self](...);
  fn DrawLine[me: Self](...);
  ...
}
external impl Window as DrawingContext { ... }
```

An adapter can make that much more convenient by making a compatible type where
the interface is [implemented internally](terminology.md#internal-impl). This
avoids having to [qualify](terminology.md#qualified-member-access-expression)
each call to methods in the interface.

```
adapter DrawInWindow for Window {
  impl as DrawingContext = Window;
}
fn Render(w: Window) {
  let d: DrawInWindow = w as DrawInWindow;
  d.SetPen(...);
  d.SetFill(...);
  d.DrawRectangle(...);
  ...
}
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

An implementation of an interface specifies values for associated constants with
a [`where` clause](#where-constraints). For example, implementations of
`NSpacePoint` for different types might have different values for `N`:

```
class Point2D {
  impl as NSpacePoint where .N = 2 {
    fn Get[addr me: Self*](i: i32) -> f64 { ... }
    fn Set[addr me: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr me: Self*](value: Array(f64, 2)) { ... }
  }
}

class Point3D {
  impl as NSpacePoint where .N = 3 {
    fn Get[addr me: Self*](i: i32) -> f64 { ... }
    fn Set[addr me: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr me: Self*](value: Array(f64, 3)) { ... }
  }
}
```

Multiple assignments to associated constants may be joined using the `and`
keyword. The list of assignments is subject to two restrictions:

-   An implementation of an interface cannot specify a value for a
    [`final`](#final-members) associated constant.
-   If an associated constant doesn't have a
    [default value](#interface-defaults), every implementation must specify its
    value.

These values may be accessed as members of the type:

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
associated class functions are written using a `fn` declaration:

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
[in the "Interfaces" section](#interfaces). For other cases, we can say that the
interface declares that each implementation will provide a type under a specific
name. For example:

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

  // Set the associated type `ElementType` to `T`.
  impl as StackAssociatedType where .ElementType = T {
    fn Push[addr me: Self*](value: ElementType) {
      me->Insert(me->End(), value);
    }
    fn Pop[addr me: Self*]() -> ElementType {
      var pos: IteratorType = me->End();
      Assert(pos != me->Begin());
      --pos;
      returned var ret: ElementType = *pos;
      me->Remove(pos);
      return var;
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return me->Begin() == me->End();
    }
  }
}
```

The keyword `Self` can be used after the `as` in an `impl` declaration as a
shorthand for the type being implemented, including in the `where` clause
specifying the values of associated types, as in:

```
external impl VeryLongTypeName as Add
    // `Self` here means `VeryLongTypeName`
    where .Result == Self {
  ...
}
```

**Alternatives considered:** See
[other syntax options considered in #731 for specifying associated types](/proposals/p0731.md#syntax-for-associated-constants).
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
```

Inside the generic function `PeekAtTopOfStack`, the `ElementType` associated
type member of `StackType` is erased. This means `StackType.ElementType` has the
API dictated by the declaration of `ElementType` in the interface
`StackAssociatedType`.

Outside the generic, associated types have the concrete type values determined
by impl lookup, rather than the erased version of that type used inside a
generic.

```
var my_array: DynamicArray(i32) = (1, 2, 3);
// PeekAtTopOfStack's `StackType` is set to `DynamicArray(i32)`
// with `StackType.ElementType` set to `i32`.
Assert(PeekAtTopOfStack(my_array) == 3);
```

This is another part of achieving
[the goal that generic functions can be used in place of regular functions without changing the return type that callers see](goals.md#path-from-regular-functions)
discussed in the [return type section](#return-type).

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
    class IteratorType {
      impl Iterator { ... }
    }
    ...
  }
}
```

For context, see
["Interface type parameters and associated types" in the generics terminology document](terminology.md#interface-type-parameters-and-associated-types).

**Comparison with other languages:** Both
[Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189)
support associated types.

### Implementation model

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
would use
[parameterized interfaces](terminology.md#interface-type-parameters-and-associated-types).
To write a parameterized version of the stack interface, instead of using
associated types, write a parameter list after the name of the interface instead
of the associated type declaration:

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
      me->fruit.Push(value);
    }
    fn Pop[addr me: Self*]() -> Fruit {
      return me->fruit.Pop();
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return me->fruit.IsEmpty();
    }
  }
  impl as StackParameterized(Veggie) {
    fn Push[addr me: Self*](value: Veggie) {
      me->veggie.Push(value);
    }
    fn Pop[addr me: Self*]() -> Veggie {
      return me->veggie.Pop();
    }
    fn IsEmpty[addr me: Self*]() -> bool {
      return me->veggie.IsEmpty();
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
    (s: StackType*, _:! singleton_type_of(T)) -> T { ... }

var produce: Produce = ...;
var top_fruit: Fruit =
    PeekAtTopOfStackParameterized(&produce, Fruit);
var top_veggie: Veggie =
    PeekAtTopOfStackParameterized(&produce, Veggie);
```

The pattern `_:! singleton_type_of(T)` is a placeholder syntax for an expression
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
  fn Equals[me: Self](rhs: T) -> bool;
  ...
}
class Complex {
  var real: f64;
  var imag: f64;
  // Can implement this interface more than once
  // as long as it has different arguments.
  impl as EquatableWith(f64) { ... }
  // Same as: impl as EquatableWith(Complex) { ... }
  impl as EquatableWith(Self) { ... }
}
```

All interface parameters must be marked as "generic", using the `:!` syntax.
This reflects these two properties of these parameters:

-   They must be resolved at compile-time, and so can't be passed regular
    dynamic values.
-   We allow either generic or template values to be passed in.

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

**Caveat:** When implementing an interface twice for a type, the interface
parameters are required to always be different. For example:

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

[Rust uses the term "type parameters"](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#clearer-trait-matching)
for both interface type parameters and associated types. The difference is that
interface parameters are "inputs" since they _determine_ which `impl` to use,
and associated types are "outputs" since they are determined _by_ the `impl`,
but play no role in selecting the `impl`.

### Impl lookup

Let's say you have some interface `I(T, U(V))` being implemented for some type
`A(B(C(D), E))`. To satisfy the [orphan rule for coherence](#orphan-rule), that
`impl` must be defined in some library that must be imported in any code that
looks up whether that interface is implemented for that type. This requires that
`impl` is defined in the same library that defines the interface or one of the
names needed by the type. That is, the `impl` must be defined with one of `I`,
`T`, `U`, `V`, `A`, `B`, `C`, `D`, or `E`. We further require anything looking
up this `impl` to import the _definitions_ of all of those names. Seeing a
forward declaration of these names is insufficient, since you can presumably see
forward declarations without seeing an `impl` with the definition. This
accomplishes a few goals:

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
do allow there to be more than one `impl` to be defined for a type, by
unambiguously picking one as most specific.

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
member of one is equal to the member of another. The `where` operator is not
associative, so a type expression using multiple must use round parens `(`...`)`
to specify grouping.

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
interface Has2DPoint {
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

This syntax is also used to specify the values of
[associated constants](#associated-constants) when implementing an interface for
a type.

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

To name these sorts of constraints, we could use `let` declarations or
`constraint` definitions:

```
let IntStack:! auto = Stack where .ElementType = i32;
constraint IntStack {
  extends Stack where .ElementType = i32;
}
```

This syntax is also used to specify the values of
[associated types](#associated-types) when implementing an interface for a type.

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

we can constrain them to be equal in a function signature:

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
from the `needles` container won't directly have a `Compare` method member, but
can still be implicitly converted to `Comparable` and can still call `Compare`
using the compound member access syntax, `needle.(Comparable.Compare)(elt)`. The
rule is that an `==` `where` constraint between two type variables does not
modify the set of member names of either type. (If you write
`where .ElementType = String` with a `=` and a concrete type, then
`.ElementType` is actually set to `String` including the complete `String` API.)

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
`Self` instead of `.Self`, since they refer to the same type.

The `.Self` construct follows these rules:

-   `X :!` introduces `.Self:! Type`, where references to `.Self` are resolved
    to `X`. This allows you to use `.Self` as an interface parameter as in
    `X:! I(.Self)`.
-   `A where` introduces `.Self:! A` and `.Foo` for each member `Foo` of `A`
-   It's an error to reference `.Self` if it refers to more than one different
    thing or isn't a type.
-   You get the innermost, most-specific type for `.Self` if it is introduced
    twice in a scope. By the previous rule, it is only legal if they all refer
    to the same generic parameter.

So in `X:! A where ...`, `.Self` is introduced twice, after the `:!` and the
`where`. This is allowed since both times it means `X`. After the `:!`, `.Self`
has the type `Type`, which gets refined to `A` after the `where`. In contrast,
it is an error if `.Self` could mean two different things, as in:

```
// ❌ Illegal: `.Self` could mean `T` or `T.A`.
fn F[T:! InterfaceA where .A is
           (InterfaceB where .B == .Self)](x: T);
```

#### Parameterized type implements interface

There are times when a function will pass a generic type parameter of the
function as an argument to a parameterized type, as in the previous case, and in
addition the function needs the result to implement a specific interface.

```
// Some parameterized type.
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

The `KeyType` in these declarations does not visibly satisfy the requirements of
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
parameter, but not its _member names_. This way you can always look at the
declaration to see how name resolution works, without having to look up the
definitions of everything it is used as an argument to.

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
support some form of this feature as part of their type inference (and
[the Rust community is considering expanding support](http://smallcultfollowing.com/babysteps//blog/2022/04/12/implied-bounds-and-perfect-derive/#expanded-implied-bounds)).

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

### Referencing names in the interface being defined

The constraint in a `where` clause is required to only reference earlier names
from this scope, as in this example:

```
interface Graph {
  let E: Edge;
  let V: Vert where .E == E and .Self == E.V;
}
```

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
  // ✅ Allowed
  t.TakesC(t.GetA() as T.B);

  // ✅ Allowed
  let b: T.B = t.GetA();
  t.TakesC(b);

  // ❌ Not allowed: t.TakesC(t.GetA());
}
```

A value of type `A`, such as the return value of `GetA()`, has the API of `P`.
Any such value also implements `Q`, and since the compiler can see that by way
of a single `where` equality, values of type `A` are treated as if they
implement `Q` [externally](terminology.md#external-impl). However, the compiler
will require a cast to `B` or `C` to see that the type implements `R`.

```
fn TakesPQR[U:! P & Q & R](u: U);

fn G[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // ✅ Allowed: `T.A` implements `P`.
  a.InP();

  // ✅ Allowed: `T.A` implements `Q` externally.
  a.(Q.InQ)();

  // ❌ Not allowed: a.InQ();

  // ✅ Allowed: values of type `T.A` may be cast
  // to `T.B`, which implements `Q` internally.
  (a as T.B).InQ();

  // ✅ Allowed: `T.B` implements `R` externally.
  (a as T.B).(R.InR)();

  // ❌ Not allowed: TakesPQR(a);

  // ✅ Allowed: `T.B` implements `P`, `Q`, and
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
  // ✅ Legal: argument has type `C.X.X.Y`
  c.TakesXXY(c.GetX().GetX().GetY());

  // ✅ Legal: argument has type `C.X.Y.X` which is equal
  // to `C.X.X.Y` following only one `where` clause.
  c.TakesXXY(c.GetX().GetY().GetX());

  // ✅ Legal: cast is legal since it matches a `where`
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
  // ✅ Legal:
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

  // Without this `observe` declaration, the
  // calls in `F` below would not be allowed.
  observe A == B == C;
}

fn TakesPQR[U:! P & Q & R](u: U);

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // ✅ Allowed: `T.A` == `T.C`
  t.TakesC(a);
  a.(R.InR());

  // ✅ Allowed: `T.A` implements `P`,
  // `T.A` == `T.B` that implements `Q`, and
  // `T.A` == `T.C` that implements `R`.
  TakesPQR(a);
}
```

Since adding an `observe` declaration only adds external implementations of
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
  fn Compare[me: Self](rhs: Self) -> CompareResult;
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
adapter ThenCompare(
      T:! Type,
      CompareList:! List(CompatibleWith(T) & Comparable))
    for T {
  impl as Comparable {
    fn Compare[me: Self](rhs: Self) -> CompareResult {
      for (let U:! auto in CompareList) {
        var result: CompareResult = (me as U).Compare(rhs as U);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

let SongByArtistThenTitle: auto =
    ThenCompare(Song, (SongByArtist, SongByTitle));
var s1: Song = ...;
var s2: SongByArtistThenTitle =
    Song(...) as SongByArtistThenTitle;
assert((s1 as SongByArtistThenTitle).Compare(s2) ==
       CompareResult.Less);
```

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

A type is called _sized_ if it is in the first two categories, and _unsized_
otherwise. Note: something with size 0 is still considered "sized". The
type-of-type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring variables of that type,
taking values of that type as parameters, returning values of that type, and
defining arrays of that type. Users will not typically need to express the
`Sized` constraint explicitly, though, since it will usually be a dependency of
some other constraint the type will need such as `Movable` or `Concrete`.

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

var z: Name = Name.Default();;
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

#### Implementation model

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

**Open question:** Should `TypeId` be
[implemented externally](terminology.md#external-impl) for types to avoid name
pollution (`.TypeName`, `.TypeHash`, etc.) unless the function specifically
requests those capabilities?

### Destructor constraints

There are four type-of-types related to
[the destructors of types](/docs/design/classes.md#destructors):

-   `Concrete` types may be local or member variables.
-   `Deletable` types may be safely deallocated by pointer using the `Delete`
    method on the `Allocator` used to allocate it.
-   `Destructible` types have a destructor and may be deallocated by pointer
    using the `UnsafeDelete` method on the correct `Allocator`, but it may be
    unsafe. The concerning case is deleting a pointer to a derived class through
    a pointer to its base class without a virtual destructor.
-   `TrivialDestructor` types have empty destructors. This type-of-type may be
    used with [specialization](#lookup-resolution-and-specialization) to unlock
    specific optimizations.

**Note:** The names `Deletable` and `Destructible` are
[**placeholders**](/proposals/p1154.md#type-of-type-naming) since they do not
conform to the decision on
[question-for-leads issue #1058: "How should interfaces for core functionality be named?"](https://github.com/carbon-language/carbon-lang/issues/1058).

The type-of-types `Concrete`, `Deletable`, and `TrivialDestructor` all extend
`Destructible`. Combinations of them may be formed using
[the `&` operator](#combining-interfaces-by-anding-type-of-types). For example,
a generic function that both instantiates and deletes values of a type `T` would
require `T` implement `Concrete & Deletable`.

Types are forbidden from explicitly implementing these type-of-types directly.
Instead they use
[`destructor` declarations in their class definition](/docs/design/classes.md#destructors)
and the compiler uses them to determine which of these type-of-types are
implemented.

## Generic `let`

A `let` statement inside a function body may be used to get the change in type
behavior of calling a generic function without having to introduce a function
call.

```
fn F(...) {
  ...
  let T:! C = U;
  X;
  Y;
  Z;
}
```

gets rewritten to:

```
fn F(...) {
  ...
  fn Closure(T:! C where .Self == U) {
    X;
    Y;
    Z;
  }
  Closure(U);
}
```

The `where .Self == U` modifier allows values to implicitly convert between type
`T`, the erased type, and type `U`, the concrete type. Note that implicit
conversion is
[only performed across a single `where` equality](#manual-type-equality). This
can be used to switch to the API of `C` when it is external, as an alternative
to [using an adapter](#use-case-accessing-external-names), or to simplify
inlining of a generic function while preserving semantics.

## Parameterized impls

There are cases where an impl definition should apply to more than a single type
and interface combination. The solution is to parameterize the impl definition,
so it applies to a family of types, interfaces, or both. This includes:

-   Declare an impl for a parameterized type, which may be external or declared
    out-of-line.
-   "Conditional conformance" where a parameterized type implements some
    interface if the parameter to the type satisfies some criteria, like
    implementing the same interface.
-   "Blanket" impls where an interface is implemented for all types that
    implement another interface, or some other criteria beyond being a specific
    type.
-   "Wildcard" impls where a family of interfaces are implemented for single
    type.

### Impl for a parameterized type

Interfaces may be implemented for a parameterized type. This can be done
lexically in the class' scope:

```
class Vector(T:! Type) {
  impl as Iterable where .ElementType = T {
    ...
  }
}
```

This is equivalent to naming the type between `impl` and `as`:

```
class Vector(T:! Type) {
  impl Vector(T) as Iterable where .ElementType = T {
    ...
  }
}
```

An impl may be declared [external](#external-impl) by adding an `external`
keyword before `impl`. External impls may also be declared out-of-line, but all
parameters must be declared in a `forall` clause:

```
external impl forall [T:! Type] Vector(T) as Iterable
    where .ElementType = T {
  ...
}
```

The parameter for the type can be used as an argument to the interface being
implemented:

```
class HashMap(Key:! Hashable, Value:! Type) {
  impl as Has(Key) { ... }
  impl as Contains(HashSet(Key)) { ... }
}
```

or externally out-of-line:

```
class HashMap(Key:! Hashable, Value:! Type) { ... }
external impl forall [Key:! Hashable, Value:! Type]
    HashMap(Key, Value) as Has(Key) { ... }
external impl forall [Key:! Hashable, Value:! Type]
    HashMap(Key, Value) as Contains(HashSet(Key)) { ... }
```

### Conditional conformance

[Conditional conformance](terminology.md#conditional-conformance) is expressing
that we have an `impl` of some interface for some type, but only if some
additional type restrictions are met. Examples where this would be useful
include being able to say that a container type, like `Vector`, implements some
interface when its element type satisfies the same interface:

-   A container is printable if its elements are.
-   A container could be compared to another container with the same element
    type using a
    [lexicographic comparison](https://en.wikipedia.org/wiki/Lexicographic_order)
    if the element type is comparable.
-   A container is copyable if its elements are.

To do this with an [`external impl`](#external-impl), specify a more-specific
`Self` type to the left of the `as` in the declaration:

```
interface Printable {
  fn Print[me: Self]();
}
class Vector(T:! Type) { ... }

// By saying "T:! Printable" instead of "T:! Type" here,
// we constrain T to be Printable for this impl.
external impl forall [T:! Printable] Vector(T) as Printable {
  fn Print[me: Self]() {
    for (let a: T in me) {
      // Can call `Print` on `a` since the constraint
      // on `T` ensures it implements `Printable`.
      a.Print();
    }
  }
}
```

To define these `impl`s inline in a `class` definition, include a `forall`
clause with a more-specific type between the `impl` and `as` keywords.

```
class Array(T:! Type, template N:! Int) {
  impl forall [P:! Printable] Array(P, N) as Printable { ... }
}
```

It is legal to add the keyword `external` before the `impl` keyword to switch to
an external impl defined lexically within the class scope. Inside the scope,
both `P` and `T` refer to the same type, but `P` has the type-of-type of
`Printable` and so has a `Print` member. The relationship between `T` and `P` is
as if there was a `where P == T` clause.

**TODO:** Need to resolve whether the `T` name can be reused, or if we require
that you need to use new names, like `P`, when creating new type variables.

**Example:** Consider a type with two parameters, like `Pair(T, U)`. In this
example, the interface `Foo(T)` is only implemented when the two types are
equal.

```
interface Foo(T:! Type) { ... }
class Pair(T:! Type, U:! Type) { ... }
external impl forall [T:! Type] Pair(T, T) as Foo(T) { ... }
```

You may also define the `impl` inline, in which case it can be internal:

```
class Pair(T:! Type, U:! Type) {
  impl Pair(T, T) as Foo(T) { ... }
}
```

**Clarification:** Method lookup will look at all internal implementations,
whether or not the conditions on those implementations hold for the `Self` type.
If the conditions don't hold, then the call will be rejected because `Self` has
the wrong type, just like any other argument/parameter type mismatch. This means
types may not implement two different interfaces internally if they share a
member name, even if their conditions are mutually exclusive:

```
class X(T:! Type) {
  impl X(i32) as Foo {
    fn F[me: Self]();
  }
  impl X(i64) as Bar {
    // ❌ Illegal: name conflict between `Foo.F` and `Bar.F`
    fn F[me: Self](n: i64);
  }
}
```

However, the same interface may be implemented multiple times as long as there
is no overlap in the conditions:

```
class X(T:! Type) {
  impl X(i32) as Foo {
    fn F[me: Self]();
  }
  impl X(i64) as Foo {
    // ✅ Allowed: `X(T).F` consistently means `X(T).(Foo.F)`
    fn F[me: Self]();
  }
}
```

This allows a type to express that it implements an interface for a list of
types, possibly with different implementations.

In general, `X(T).F` can only mean one thing, regardless of `T`.

**Comparison with other languages:**
[Swift supports conditional conformance](https://github.com/apple/swift-evolution/blob/master/proposals/0143-conditional-conformances.md),
but bans cases where there could be ambiguity from overlap.
[Rust also supports conditional conformance](https://doc.rust-lang.org/rust-by-example/generics/where.html).

#### Conditional methods

A method could be defined conditionally for a type by using a more specific type
in place of `Self` in the method declaration. For example, this is how to define
a vector type that only has a `Sort` method if its elements implement the
`Comparable` interface:

```
class Vector(T:! Type) {
  // `Vector(T)` has a `Sort()` method if `T` is `Comparable`.
  fn Sort[C:! Comparable, addr me: Vector(C)*]();
}
```

**Comparison with other languages:** In
[Rust](https://doc.rust-lang.org/book/ch10-02-traits.html#using-trait-bounds-to-conditionally-implement-methods)
this feature is part of conditional conformance. Swift supports conditional
methods using
[conditional extensions](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID553)
or
[contextual where clauses](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID628).

### Blanket impls

A _blanket impl_ is an `impl` that could apply to more than one root type, so
the `impl` will use a type variable for the `Self` type. Here are some examples
where blanket impls arise:

-   Any type implementing `Ordered` should get an implementation of
    `PartiallyOrdered`.

    ```
    external impl forall [T:! Ordered] T as PartiallyOrdered { ... }
    ```

-   `T` implements `CommonType(T)` for all `T`

    ```
    external impl forall [T:! Type] T as CommonType(T)
        where .Result = T { }
    ```

    This means that every type is the common type with itself.

Blanket impls must always be [external](#external-impl) and defined lexically
out-of-line.

#### Difference between blanket impls and named constraints

A blanket interface can be used to say "any type implementing `interface I` also
implements `interface B`." Compare this with defining a `constraint C` that
requires `I`. In that case, `C` will also be implemented any time `I` is. There
are differences though:

-   There can be other implementations of `interface B` without a corresponding
    implementation of `I`, unless `B` has a requirement on `I`. However, the
    types implementing `C` will be the same as the types implementing `I`.
-   More specialized implementations of `B` can override the blanket
    implementation.

### Wildcard impls

A _wildcard impl_ is an impl that defines a family of interfaces for a single
`Self` type. For example, the `BigInt` type might implement `AddTo(T)` for all
`T` that implement `ImplicitAs(i32)`. The implementation would first convert `T`
to `i32` and then add the `i32` to the `BigInt` value.

```
class BigInt {
  external impl forall [T:! ImplicitAs(i32)] as AddTo(T) { ... }
}
// Or out-of-line:
external impl forall [T:! ImplicitAs(i32)] BigInt as AddTo(T) { ... }
```

Wildcard impls must always be [external](#external-impl), to avoid having the
names in the interface defined for the type multiple times.

### Combinations

The different kinds of parameters to impls may be combined. For example, if `T`
implements `As(U)`, then this implements `As(Optional(U))` for `Optional(T)`:

```
external impl forall [U:! Type, T:! As(U)]
  Optional(T) as As(Optional(U)) { ... }
```

This has a wildcard parameter `U`, and a condition on parameter `T`.

### Lookup resolution and specialization

As much as possible, we want rules for where an impl is allowed to be defined
and for selecting which impl to use that achieve these three goals:

-   Implementations have coherence, as
    [defined in terminology](terminology.md#coherence). This is
    [a goal for Carbon](goals.md#coherence). More detail can be found in
    [this appendix with the rationale and alternatives considered](appendix-coherence.md).
-   Libraries will work together as long as they pass their separate checks.
-   A generic function can assume that some impl will be successfully selected
    if it can see an impl that applies, even though another more specific impl
    may be selected.

For this to work, we need a rule that picks a single `impl` in the case where
there are multiple `impl` definitions that match a particular type and interface
combination. This is called _specialization_ when the rule is that most specific
implementation is chosen, for some definition of specific.

#### Type structure of an impl declaration

Given an impl declaration, find the type structure by deleting deduced
parameters and replacing type parameters by a `?`. The type structure of this
declaration:

```
impl forall [T:! ..., U:! ...] Foo(T, i32) as Bar(String, U) { ... }
```

is:

```
impl Foo(?, i32) as Bar(String, ?)
```

To get a uniform representation across different `impl` definitions, before type
parameters are replaced the declarations are normalized as follows:

-   For impls declared lexically inline in a class definition, the type is added
    between the `impl` and `as` keywords if the type is left out.
-   Pointer types `T*` are replaced with `Ptr(T)`.
-   The `external` keyword is removed, if present.
-   The `forall` clause introducing type parameters is removed, if present.
-   Any `where` clauses that are setting associated constants or types are
    removed.

The type structure will always contain a single interface name, which is the
name of the interface being implemented, and some number of type names. Type
names can be in the `Self` type to the left of the `as` keyword, or as
parameters to other types or the interface. These names must always be defined
either in the current library or be publicly defined in some library this
library depends on.

#### Orphan rule

To achieve coherence, we need to ensure that any given impl can only be defined
in a library that must be imported for it to apply. Specifically, given a
specific type and specific interface, impls that can match can only be in
libraries that must have been imported to name that type or interface. This is
achieved with the _orphan rule_.

**Orphan rule:** Some name from the type structure of an `impl` declaration must
be defined in the same library as the `impl`, that is some name must be _local_.

Only the implementing interface and types (self type and type parameters) in the
type structure are relevant here; an interface mentioned in a constraint is not
sufficient since it
[need not be imported](/proposals/p0920.md#orphan-rule-could-consider-interface-requirements-in-blanket-impls).

Since Carbon in addition requires there be no cyclic library dependencies, we
conclude that there is at most one library that can define impls with a
particular type structure.

#### Overlap rule

Given a specific concrete type, say `Foo(bool, i32)`, and an interface, say
`Bar(String, f32)`, the overlap rule picks, among all the matching impls, which
type structure is considered "most specific" to use as the implementation of
that type for that interface.

Given two different type structures of impls matching a query, for example:

```
impl Foo(?, i32) as Bar(String, ?)
impl Foo(?, ?) as Bar(String, f32)
```

We pick the type structure with a non-`?` at the first difference as most
specific. Here we see a difference between `Foo(?, i32)` and `Foo(?, ?)`, so we
select the one with `Foo(?, i32)`, ignoring the fact that it has another `?`
later in its type structure

This rule corresponds to a depth-first traversal of the type tree to identify
the first difference, and then picking the most specific choice at that
difference.

#### Prioritization rule

Since at most one library can define impls with a given type structure, all
impls with a given type structure must be in the same library. Furthermore by
the [impl declaration access rules](#access), they will be defined in the API
file for the library if they could match any query from outside the library. If
there is more than one impl with that type structure, they must be
[defined](#implementing-interfaces) or [declared](#declaring-implementations)
together in a prioritization block. Once a type structure is selected for a
query, the first impl in the prioritization block that matches is selected.

**Open question:** How are prioritization blocks written? A block starts with a
keyword like `match_first` or `impl_priority` and then a sequence of impl
declarations inside matching curly braces `{` ... `}`.

```
match_first {
  // If T is Foo prioritized ahead of T is Bar
  impl forall [T:! Foo] T as Bar { ... }
  impl forall [T:! Baz] T as Bar { ... }
}
```

**Open question:** How do we pick between two different prioritization blocks
when they contain a mixture of type structures? There are three options:

-   Prioritization blocks implicitly define all non-empty intersections of
    contained impls, which are then selected by their type structure.
-   The compiler first picks the impl with the type pattern most favored for the
    query, and then picks the definition of the highest priority matching impl
    in the same prioritization block.
-   All the impls in a prioritization block are required to have the same type
    structure, at a cost in expressivity.

To see the difference between the first two options, consider two libraries with
type structures as follows:

-   Library B has `impl (A, ?, ?, D) as I` and `impl (?, B, ?, D) as I` in the
    same prioritization block.
-   Library C has `impl (A, ?, C, ?) as I`.

For the query `(A, B, C, D) as I`, using the intersection rule, library B is
considered to have the intersection impl with type structure
`impl (A, B, ?, D) as I` which is the most specific. If we instead just
considered the rules mentioned explicitly, then `impl (A, ?, C, ?) as I` from
library C is the most specific. The advantage of the implicit intersection rule
is that if library B is changed to add an impl with type structure
`impl (A, B, ?, D) as I`, it won't shift which library is serving that query.

#### Acyclic rule

A cycle is when a query, such as "does type `T` implement interface `I`?",
considers an impl that might match, and whether that impl matches is ultimately
dependent on whether that query is true. These are cycles in the graph of (type,
interface) pairs where there is an edge from pair A to pair B if whether type A
implements interface A determines whether type B implements interface B.

The test for whether something forms a cycle needs to be precise enough, and not
erase too much information when considering this graph, that these impls are not
considered to form cycles with themselves:

```
impl forall [T:! Printable] Optional(T) as Printable;
impl forall [T:! Type, U:! ComparableTo(T)] U as ComparableTo(Optional(T));
```

**Example:** If `T` implements `ComparableWith(U)`, then `U` should implement
`ComparableWith(T)`.

```
external impl forall [U:! Type, T:! ComparableWith(U)]
    U as ComparableWith(T);
```

This is a cycle where which types implement `ComparableWith` determines which
types implement the same interface.

**Example:** Cycles can create situations where there are multiple ways of
selecting impls that are inconsistent with each other. Consider an interface
with two blanket `impl` declarations:

```
class Y {}
class N {}
interface True {}
impl Y as True {}
interface Z(T:! Type) { let Cond:! Type; }
match_first {
  impl forall [T:! Type, U:! Z(T) where .Cond is True] T as Z(U)
      where .Cond = N { }
  impl forall [T:! Type, U:! Type] T as Z(U)
      where .Cond = Y { }
}
```

What is `i8.(Z(i16).Cond)`? It depends on which of the two blanket impls are
selected.

-   An implementation of `Z(i16)` for `i8` could come from the first blanket
    impl with `T == i8` and `U == i16` if `i16 is Z(i8)` and
    `i16.(Z(i8).Cond) == Y`. This condition is satisfied if `i16` implements
    `Z(i8)` using the second blanket impl. In this case,
    `i8.(Z(i16).Cond) == N`.
-   Equally well `Z(i8)` could be implemented for `i16` using the first blanket
    impl and `Z(i16)` for `i8` using the second. In this case,
    `i8.(Z(i16).Cond) == Y`.

There is no reason to to prefer one of these outcomes over the other.

**Example:** Further, cycles can create contradictions in the type system:

```
class A {}
class B {}
class C {}
interface D(T:! Type) { let Cond:! Type; }
match_first {
  impl forall [T:! Type, U:! D(T) where .Cond = B] T as D(U)
      where .Cond = C { }
  impl forall [T:! Type, U:! D(T) where .Cond = A] T as D(U)
      where .Cond = B { }
  impl forall [T:! Type, U:! Type] T as D(U)
      where .Cond = A { }
}
```

What is `i8.(D(i16).Cond)`? The answer is determined by which blanket impl is
selected to implement `D(i16)` for `i8`:

-   If the third blanket impl is selected, then `i8.(D(i16).Cond) == A`. This
    implies that `i16.(D(i8).Cond) == B` using the second blanket impl. If that
    is true, though, then our first impl choice was incorrect, since the first
    blanket impl applies and is higher priority. So `i8.(D(i16).Cond) == C`. But
    that means that `i16 as D(i8)` can't use the second blanket impl.
-   For the second blanket impl to be selected, so `i8.(D(i16).Cond) == B`,
    `i16.(D(i8).Cond)` would have to be `A`. This happens when `i16` implements
    `D(i8)` using the third blanket impl. However, `i8.(D(i16).Cond) == B` means
    that there is a higher priority implementation of `D(i8).Cond` for `i16`.

In either case, we arrive at a contradiction.

The workaround for this problem is to either split an interface in the cycle in
two, with a blanket implementation of one from the other, or move some of the
criteria into a [named constraint](#named-constraints).

**Concern:** Cycles could be spread out across libraries with no dependencies
between them. This means there can be problems created by a library that are
only detected by its users.

**Open question:** Should Carbon reject cycles in the absence of a query? The
two options here are:

-   Combining impls gives you an immediate error if there exists queries using
    those impls that have cycles.
-   Only when a query reveals a cyclic dependency is an error reported.

**Open question:** In the second case, should we ignore cycles if they don't
affect the result of the query? For example, the cycle might be among
implementations that are lower priority.

#### Termination rule

It is possible to define a set of impls where there isn't a cycle, but the graph
is infinite. Without some rule to prevent exhaustive exploration of the graph,
determining whether a type implements an interface could run forever.

**Example:** It could be that `A` implements `B`, so `A is B` if
`Optional(A) is B`, if `Optional(Optional(A)) is B`, and so on. This could be
the result of a single impl:

```
impl forall [A:! Type where Optional(.Self) is B] A as B { ... }
```

This problem can also result from a chain of impls, as in `A is B` if `A* is C`,
if `Optional(A) is B`, and so on.

Rust solves this problem by imposing a recursion limit, much like C++ compilers
use to terminate template recursion. This goes against
[Carbon's goal of predictability in generics](goals.md#predictability), but at
this time there are no known alternatives. Unfortunately, the approach Carbon
uses to avoid undecidability for type equality,
[providing an explicit proof in the source](#manual-type-equality), can't be
used here. The code triggering the query asking whether some type implements an
interface will typically be generic code with know specific knowledge about the
types involved, and won't be in a position to provide a manual proof that the
implementation should exist.

**Open question:** Is there some restriction on `impl` declarations that would
allow our desired use cases, but allow the compiler to detect non-terminating
cases? Perhaps there is some sort of complexity measure Carbon can require
doesn't increase when recursing?

### `final` impls

There are cases where knowing that a parameterized impl won't be specialized is
particularly valuable. This could let the compiler know the return type of a
generic function call, such as using an operator:

```
// Interface defining the behavior of the prefix-* operator
interface Deref {
  let Result:! Type;
  fn DoDeref[me: Self]() -> Result;
}

// Types implementing `Deref`
class Ptr(T:! Type) {
  ...
  external impl as Deref where .Result = T {
    fn DoDeref[me: Self]() -> Result { ... }
  }
}
class Optional(T:! Type) {
  ...
  external impl as Deref where .Result = T {
    fn DoDeref[me: Self]() -> Result { ... }
  }
}

fn F[T:! Type](x: T) {
  // uses Ptr(T) and Optional(T) in implementation
}
```

The concern is the possibility of specializing `Optional(T) as Deref` or
`Ptr(T) as Deref` for a more specific `T` means that the compiler can't assume
anything about the return type of `Deref.DoDeref` calls. This means `F` would in
practice have to add a constraint, which is both verbose and exposes what should
be implementation details:

```
fn F[T:! Type where Optional(T).(Deref.Result) == .Self
                and Ptr(T).(Deref.Result) == .Self](x: T) {
  // uses Ptr(T) and Optional(T) in implementation
}
```

To mark an impl as not able to be specialized, prefix it with the keyword
`final`:

```
class Ptr(T:! Type) {
  ...
  // Note: added `final`
  final external impl as Deref where .Result = T {
    fn DoDeref[me: Self]() -> Result { ... }
  }
}
class Optional(T:! Type) {
  ...
  // Note: added `final`
  final external impl as Deref where .Result = T {
    fn DoDeref[me: Self]() -> Result { ... }
  }
}

// ❌ Illegal: external impl Ptr(i32) as Deref { ... }
// ❌ Illegal: external impl Optional(i32) as Deref { ... }
```

This prevents any higher-priority impl that overlaps a final impl from being
defined. Further, if the Carbon compiler sees a matching final impl, it can
assume it won't be specialized so it can use the assignments of the associated
types in that impl definition.

```
fn F[T:! Type](x: T) {
  var p: Ptr(T) = ...;
  // *p has type `T`
  var o: Optional(T) = ...;
  // *o has type `T`
}
```

#### Libraries that can contain `final` impls

To prevent the possibility of two unrelated libraries defining conflicting
impls, Carbon restricts which libraries may declare an impl as `final` to only:

-   the library declaring the impl's interface and
-   the library declaring the root of the `Self` type.

This means:

-   A blanket impl with type structure `impl ? as MyInterface(...)` may only be
    defined in the same library as `MyInterface`.
-   An impl with type structure `impl MyType(...) as MyInterface(...)` may be
    defined in the library with `MyType` or `MyInterface`.

These restrictions ensure that the Carbon compiler can locally check that no
higher-priority impl is defined superseding a `final` impl.

-   An impl with type structure `impl MyType(...) as MyInterface(...)` defined
    in the library with `MyType` must import the library defining `MyInterface`,
    and so will be able to see any final blanket impls.
-   A blanket impl with type structure
    `impl ? as MyInterface(...ParameterType(...)...)` may be defined in the
    library with `ParameterType`, but that library must import the library
    defining `MyInterface`, and so will be able to see any `final` blanket impls
    that might overlap. A final impl with type structure
    `impl MyType(...) as MyInterface(...)` would be given priority over any
    overlapping blanket impl defined in the `ParameterType` library.
-   An impl with type structure
    `impl MyType(...ParameterType(...)...) as MyInterface(...)` may be defined
    in the library with `ParameterType`, but that library must import the
    libraries defining `MyType` and `MyInterface`, and so will be able to see
    any `final` impls that might overlap.

### Comparison to Rust

Rust has been designing a specialization feature, but it has not been completed.
Luckily, Rust team members have done a lot of blogging during their design
process, so Carbon can benefit from the work they have done. However, getting
specialization to work for Rust is complicated by the need to maintain
compatibility with existing Rust code. This motivates a number of Rust rules
where Carbon can be simpler. As a result there are both similarities and
differences between the Carbon and Rust plans:

-   A Rust impl defaults to not being able to be specialized, with a `default`
    keyword used to opt-in to allowing specialization, reflecting the existing
    code base developed without specialization. Carbon impls default to allowing
    specialization, with restrictions on which may be declared `final`.
-   Since Rust impls are not specializable by default, generic functions can
    assume that if a matching blanket impl is found, the associated types from
    that impl will be used. In Carbon, if a generic function requires an
    associated type to have a particular value, the function commonly will need
    to state that using an explicit constraint.
-   Carbon will not have the "fundamental" attribute used by Rust on types or
    traits, as described in
    [Rust RFC 1023: "Rebalancing Coherence"](https://rust-lang.github.io/rfcs/1023-rebalancing-coherence.html).
-   Carbon will not use "covering" rules, as described in
    [Rust RFC 2451: "Re-Rebalancing Coherence"](https://rust-lang.github.io/rfcs/2451-re-rebalancing-coherence.html)
    and
    [Little Orphan Impls: The covered rule](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/#the-covered-rule).
-   Like Rust, Carbon does use ordering, favoring the `Self` type and then the
    parameters to the interface in left-to-right order, see
    [Rust RFC 1023: "Rebalancing Coherence"](https://rust-lang.github.io/rfcs/1023-rebalancing-coherence.html)
    and
    [Little Orphan Impls: The ordered rule](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/#the-ordered-rule),
    but the specifics are different.
-   Carbon is not planning to support any inheritance of implementation between
    impls. This is more important to Rust since Rust does not support class
    inheritance for implementation reuse. Rust has considered multiple
    approaches here, see
    [Aaron Turon: "Specialize to Reuse"](http://aturon.github.io/tech/2015/09/18/reuse/)
    and
    [Supporting blanket impls in specialization](http://smallcultfollowing.com/babysteps/blog/2016/10/24/supporting-blanket-impls-in-specialization/).
-   [Supporting blanket impls in specialization](http://smallcultfollowing.com/babysteps/blog/2016/10/24/supporting-blanket-impls-in-specialization/)
    proposes a specialization rule for Rust that considers type structure before
    other constraints, as in Carbon, though the details differ.
-   Rust has more orphan restrictions to avoid there being cases where it is
    ambiguous which impl should be selected. Carbon instead has picked a total
    ordering on type structures, picking one as higher priority even without one
    being more specific in the sense of only applying to a subset of types.

## Forward declarations and cyclic references

Interfaces, named constraints, and their implementations may be forward declared
and then later defined. This is needed to allow cyclic references, for example
when declaring the edges and nodes of a graph. It is also a tool that may be
used to make code more readable.

The [interface](#interfaces), [named constraint](#named-constraints), and
[implementation](#implementing-interfaces) sections describe the syntax for
their _definition_, which consists of a declaration followed by a body contained
in curly braces `{` ... `}`. A _forward declaration_ is a declaration followed
by a semicolon `;`. A forward declaration is a promise that the entity being
declared will be defined later. Between the first declaration of an entity,
which may be in a forward declaration or the first part of a definition, and the
end of the definition the interface or implementation is called _incomplete_.
There are additional restrictions on how the name of an incomplete entity may be
used.

### Declaring interfaces and named constraints

The declaration for an interface or named constraint consists of:

-   an optional access-control keyword like `private`,
-   the keyword introducer `interface`, `constraint`, or `template constraint`,
-   the name of the interface or constraint, and
-   the parameter list, if any.

The name of an interface or constraint can not be used until its first
declaration is complete. In particular, it is illegal to use the name of the
interface in its parameter list. There is a
[workaround](#interfaces-with-parameters-constrained-by-the-same-interface) for
the use cases when this would come up.

An expression forming a constraint, such as `C & D`, is incomplete if any of the
interfaces or constraints used in the expression are incomplete. A constraint
expression using a [`where` clause](#where-constraints), like `C where ...`, is
invalid if `C` is incomplete, since there is no way to look up member names of
`C` that appear after `where`.

An interface or named constraint may be forward declared subject to these rules:

-   The definition must be in the same file as the declaration.
-   Only the first declaration may have an access-control keyword.
-   An incomplete interface or named constraint may be used as constraints in
    declarations of types, functions, interfaces, or named constraints. This
    includes an `impl as` or `extends` declaration inside an interface or named
    constraint, but excludes specifying the values for associated constants
    because that would involve name lookup into the incomplete constraint.
-   An attempt to define the body of a generic function using an incomplete
    interface or named constraint is illegal.
-   An attempt to call a generic function using an incomplete interface or named
    constraint in its signature is illegal.
-   Any name lookup into an incomplete interface or named constraint is an
    error. For example, it is illegal to attempt to access a member of an
    interface using `MyInterface.MemberName` or constrain a member using a
    `where` clause.

### Declaring implementations

The declaration of an interface implementation consists of:

-   optional modifier keywords `final`, `external`,
-   the keyword introducer `impl`,
-   an optional deduced parameter list in square brackets `[`...`]`,
-   a type, including an optional parameter pattern,
-   the keyword `as`, and
-   a [type-of-type](#type-of-types), including an optional
    [parameter pattern](#parameterized-interfaces) and
    [`where` clause](#where-constraints) assigning
    [associated constants](#associated-constants) and
    [associated types](#associated-types).

An implementation of an interface for a type may be forward declared subject to
these rules:

-   The definition must be in the same library as the declaration. They must
    either be in the same file, or the declaration can be in the API file and
    the definition in an impl file. **Future work:** Carbon may require the
    definition of [parameterized impls](#parameterized-impls) to be in the API
    file, to support separate compilation.
-   If there is both a forward declaration and a definition, only the first
    declaration must specify the assignment of associated constants with a
    `where` clause. Later declarations may omit the `where` clause by writing
    `where _` instead.
-   You may forward declare an implementation of a defined interface but not an
    incomplete interface. This allows the assignment of associated constants in
    the `impl` declaration to be verified. An impl forward declaration may be
    for any declared type, whether it is incomplete or defined. Note that this
    does not apply to `impl as` declarations in an interface or named constraint
    definition, as those are considered interface requirements not forward
    declarations.
-   Every internal implementation must be declared (or defined) inside the scope
    of the class definition. It may also be declared before the class definition
    or defined afterwards. Note that the class itself is incomplete in the scope
    of the class definition, but member function bodies defined inline are
    processed
    [as if they appeared immediately after the end of the outermost enclosing class](/docs/project/principles/information_accumulation.md#exceptions).
-   For [coherence](goals.md#coherence), we require that any impl that matches
    an [impl lookup](#impl-lookup) query in the same file, must be declared
    before the query. This can be done with a definition or a forward
    declaration.

### Matching and agreeing

Carbon needs to determine if two declarations match in order to say which
definition a forward declaration corresponds to and to verify that nothing is
defined twice. Declarations that match must also agree, meaning they are
consistent with each other.

Interface and named constraint declarations match if their names are the same
after name and alias resolution. To agree:

-   The introducer keyword or keywords much be the same.
-   The types and order of parameters in the parameter list, if any, must match.
    The parameter names may be omitted, but if they are included in both
    declarations, they must match.
-   Types agree if they correspond to the same expression tree, after name and
    alias resolution and canonicalization of parentheses. Note that no other
    evaluation of type expressions is performed.

Interface implementation declarations match if the type and interface
expressions match:

-   If the type part is omitted, it is rewritten to `Self` in the context of the
    declaration.
-   `Self` is rewritten to its meaning in the scope it is used. In a class
    scope, this should match the type name and optional parameter expression
    after `class`. So in `class MyClass extends MyBase { ... }`, `Self` is
    rewritten to `MyClass`. In `class Vector(T:! Movable) { ... }`, `Self` is
    rewritten to `Vector(T:! Movable)`.
-   Types match if they have the same name after name and alias resolution and
    the same parameters, or are the same type parameter.
-   Interfaces match if they have the same name after name and alias resolution
    and the same parameters. Note that a named constraint that is equivalent to
    an interface, as in `constraint Equivalent { extends MyInterface; }`, is not
    considered to match.

For implementations to agree:

-   The presence of modifier keywords such as `external` before `impl` must
    match between a forward declaration and definition.
-   If either declaration includes a `where` clause, they must both include one.
    If neither uses `where _`, they must match in that they produce the
    associated constants with the same values considered separately.

### Declaration examples

```
// Forward declaration of interfaces
interface Interface1;
interface Interface2;
interface Interface3;
interface Interface4;
interface Interface5;
interface Interface6;

// Forward declaration of class type
class MyClass;

// ❌ Illegal: Can't declare implementation of incomplete
//             interface.
// external impl MyClass as Interface1;

// Definition of interfaces that were previously declared
interface Interface1 {
  let T1:! Type;
}
interface Interface2 {
  let T2:! Type;
}
interface Interface3 {
  let T3:! Type;
}
interface Interface4 {
  let T4:! Type;
}

// Forward declaration of external implementations
external impl MyClass as Interface1 where .T1 = i32;
external impl MyClass as Interface2 where .T2 = bool;

// Forward declaration of an internal implementation
impl MyClass as Interface3 where .T3 = f32;
impl MyClass as Interface4 where .T4 = String;

interface Interface5 {
  let T5:! Type;
}
interface Interface6 {
  let T6:! Type;
}

// Definition of the previously declared class type
class MyClass {
  // Definition of previously declared external impl.
  // Note: no need to repeat assignments to associated
  // constants.
  external impl as Interface1 where _ { }

  // Definition of previously declared internal impl.
  // Note: allowed even though `MyClass` is incomplete.
  // Note: allowed but not required to repeat `where`
  // clause.
  impl as Interface3 where .T3 = f32 { }

  // Redeclaration of previously declared internal impl.
  // Every internal implementation must be declared in
  // the class definition.
  impl as Interface4 where _;

  // Forward declaration of external implementation.
  external impl MyClass as Interface5 where .T5 = u64;

  // Forward declaration of internal implementation.
  impl MyClass as Interface6 where .T6 = u8;
}

// It would be legal to move the following definitions
// from the API file to the implementation file for
// this library.

// Definition of previously declared external impls.
external impl MyClass as Interface2 where _ { }
external impl MyClass as Interface5 where _ { }

// Definition of previously declared internal impls.
impl MyClass as Interface4 where _ { }
impl MyClass as Interface6 where _ { }
```

### Example of declaring interfaces with cyclic references

In this example, `Node` has an `EdgeType` associated type that is constrained to
implement `Edge`, and `Edge` has a `NodeType` associated type that is
constrained to implement `Node`. Furthermore, the `NodeType` of an `EdgeType` is
the original type, and the other way around. This is accomplished by naming and
then forward declaring the constraints that can't be stated directly:

```
// Forward declare interfaces used in
// parameter lists of constraints.
interface Edge;
interface Node;

// Forward declare named constraints used in
// interface definitions.
private constraint EdgeFor(N:! Node);
private constraint NodeFor(E:! Edge);

// Define interfaces using named constraints.
interface Edge {
  let NodeType:! NodeFor(Self);
  fn Head[me: Self]() -> NodeType;
}
interface Node {
  let EdgeType:! EdgeFor(Self);
  fn Edges[me: Self]() -> Vector(EdgeType);
}

// Now that the interfaces are defined, can
// refer to members of the interface, so it is
// now legal to define the named constraints.
constraint EdgeFor(N:! Node) {
  extends Edge where .NodeType == N;
}
constraint NodeFor(E:! Edge) {
  extends Node where .EdgeType == E;
}
```

### Interfaces with parameters constrained by the same interface

To work around
[the restriction about not being able to name an interface in its parameter list](#declaring-interfaces-and-named-constraints),
instead include that requirement in the body of the interface.

```
// Want to require that `T` satisfies `CommonType(Self)`,
// but that can't be done in the parameter list.
interface CommonType(T:! Type) {
  let Result:! Type;
  // Instead add the requirement inside the definition.
  impl T as CommonType(Self);
}
```

Note however that `CommonType` is still incomplete inside its definition, so no
constraints on members of `CommonType` are allowed.

```
interface CommonType(T:! Type) {
  let Result:! Type;
  // ❌ Illegal: `CommonType` is incomplete
  impl T as CommonType(Self) where .Result == Result;
}
```

Instead, a forward-declared named constraint can be used in place of the
constraint that can only be defined later. This is
[the same strategy used to work around cyclic references](#example-of-declaring-interfaces-with-cyclic-references).

```
private constraint CommonTypeResult(T:! Type, R:! Type);

interface CommonType(T:! Type) {
  let Result:! Type;
  // ✅ Allowed: `CommonTypeResult` is incomplete, but
  //             no members are accessed.
  impl T as CommonTypeResult(Self, Result);
}

constraint CommonTypeResult(T:! Type, R:! Type) {
  extends CommonType(T) where .Result == R;
}
```

## Interface members with definitions

Interfaces may provide definitions for members, such as a function body for an
associated function or method or a value for an associated constant. If these
definitions may be overridden in implementations, they are called "defaults" and
prefixed with the `default` keyword. Otherwise they are called "final members"
and prefixed with the `final` keyword.

### Interface defaults

An interface may provide a default implementation of methods in terms of other
methods in the interface.

```
interface Vector {
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: f64) -> Self;
  // Default definition of `Invert` calls `Scale`.
  default fn Invert[me: Self]() -> Self {
    return me.Scale(-1.0);
  }
}
```

A default function or method may also be defined out of line, later in the same
file as the interface definition:

```
interface Vector {
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: f64) -> Self;
  default fn Invert[me: Self]() -> Self;
}
// `Vector` is considered complete at this point,
// even though `Vector.Invert` is still incomplete.
fn Vector.Invert[me: Self]() -> Self {
  return me.Scale(-1.0);
}
```

An impl of that interface for a type may omit a definition of `Invert` to use
the default, or provide a definition to override the default.

Interface defaults are helpful for [evolution](#evolution), as well as reducing
boilerplate. Defaults address the gap between the minimum necessary for a type
to provide the desired functionality of an interface and the breadth of API that
developers desire. As an example, in Rust the
[iterator trait](https://doc.rust-lang.org/std/iter/trait.Iterator.html) only
has one required method but dozens of "provided methods" with defaults.

Defaults may also be provided for associated constants, such as associated
types, and interface parameters, using the `= <default value>` syntax.

```
interface Add(Right:! Type = Self) {
  default let Result:! Type = Self;
  fn DoAdd[me: Self](right: Right) -> Result;
}

impl String as Add() {
  // Right == Result == Self == String
  fn DoAdd[me: Self](right: Self) -> Self;
}
```

Note that `Self` is a legal default value for an associated type or type
parameter. In this case the value of those names is not determined until `Self`
is, so `Add()` is equivalent to the constraint:

```
// Equivalent to Add()
constraint AddDefault {
  extends Add(Self);
}
```

Note also that the parenthesis are required after `Add`, even when all
parameters are left as their default values.

More generally, default expressions may reference other associated types or
`Self` as parameters to type constructors. For example:

```
interface Iterator {
  let Element:! Type;
  default let Pointer:! Type = Element*;
}
```

Carbon does **not** support providing a default implementation of a required
interface.

```
interface TotalOrder {
  fn TotalLess[me: Self](right: Self) -> bool;
  // ❌ Illegal: May not provide definition
  //             for required interface.
  impl as PartialOrder {
    fn PartialLess[me: Self](right: Self) -> bool {
      return me.TotalLess(right);
    }
  }
}
```

The workaround for this restriction is to use a [blanket impl](#blanket-impls)
instead:

```
interface TotalOrder {
  fn TotalLess[me: Self](right: Self) -> bool;
  impl as PartialOrder;
}

external impl forall [T:! TotalOrder] T as PartialOrder {
  fn PartialLess[me: Self](right: Self) -> bool {
    return me.TotalLess(right);
  }
}
```

Note that by the [orphan rule](#orphan-rule), this blanket impl must be defined
in the same library as `PartialOrder`.

**Comparison with other languages:** Rust supports specifying defaults for
[methods](https://doc.rust-lang.org/book/ch10-02-traits.html#default-implementations),
[interface parameters](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading),
and
[associated constants](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants-examples).
Rust has found them valuable.

### `final` members

As an alternative to providing a definition of an interface member as a default,
members marked with the `final` keyword will not allow that definition to be
overridden in impls.

```
interface TotalOrder {
  fn TotalLess[me: Self](right: Self) -> bool;
  final fn TotalGreater[me: Self](right: Self) -> bool {
    return right.TotalLess(me);
  }
}

class String {
  impl as TotalOrder {
    fn TotalLess[me: Self](right: Self) -> bool { ... }
    // ❌ Illegal: May not provide definition of final
    //             method `TotalGreater`.
    fn TotalGreater[me: Self](right: Self) -> bool { ... }
  }
}

interface Add(T:! Type = Self) {
  // `AddWith` *always* equals `T`
  final let AddWith:! Type = T;
  // Has a *default* of `Self`
  let Result:! Type = Self;
  fn DoAdd[me: Self](right: AddWith) -> Result;
}
```

Final members may also be defined out-of-line:

```
interface TotalOrder {
  fn TotalLess[me: Self](right: Self) -> bool;
  final fn TotalGreater[me: Self](right: Self) -> bool;
}
// `TotalOrder` is considered complete at this point, even
// though `TotalOrder.TotalGreater` is not yet defined.
fn TotalOrder.TotalGreater[me: Self](right: Self) -> bool {
 return right.TotalLess(me);
}
```

There are a few reasons for this feature:

-   When overriding would be inappropriate.
-   Matching the functionality of non-virtual methods in base classes, so
    interfaces can be a replacement for inheritance.
-   Potentially reduce dynamic dispatch when using the interface in a
    [`DynPtr`](#dynamic-types).

Note that this applies to associated entities, not interface parameters.

## Interface requiring other interfaces revisited

Recall that an
[interface can require another interface be implemented for the type](#interface-requiring-other-interfaces),
as in:

```
interface Iterable {
  impl as Equatable;
  // ...
}
```

This states that the type implementing the interface `Iterable`, which in this
context is called `Self`, must also implement the interface `Equatable`. As is
done with [conditional conformance](#conditional-conformance), we allow another
type to be specified between `impl` and `as` to say some type other than `Self`
must implement an interface. For example,

```
interface IntLike {
  impl i32 as As(Self);
  // ...
}
```

says that if `Self` implements `IntLike`, then `i32` must implement `As(Self)`.
Similarly,

```
interface CommonTypeWith(T:! Type) {
  impl T as CommonTypeWith(Self);
  // ...
}
```

says that if `Self` implements `CommonTypeWith(T)`, then `T` must implement
`CommonTypeWith(Self)`.

The previous description of `impl as` in an interface definition matches the
behavior of using a default of `Self` when the type between `impl` and `as` is
omitted. So the previous definition of `interface Iterable` is equivalent to:

```
interface Iterable {
  // ...
  impl Self as Equatable;
  // Equivalent to: impl as Equatable;
}
```

When implementing an interface with an `impl as` requirement, that requirement
must be satisfied by an implementation in an imported library, an implementation
somewhere in the same file, or a constraint in the impl declaration.
Implementing the requiring interface is a promise that the requirement will be
implemented. This is like a
[forward declaration of an impl](#declaring-implementations) except that the
definition can be broader instead of being required to match exactly.

```
// `Iterable` requires `Equatable`, so there must be some
// impl of `Equatable` for `Vector(i32)` in this file.
external impl Vector(i32) as Iterable { ... }

fn RequiresEquatable[T:! Equatable](x: T) { ... }
fn ProcessVector(v: Vector(i32)) {
  // ✅ Allowed since `Vector(i32)` is known to
  // implement `Equatable`.
  RequiresEquatable(v);
}

// Satisfies the requirement that `Vector(i32)` must
// implement `Equatable` since `i32` is `Equatable`.
external impl forall [T:! Equatable] Vector(T) as Equatable { ... }
```

In some cases, the interface's requirement can be trivially satisfied by the
implementation itself, as in:

```
impl forall [T:! Type] T as CommonTypeWith(T) { ... }
```

Here is an example where the requirement of interface `Iterable` that the type
implements interface `Equatable` is satisfied by a constraint in the `impl`
declaration:

```
class Foo(T:! Type) {}
// This is allowed because we know that an `impl Foo(T) as Equatable`
// will exist for all types `T` for which this impl is used, even
// though there's neither an imported impl nor an impl in this file.
external impl forall [T:! Type where Foo(T) is Equatable]
    Foo(T) as Iterable {}
```

This might be used to provide an implementation of `Equatable` for types that
already satisfy the requirement of implementing `Iterable`:

```
class Bar {}
external impl Foo(Bar) as Equatable {}
// Gives `Foo(Bar) is Iterable` using the blanket impl of
// `Iterable` for `Foo(T)`.
```

### Requirements with `where` constraints

An interface implementation requirement with a `where` clause is harder to
satisfy. Consider an interface `B` that has a requirement that interface `A` is
also implemented.

```
interface A(T:! Type) {
  let Result:! Type;
}
interface B(T:! Type) {
  impl as A(T) where .Result == i32;
}
```

An implementation of `B` for a set of types can only be valid if there is a
visible implementation of `A` with the same `T` parameter for those types with
the `.Result` associated type set to `i32`. That is
[not sufficient](/proposals/p1088.md#less-strict-about-requirements-with-where-clauses),
though, unless the implementation of `A` can't be specialized, either because it
is [marked `final`](#final-impls) or is not
[parameterized](#parameterized-impls). Implementations in other libraries can't
make `A` be implemented for fewer types, but can cause `.Result` to have a
different assignment.

## Observing a type implements an interface

An [`observe` declaration](#observe-declarations) can be used to show that two
types are equal so code can pass type checking without explicitly writing casts,
without requiring the compiler to do a unbounded search that may not terminate.
An `observe` declaration can also be used to show that a type implements an
interface, in cases where the compiler will not work this out for itself.

### Observing interface requirements

One situation where this occurs is when there is a chain of
[interfaces requiring other interfaces](#interface-requiring-other-interfaces-revisited).
During the `impl` validation done during type checking, Carbon will only
consider the interfaces that are direct requirements of the interfaces the type
is known to implement. An `observe...is` declaration can be used to add an
interface that is a direct requirement to the set of interfaces whose direct
requirements will be considered for that type. This allows a developer to
provide a proof that there is a sequence of requirements that demonstrate that a
type implements an interface, as in this example:

```
interface A { }
interface B { impl as A; }
interface C { impl as B; }
interface D { impl as C; }

fn RequiresA[T:! A](x: T);
fn RequiresC[T:! C](x: T);
fn RequiresD[T:! D](x: T) {
  // ✅ Allowed: `D` directly requires `C` to be implemented.
  RequiresC(x);

  // ❌ Illegal: No direct connection between `D` and `A`.
  // RequiresA(x);

  // `T` is `D` and `D` directly requires `C` to be
  // implemented.
  observe T is C;

  // `T` is `C` and `C` directly requires `B` to be
  // implemented.
  observe T is B;

  // ✅ Allowed: `T` is `B` and `B` directly requires
  //             `A` to be implemented.
  RequiresA(x);
}
```

Note that `observe` statements do not affect the selection of impls during code
generation. For coherence, the impl used for a (type, interface) pair must
always be the same, independent of context. The
[termination rule](#termination-rule) governs when compilation may fail when the
compiler can't determine the impl to select.

### Observing blanket impls

An `observe...is` declaration can also be used to observe that a type implements
an interface because there is a [blanket impl](#blanket-impls) in terms of
requirements a type is already known to satisfy. Without an `observe`
declaration, Carbon will only use blanket impls that are directly satisfied.

```
interface A { }
interface B { }
interface C { }
interface D { }

impl forall [T:! A] T as B { }
impl forall [T:! B] T as C { }
impl forall [T:! C] T as D { }

fn RequiresD(T:! D)(x: T);
fn RequiresB(T:! B)(x: T);

fn RequiresA(T:! A)(x: T) {
  // ✅ Allowed: There is a blanket implementation
  //             of `B` for types implementing `A`.
  RequiresB(x);

  // ❌ Illegal: No implementation of `D` for type
  //             `T` implementing `A`
  // RequiresD(x);

  // There is a blanket implementation of `B` for
  // types implementing `A`.
  observe T is B;

  // There is a blanket implementation of `C` for
  // types implementing `B`.
  observe T is C;

  // ✅ Allowed: There is a blanket implementation
  //             of `D` for types implementing `C`.
  RequiresD(x);
}
```

In the case of an error, a quality Carbon implementation will do a deeper search
for chains of requirements and blanket impls and suggest `observe` declarations
that would make the code compile if any solution is found.

## Operator overloading

Operations are overloaded for a type by implementing an interface specific to
that interface for that type. For example, types implement the `Negatable`
interface to overload the unary `-` operator:

```
// Unary `-`.
interface Negatable {
  let Result:! Type = Self;
  fn Negate[me: Self]() -> Result;
}
```

Expressions using operators are rewritten into calls to these interface methods.
For example, `-x` would be rewritten to `x.(Negatable.Negate)()`.

The interfaces and rewrites used for a given operator may be found in the
[expressions design](/docs/design/expressions/README.md).
[Question-for-leads issue #1058](https://github.com/carbon-language/carbon-lang/issues/1058)
defines the naming scheme for these interfaces.

### Binary operators

Binary operators will have an interface that is
[parameterized](#parameterized-interfaces) based on the second operand. For
example, to say a type may be converted to another type using an `as`
expression, implement the
[`As` interface](/docs/design/expressions/as_expressions.md#extensibility):

```
interface As(Dest:! Type) {
  fn Convert[me: Self]() -> Dest;
}
```

The expression `x as U` is rewritten to `x.(As(U).Convert)()`. Note that the
parameterization of the interface means it can be implemented multiple times to
support multiple operand types.

Unlike `as`, for most binary operators the interface's argument will be the
_type_ of the right-hand operand instead of its _value_. Consider an interface
for a binary operator like `*`:

```
// Binary `*`.
interface MultipliableWith(U:! Type) {
  let Result:! Type = Self;
  fn Multiply[me: Self](other: U) -> Result;
}
```

A use of binary `*` in source code will be rewritten to use this interface:

```
var left: Meters = ...;
var right: f64 = ...;
var result: auto = left * right;
// Equivalent to:
var equivalent: left.(MultipliableWith(f64).Result)
    = left.(MultipliableWith(f64).Multiply)(right);
```

Note that if the types of the two operands are different, then swapping the
order of the operands will result in a different implementation being selected.
It is up to the developer to make those consistent when that is appropriate. The
standard library will provide [adapters](#adapting-types) for defining the
second implementation from the first, as in:

```
interface ComparableWith(RHS:! Type) {
  fn Compare[me: Self](right: RHS) -> CompareResult;
}

adapter ReverseComparison
    (T:! Type, U:! ComparableWith(RHS)) for T {
  impl as ComparableWith(U) {
    fn Compare[me: Self](right: RHS) -> CompareResult {
      return ReverseCompareResult(right.Compare(me));
    }
  }
}

external impl SongByTitle as ComparableWith(SongTitle);
external impl SongTitle as ComparableWith(SongByTitle)
    = ReverseComparison(SongTitle, SongByTitle);
```

In some cases the reverse operation may not be defined. For example, a library
might support subtracting a vector from a point, but not the other way around.

Further note that even if the reverse implementation exists,
[the impl prioritization rule](#prioritization-rule) might not pick it. For
example, if we have two types that support comparison with anything implementing
an interface that the other implements:

```
interface IntLike {
  fn AsInt[me: Self]() -> i64;
}

class EvenInt { ... }
external impl EvenInt as IntLike;
external impl EvenInt as ComparableWith(EvenInt);
// Allow `EvenInt` to be compared with anything that
// implements `IntLike`, in either order.
external impl forall [T:! IntLike] EvenInt as ComparableWith(T);
external impl forall [T:! IntLike] T as ComparableWith(EvenInt);

class PositiveInt { ... }
external impl PositiveInt as IntLike;
external impl PositiveInt as ComparableWith(PositiveInt);
// Allow `PositiveInt` to be compared with anything that
// implements `IntLike`, in either order.
external impl forall [T:! IntLike] PositiveInt as ComparableWith(T);
external impl forall [T:! IntLike] T as ComparableWith(PositiveInt);
```

Then it will favor selecting the implementation based on the type of the
left-hand operand:

```
var even: EvenInt = ...;
var positive: PositiveInt = ...;
// Uses `EvenInt as ComparableWith(T)` impl
if (even < positive) { ... }
// Uses `PositiveInt as ComparableWith(T)` impl
if (positive > even) { ... }
```

### `like` operator for implicit conversions

Because the type of the operands is directly used to select the implementation
to use, there are no automatic implicit conversions, unlike with function or
method calls. Given both a method and an interface implementation for
multiplying by a value of type `f64`:

```
class Meters {
  fn Scale[me: Self](s: f64) -> Self;
}
// "Implementation One"
external impl Meters as MultipliableWith(f64)
    where .Result = Meters {
  fn Multiply[me: Self](other: f64) -> Result {
    return me.Scale(other);
  }
}
```

the method will work with any argument that can be implicitly converted to `f64`
but the operator overload will only work with values that have the specific type
of `f64`:

```
var height: Meters = ...;
var scale: f32 = 1.25;
// ✅ Allowed: `scale` implicitly converted
//             from `f32` to `f64`.
var allowed: Meters = height.Scale(scale);
// ❌ Illegal: `Meters` doesn't implement
//             `MultipliableWith(f32)`.
var illegal: Meters = height * scale;
```

The workaround is to define a parameterized implementation that performs the
conversion. The implementation is for types that implement the
[`ImplicitAs` interface](/docs/design/expressions/implicit_conversions.md#extensibility).

```
// "Implementation Two"
external impl forall [T:! ImplicitAs(f64)]
    Meters as MultipliableWith(T) where .Result = Meters {
  fn Multiply[me: Self](other: T) -> Result {
    // Carbon will implicitly convert `other` from type
    // `T` to `f64` to perform this call.
    return me.(Meters.(MultipliableWith(f64).Multiply))(other);
  }
}
// ✅ Allowed: uses `Meters as MultipliableWith(T)` impl
//             with `T == f32` since `f32 is ImplicitAs(f64)`.
var now_allowed: Meters = height * scale;
```

Observe that the [prioritization rule](#prioritization-rule) will still prefer
the unparameterized impl when there is an exact match.

To reduce the boilerplate needed to support these implicit conversions when
defining operator overloads, Carbon has the `like` operator. This operator can
only be used in the type or type-of-type part of an `impl` declaration, as part
of a forward declaration or definition, in a place of a type.

```
// Notice `f64` has been replaced by `like f64`
// compared to "implementation one" above.
external impl Meters as MultipliableWith(like f64)
    where .Result = Meters {
  fn Multiply[me: Self](other: f64) -> Result {
    return me.Scale(other);
  }
}
```

This `impl` definition actually defines two implementations. The first is the
same as this definition with `like f64` replaced by `f64`, giving something
equivalent to "implementation one". The second implementation replaces the
`like f64` with a parameter that ranges over types that can be implicitly
converted to `f64`, equivalent to "implementation two".

In general, each `like` adds one additional impl. There is always the impl with
all of the `like` expressions replaced by their arguments with the definition
supplied in the source code. In addition, for each `like` expression, there is
an impl with it replaced by a new parameter. These additional impls will
delegate to the main impl, which will trigger implicit conversions according to
[Carbon's ordinary implicit conversion rules](/docs/design/expressions/implicit_conversions.md).
In this example, there are two uses of `like`, producing three implementations

```
external impl like Meters as MultipliableWith(like f64)
    where .Result = Meters {
  fn Multiply[me: Self](other: f64) -> Result {
    return me.Scale(other);
  }
}
```

is equivalent to "implementation one", "implementation two", and:

```
external impl forall [T:! ImplicitAs(Meters)]
    T as MultipliableWith(f64) where .Result = Meters {
  fn Multiply[me: Self](other: f64) -> Result {
    // Will implicitly convert `me` to `Meters` in order to
    // match the signature of this `Multiply` method.
    return me.(Meters.(MultipliableWith(f64).Multiply))(other);
  }
}
```

`like` may be used in forward declarations in a way analogous to impl
definitions.

```
external impl like Meters as MultipliableWith(like f64)
    where .Result = Meters;
}
```

is equivalent to:

```
// All `like`s removed. Same as the declaration part of
// "implementation one", without the body of the definition.
external impl Meters as MultipliableWith(f64)
    where .Result = Meters;

// First `like` replaced with a wildcard.
external impl forall [T:! ImplicitAs(Meters)]
    T as MultipliableWith(f64) where .Result = Meters;

// Second `like` replaced with a wildcard. Same as the
// declaration part of "implementation two", without the
// body of the definition.
external impl forall [T:! ImplicitAs(f64)]
    Meters as MultipliableWith(T) where .Result = Meters;
```

In addition, the generated impl definition for a `like` is implicitly injected
at the end of the (unique) source file in which the impl is first declared. That
is, it is injected in the API file if the impl is declared in an API file, and
in the sole impl file declaring the impl otherwise. This means an `impl`
declaration using `like` in an API file also makes the parameterized definition

If one `impl` declaration uses `like`, other declarations must use `like` in the
same way to match.

The `like` operator may be nested, as in:

```
external impl like Vector(like String) as Printable;
```

Which will generate implementations with declarations:

```
external impl Vector(String) as Printable;
external impl forall [T:! ImplicitAs(Vector(String))] T as Printable;
external impl forall [T:! ImplicitAs(String)] Vector(T) as Printable;
```

The generated implementations must be legal or the `like` is illegal. For
example, it must be legal to define those impls in this library by the
[orphan rule](#orphan-rule). In addition, the generated `impl` definitions must
only require implicit conversions that are guaranteed to exist. For example,
there existing an implicit conversion from `T` to `String` does not imply that
there is one from `Vector(T)` to `Vector(String)`, so the following use of
`like` is illegal:

```
// ❌ Illegal: Can't convert a value with type
//             `Vector(T:! ImplicitAs(String))`
//             to `Vector(String)` for `me`
//             parameter of `Printable.Print`.
external impl Vector(like String) as Printable;
```

Since the additional implementation definitions are generated eagerly, these
errors will be reported in the file with the first declaration.

The argument to `like` must either not mention any type parameters, or those
parameters must be able to be determined due to being repeated outside of the
`like` expression.

```
// ✅ Allowed: no parameters
external impl like Meters as Printable;

// ❌ Illegal: No other way to determine `T`
external impl forall [T:! IntLike] like T as Printable;

// ❌ Illegal: `T` being used in a `where` clause
//             is insufficient.
external impl forall [T:! IntLike] like T
    as MultipliableWith(i64) where .Result = T;

// ❌ Illegal: `like` can't be used in a `where`
//             clause.
external impl Meters as MultipliableWith(f64)
    where .Result = like Meters;

// ✅ Allowed: `T` can be determined by another
//             part of the query.
external impl forall [T:! IntLike] like T
    as MultipliableWith(T) where .Result = T;
external impl forall [T:! IntLike] T
    as MultipliableWith(like T) where .Result = T;

// ✅ Allowed: Only one `like` used at a time, so this
//             is equivalent to the above two examples.
external impl forall [T:! IntLike] like T
    as MultipliableWith(like T) where .Result = T;
```

## Parameterized types

Types may have generic parameters. Those parameters may be used to specify types
in the declarations of its members, such as data fields, member functions, and
even interfaces being implemented. For example, a container type might be
parameterized by the type of its elements:

```
class HashMap(
    KeyType:! Hashable & EqualityComparable & Movable,
    ValueType:! Movable) {
  // `Self` is `HashMap(KeyType, ValueType)`.

  // Parameters may be used in function signatures.
  fn Insert[addr me: Self*](k: KeyType, v: ValueType);

  // Parameters may be used in field types.
  private var buckets: Vector((KeyType, ValueType));

  // Parameters may be used in interfaces implemented.
  impl as Container where .ElementType = (KeyType, ValueType);
  impl as ComparableWith(HashMap(KeyType, ValueType));
}
```

Note that, unlike functions, every parameter to a type must either be generic or
template, using `:!` or `template...:!`, not dynamic, with a plain `:`.

Two types are the same if they have the same name and the same arguments.
Carbon's [manual type equality](#manual-type-equality) approach means that the
compiler may not always be able to tell when two type expressions are equal
without help from the user, in the form of
[`observe` declarations](#observe-declarations). This means Carbon will not in
general be able to determine when types are unequal.

Unlike an [interface's parameters](#parameterized-interfaces), a type's
parameters may be [deduced](terminology.md#deduced-parameter), as in:

```
fn ContainsKey[KeyType:! Movable, ValueType:! Movable]
    (haystack: HashMap(KeyType, ValueType), needle: KeyType)
    -> bool { ... }
fn MyMapContains(s: String) {
  var map: HashMap(String, i32) = (("foo", 3), ("bar", 5));
  // ✅ Deduces `KeyType` = `String` from the types of both arguments.
  // Deduces `ValueType` = `i32` from the type of the first argument.
  return ContainsKey(map, s);
}
```

Note that restrictions on the type's parameters from the type's declaration can
be [implied constraints](#implied-constraints) on the function's parameters.

### Specialization

[Specialization](terminology.md#generic-specialization) is used to improve
performance in specific cases when a general strategy would be inefficient. For
example, you might use
[binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm) for
containers that support random access and keep their contents in sorted order
but [linear search](https://en.wikipedia.org/wiki/Linear_search) in other cases.
Types, like functions, may not be specialized directly in Carbon. This effect
can be achieved, however, through delegation.

For example, imagine we have a parameterized class `Optional(T)` that has a
default storage strategy that works for all `T`, but for some types we have a
more efficient approach. For pointers we can use a
[null value](https://en.wikipedia.org/wiki/Null_pointer) to represent "no
pointer", and for booleans we can support `True`, `False`, and `None` in a
single byte. Clients of the optional library may want to add additional
specializations for their own types. We make an interface that represents "the
storage of `Optional(T)` for type `T`," written here as `OptionalStorage`:

```
interface OptionalStorage {
  let Storage:! Type;
  fn MakeNone() -> Storage;
  fn Make(x: Self) -> Storage;
  fn IsNone(x: Storage) -> bool;
  fn Unwrap(x: Storage) -> Self;
}
```

The default implementation of this interface is provided by a
[blanket implementation](#blanket-impls):

```
// Default blanket implementation
impl forall [T:! Movable] T as OptionalStorage
    where .Storage = (bool, T) {
  ...
}
```

This implementation can then be
[specialized](#lookup-resolution-and-specialization) for more specific type
patterns:

```
// Specialization for pointers, using nullptr == None
final external impl forall [T:! Type] T* as OptionalStorage
    where .Storage = Array(Byte, sizeof(T*)) {
  ...
}
// Specialization for type `bool`.
final external impl bool as OptionalStorage
    where .Storage = Byte {
  ...
}
```

Further, libraries can implement `OptionalStorage` for their own types, assuming
the interface is not marked `private`. Then the implementation of `Optional(T)`
can delegate to `OptionalStorage` for anything that can vary with `T`:

```
class Optional(T:! Movable) {
  fn None() -> Self {
    return {.storage = T.(OptionalStorage.MakeNone)()};
  }
  fn Some(x: T) -> Self {
    return {.storage = T.(OptionalStorage.Make)(x)};
  }
  ...
  private var storage: T.(OptionalStorage.Storage);
}
```

Note that the constraint on `T` is just `Movable`, not
`Movable & OptionalStorage`, since the `Movable` requirement is
[sufficient to guarantee](#lookup-resolution-and-specialization) that some
implementation of `OptionalStorage` exists for `T`. Carbon does not require
callers of `Optional`, even generic callers, to specify that the argument type
implements `OptionalStorage`:

```
// ✅ Allowed: `T` just needs to be `Movable` to form `Optional(T)`.
//             A `T:! OptionalStorage` constraint is not required.
fn First[T:! Movable & Eq](v: Vector(T)) -> Optional(T);
```

Adding `OptionalStorage` to the constraints on the parameter to `Optional` would
obscure what types can be used as arguments. `OptionalStorage` is an
implementation detail of `Optional` and need not appear in its public API.

In this example, a `let` is used to avoid repeating `OptionalStorage` in the
definition of `Optional`, since it has no name conflicts with the members of
`Movable`:

```
class Optional(T:! Movable) {
  private let U:! Movable & OptionalStorage = T;
  fn None() -> Self {
    return {.storage = U.MakeNone()};
  }
  fn Some(x: T) -> Self {
    return {.storage = u.Make(x)};
  }
  ...
  private var storage: U.Storage;
}
```

## Future work

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

### Evolution

There are a collection of use cases for making different changes to interfaces
that are already in use. These should be addressed either by describing how they
can be accomplished with existing generics features, or by adding features.

In addition, evolution from (C++ or Carbon) templates to generics needs to be
supported and made safe.

### Testing

The idea is that you would write tests alongside an interface that validate the
expected behavior of any type implementing that interface.

### Impls with state

A feature we might consider where an `impl` itself can have state.

### Generic associated types and higher-ranked types

This would be some way to express the requirement that there is a way to go from
a type to an implementation of an interface parameterized by that type.

#### Generic associated types

Generic associated types are about when this is a requirement of an interface.
These are also called "associated type constructors."

Rust has
[stabilized this feature](https://github.com/rust-lang/rust/pull/96709).

#### Higher-ranked types

Higher-ranked types are used to represent this requirement in a function
signature. They can be
[emulated using generic associated types](https://smallcultfollowing.com/babysteps//blog/2016/11/03/associated-type-constructors-part-2-family-traits/).

### Field requirements

We might want to allow interfaces to express the requirement that any
implementing type has a particular field. This would be to match the
expressivity of inheritance, which can express "all subtypes start with this
list of fields."

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
-   [#931: Generic impls access (details 4)](https://github.com/carbon-language/carbon-lang/pull/931)
-   [#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920)
-   [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
-   [#983: Generic details 7: final impls](https://github.com/carbon-language/carbon-lang/pull/983)
-   [#990: Generics details 8: interface default and final members](https://github.com/carbon-language/carbon-lang/pull/990)
-   [#1013: Generics: Set associated constants using `where` constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
-   [#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
-   [#1088: Generic details 10: interface-implemented requirements](https://github.com/carbon-language/carbon-lang/pull/1088)
-   [#1144: Generic details 11: operator overloading](https://github.com/carbon-language/carbon-lang/pull/1144)
-   [#1146: Generic details 12: parameterized types](https://github.com/carbon-language/carbon-lang/pull/1146)
-   [#1327: Generics: `impl forall`](https://github.com/carbon-language/carbon-lang/pull/1327)
-   [#2107: Clarify rules around `Self` and `.Self`](https://github.com/carbon-language/carbon-lang/pull/2107)
