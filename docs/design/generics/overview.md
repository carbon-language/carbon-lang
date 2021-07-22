# Carbon generics overview

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document is a high-level description of Carbon's generics design, with
pointers to other design documents that dive deeper into individual topics.

<!-- toc -->

## Table of contents

-   [Goals](#goals)
-   [Summary](#summary)
-   [What are generics?](#what-are-generics)
    -   [Interfaces](#interfaces)
        -   [Defining interfaces](#defining-interfaces)
        -   [Contrast with templates](#contrast-with-templates)
    -   [Implementing interfaces](#implementing-interfaces)
        -   [Qualified and unqualified access](#qualified-and-unqualified-access)
    -   [Type-of-types](#type-of-types)
    -   [Generic functions](#generic-functions)
        -   [Deduced parameters](#deduced-parameters)
        -   [Generic type parameters](#generic-type-parameters)
    -   [Requiring or extending another interface](#requiring-or-extending-another-interface)
    -   [Combining interfaces](#combining-interfaces)
        -   [Structural interfaces](#structural-interfaces)
        -   [Type erasure](#type-erasure)
-   [Future work](#future-work)

<!-- tocstop -->

## Goals

The goal of Carbon generics is to provide an alternative to Carbon (or C++)
templates. Generics in this form should provide many advantages, including:

-   Function calls and bodies are checked independently against the function
    signatures.
-   Clearer and earlier error messages.
-   Fast builds, particularly development builds.
-   Support for both static and dynamic dispatch.

For more detail, see [the detailed discussion of generics goals](goals.md) and
[generics terminology](terminology.md).

## Summary

Summary of how Carbon generics work:

-   _Generics_ are parameterized functions and types that can apply generally.
    They are used to avoid writing specialized, near-duplicate code for similar
    situations.
-   Generics are written using _interfaces_ which have a name and describe
    methods, functions, and other items for types to implement.
-   Types must explicitly _implement_ interfaces to indicate that they support
    its functionality. A given type may implement an interface at most once.
-   Implementations may be part of the type's definition, in which case you can
    directly call the interface's methods on those types. Or, they may be
    external, in which case the implementation is allowed to be defined in the
    library defining the interface.
-   Interfaces are used as the type of a generic type parameter, acting as a
    _type-of-type_. Type-of-types in general specify the capabilities and
    requirements of the type. Types define specific implementations of those
    capabilities. Inside such a generic function, the API of the type is
    [erased](terminology.md#type-erasure), except for the names defined in the
    type-of-type.
-   _Deduced parameters_ are parameters whose values are determined by the
    values and (most commonly) the types of the explicit arguments. Generic type
    parameters are typically deduced.
-   A function with a generic type parameter can have the same function body as
    an unparameterized one. Functions can freely mix generic, template, and
    regular parameters.
-   Interfaces can require other interfaces be implemented, or
    [extend](terminology.md#extending-an-interface) them.
-   The `&` operation on type-of-types allows you conveniently combine
    interfaces. It gives you all the names that don't conflict.
-   You may also declare a new type-of-type directly using
    ["structural interfaces"](terminology.md#structural-interfaces). Structural
    interfaces can express requirements that multiple interfaces be implemented,
    and give you control over how name conflicts are handled.
-   Alternatively, you may resolve name conflicts by using a qualified syntax to
    directly call a function from a specific interface.

## What are generics?

Generics are a mechanism for writing parameterized code that applies generally
instead of making near-duplicates for very similar situations, much like C++
templates. For example, instead of having one function per type-you-can-sort:

```
fn SortInt32Vector(a: Vector(Int32)*) { ... }
fn SortStringVector(a: Vector(String)*) { ... }
...
```

You might have one generic function that could sort any array with comparable
elements:

```
fn SortVector(T:$ Comparable, a: Vector(T)*) { ... }
```

The syntax above adds a `$` to indicate that the parameter named `T` is generic.

Given an `Int32` vector `iv`, `SortVector(Int32, &iv)` is equivalent to
`SortInt32Vector(&iv)`. Similarly for a `String` vector `sv`,
`SortVector(String, &sv)` is equivalent to `SortStringVector(&sv)`. Thus, we can
sort any vector containing comparable elements using this single `SortVector`
function.

This ability to generalize makes `SortVector` a _generic_.

**NOTE:** The `:$` syntax is a placeholder. The syntax is being decided in
[question-for-leads issue #565](https://github.com/carbon-language/carbon-lang/issues/565).

### Interfaces

The `SortVector` function requires a definition of `Comparable`, with the goal
that the compiler can:

-   completely type check a generic definition without information from where
    it's called.
-   completely type check a call to a generic with information only from the
    function's signature, and not from its body.

In this example, then, `Comparable` is an _interface_.

Interfaces describe all the requirements needed for the type `T`. Given that the
compiler knows `T` satisfies those requirements, it can type check the body of
the `SortVector` function. This includes checking that the `Comparable`
requirement covers all of the uses of `T` inside the function.

Later, when the compiler comes across a call to `SortVector`, it can type check
against the requirements expressed in the function's signature. Using only the
types at the call site, the compiler can check that the member elements of the
passed-in array satisfy the function's requirements. There is no need to look at
the body of the `SortVector` function, since we separately checked that those
requirements were sufficient.

#### Defining interfaces

Interfaces, then, have a name and describe methods, functions, and other items
for types to implement.

Example:

```
interface Comparable {
  // `Less` is an associated method.
  fn Less[me: Self](that: Self) -> Bool;
}
```

**Note:** The method syntax was decided in
[question-for-leads issue #494](https://github.com/carbon-language/carbon-lang/issues/494).

Interfaces describe functionality, but not data; no variables may be declared in
an interface.

#### Contrast with templates

Contrast these generics with a C++ template, where the compiler may be able to
do some checking given a function definition, but more checking of the
definition is required after seeing the call sites once all the
[instantiations](terminology.md#instantiation) are known.

Note: The doc on [Generics terminology](terminology.md) goes into more detail
about the
[differences between generics and templates](terminology.md#generic-versus-template-parameters).

### Implementing interfaces

Interfaces themselves only describe functionality by way of method descriptions.
A type needs to _implement_ an interface to indicate that it supports its
functionality. A given type may implement an interface at most once.

Consider this interface:

```
interface Printable {
  fn Print[me: Self]();
}
```

The `interface` keyword is used to define a
[_nominal interface_](terminology.md#nominal-interfaces). That means that types
need to explicitly implement them, using an `impl` block, such as here:

```
struct Song {
  // ...

  // Implementing `Printable` for `Song` inside the definition of `Song`
  // means all names of `Printable`, such as `F`, are included as a part
  // of the `Song` API.
  impl as Printable {
    // Could use `Self` in place of `Song` here.
    fn Print[me: Song]() { ... }
  }
}

// Implement `Comparable` for `Song` without changing the API of `Song`
// using an `external impl` declaration. This may be defined in either
// the library defining `Song` or `Comparable`.
external impl Song as Comparable {
  // Could use either `Self` or `Song` here.
  fn Less[me: Self](that: Self) -> Bool { ... }
}
```

**Note:** The interface implementation syntax was decided in
[question-for-leads issue #575](https://github.com/carbon-language/carbon-lang/issues/575).
TODO: move these syntax issues to details and link.

Implementations may be defined within the struct definition itself or
externally. External implementations may be defined in the library defining the
interface.

#### Qualified and unqualified access

The methods of an interface implemented within the struct definition may be
called with the unqualified syntax. All methods of implemented interfaces may be
called with the qualified syntax, whether they are defined internally or
externally.

```
var song: Song;
// `song.Print()` is allowed, unlike `song.Play()`.
song.Print();
// `Less` is defined in `Comparable`, which is implemented
// externally for `Song`
song.(Comparable.Less)(song);
// Can also call `Print` using the qualified syntax:
song.(Printable.Print)();
```

### Type-of-types

To type check a function, the compiler needs to be able to verify that uses of a
value match the capabilities of the value's type. In `SortVector`, the parameter
`T` is a type, but that type is a generic parameter. That means that the
specific type value assigned to `T` is not known when type checking the
`SortVector` function. Instead it is the constraints on `T` that let the
compiler know what operations may be performed on values of type `T`. Those
constraints are represented by the type of `T`, a
[**_type-of-type_**](terminology.md#type-constraints).

In general, a type-of-type describes the capabilities of a type, while a type
defines specific implementations of those capabilities.

An interface, like `Comparable`, may be used as a type-of-type. In that case,
the constraint on the type is that it must implement the interface `Comparable`.
A type-of-type also defines a set of names and a mapping to corresponding
qualified names. You may combine interfaces into new type-of-types using
[the `&` operator](#combining-interfaces) or
[structural interfaces](#structural-interfaces).

### Generic functions

We want to be able to call generic functions just like ordinary functions, and
write generic function bodies like ordinary functions. There are only a few
differences, like that you can't take the address of generic functions.

#### Deduced parameters

This `SortVector` function is explicitly providing type information that is
already included in the type of the second argument. To eliminate the argument
at the call site, use a _deduced parameter_.

```
fn SortVectorDeduced[T:$ Comparable](a: Vector(T)*) { ... }
```

The `T` parameter is defined in square brackets before the explicit parameter
list in parenthesis to indicate it should be deduced. This means you may call
the function without the type argument, just like the ordinary functions
`SortInt32Vector` or `SortStringVector`:

```
SortVectorDeduced(&anIntVector);
// or
SortVectorDeduced(&aStringVector);
```

and the compiler deduces that the `T` argument should be set to `Int32` or
`String` from the type of the argument.

Deduced arguments are always determined from the call and its explicit
arguments. There is no syntax for specifying deduced arguments directly at the
call site.

```
// ERROR: can't determine `U` from explicit parameters
fn Illegal[T:$ Type, U:$ Type](x: T) -> U { ... }
```

#### Generic type parameters

A function with a generic type parameter can have the same function body as an
unparameterized one.

```
fn PrintIt[T:$ Printable](p: T*) {
  p->Print();
}

fn PrintIt(p: Song*) {
  p->Print();
}
```

Inside the function body, you can treat the generic type parameter just like any
other type. There is no need to refer to or access generic parameters
differently because they are defined as generic, as long as you only refer to
the names defined by [type-of-type](#type-of-types) for the type parameter.

You may also refer to any of the methods of interfaces required by the
type-of-type using the [qualified syntax](#qualified-and-unqualified-access), as
shown in the following sections.

A function can have a mix of generic, template, and regular parameters.
Likewise, it's allowed to pass a template or generic value to a generic or
regular parameter. _However, passing a generic value to a template parameter is
future work._

### Requiring or extending another interface

Interfaces can require other interfaces be implemented:

```
interface Equatable {
  fn IsEqual[me: Self](that: Self) -> Bool;
}

// `Iterable` requires that `Equatable` is implemented.
interface Iterable {
  impl as Equatable;
  fn Advance[addr me: Self*]();
}
```

The `extends` keyword is used to [extend](terminology.md#extending-an-interface)
another interface. If interface `Child` extends interface `Parent`, `Parent`'s
interface is both required and all its methods are included in `Child`'s
interface.

```
// `Hashable` extends `Equatable`.
interface Hashable {
  extends Equatable;
  fn Hash[me: Self]() -> UInt64;
}
// `Hashable` is equivalent to:
interface Hashable {
  impl as Equatable;
  alias IsEqual = Equatable.IsEqual;
  fn Hash[me: Self]() -> UInt64;
}
```

A type may implement the parent interface implicitly by implementing all the
methods in the child implementation.

```
struct Key {
  // ...
  impl as Hashable {
    fn IsEqual[me: Key](that: Key) -> Bool { ... }
    fn Hash[me: Key]() -> UInt64 { ... }
  }
  // No need to separately implement `Equatable`.
}
var k: Key = ...;
k.Hash();
k.IsEqual(k);
```

### Combining interfaces

The `&` operation on type-of-types allows you conveniently combine interfaces.
It gives you all the names that don't conflict.

```
interface Renderable {
  fn GetCenter[me: Self]() -> (Int, Int);
  // Draw the object to the screen
  fn Draw[me: Self]();
}
interface EndOfGame {
  fn SetWinner[addr me: Self*](player: Int);
  // Indicate the game was a draw
  fn Draw[addr me: Self*]();
}

fn F[T:$ Renderable & EndOfGame](game_state: T*) -> (Int, Int) {
  game_state->SetWinner(1);
  return game_state->Center();
}
```

Names with conflicts can be accessed using the
[qualified syntax](#qualified-and-unqualified-access).

```
fn BothDraws[T:$ Renderable & EndOfGame](game_state: T*) {
  game_state->(Renderable.Draw)();
  game_state->(GameState.Draw)();
}
```

#### Structural interfaces

You may also declare a new type-of-type directly using
["structural interfaces"](terminology.md#structural-interfaces). Structural
interfaces can express requirements that multiple interfaces be implemented, and
give you control over how name conflicts are handled. Structural interfaces have
other applications and capabilities not covered here.

```
structural interface Combined {
  impl as Renderable;
  impl as EndOfGame;
  alias Draw_Renderable = Renderable.Draw;
  alias Draw_EndOfGame = EndOfGame.Draw;
  alias SetWinner = EndOfGame.SetWinner;
}

fn CallItAll[T:$ Combined](game_state: T*, int winner) {
  if (winner > 0) {
    game_state->SetWinner(winner);
  } else {
    game_state->Draw_EndOfGame();
  }
  game_state->Draw_Renderable();
  // Can still use qualified syntax for names
  // not defined in the structural interface
  return game_state->(Renderable.Center)();
}
```

#### Type erasure

Inside a generic function, the API of a type argument is
[erased](terminology.md#type-erasure) except for the names defined in the
type-of-type.

For example: If there were a class `CDCover` defined this way:

```
struct CDCover  {
  impl as Printable {
    ...
  }
}
```

it can be passed to this `PrintIt` function:

```
fn PrintIt[T:$ Printable](p: T*) {
  p->Print();
}
```

At that point, two erasures occur:

-   All of `CDCover`'s API _except_ `Printable` is erased during the cast from
    `CDCover` to `Printable`, which is the [facet](terminology.md#facets) type
    `CDCover as Printable`.
-   When you call `PrintIt`, the type connection to `CDCover` is lost. Outside
    of `PrintIt` you can cast a `CDCover as Printable` value back to `CDCover`.
    Inside of `PrintIt`, you can't cast `p` or `T` back to `CDCover`.

## Future work

-   Be able to have non-type generic parameters like the `UInt` size of an array
    or tuple.
-   A "newtype" mechanism called "adapting types" may be provided to create new
    types that are compatible with existing types but with different interface
    implementations. This could be used to add or replace implementations, or
    define implementations for reuse.
-   Associated types and interface parameters will be provided to allow function
    signatures to vary with the implementing type. The biggest difference
    between these is that associated types ("output types") may be deduced from
    a type, and types can implement the same interface multiple times with
    different interface parameters ("input types").
-   Other kinds of constraints will be finalized.
-   Implementations can be parameterized to apply to multiple types. These
    implementations would be restricted to various conditions are true for the
    parameters. When there are two implementations that can apply, there is a
    specialization rule that picks the more specific one.
-   Support functions should have a way to accept types that types that vary at
    runtime.
-   You should have the ability to mark items as `upcoming` or `deprecated` to
    support evolution.
-   Types should be able to define overloads for operators by implementing
    standard interfaces.
-   There should be a way to provide default implementations of methods in
    interfaces and other ways to reuse code across implementations.
-   There should be a way to define generic associated and higher-ranked/kinded
    types.
