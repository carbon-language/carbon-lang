# Generics: Overview

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
        -   [Accessing members of interfaces](#accessing-members-of-interfaces)
    -   [Type-of-types](#type-of-types)
    -   [Generic functions](#generic-functions)
        -   [Deduced parameters](#deduced-parameters)
        -   [Generic type parameters](#generic-type-parameters)
    -   [Requiring or extending another interface](#requiring-or-extending-another-interface)
    -   [Combining interfaces](#combining-interfaces)
        -   [Named constraints](#named-constraints)
        -   [Type erasure](#type-erasure)
    -   [Adapting types](#adapting-types)
    -   [Interface input and output types](#interface-input-and-output-types)
        -   [Associated types](#associated-types)
        -   [Parameterized interfaces](#parameterized-interfaces)
    -   [Constraints](#constraints)
    -   [Parameterized impls](#parameterized-impls)
    -   [Operator overloading](#operator-overloading)
-   [Future work](#future-work)
-   [References](#references)

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
    methods, functions, and other entities for types to implement.
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
-   Interfaces can require other interfaces be implemented.
-   Interfaces can [extend](terminology.md#extending-an-interface) required
    interfaces.
-   The `&` operation on type-of-types allows you conveniently combine
    interfaces. It gives you all the names that don't conflict.
-   You may also declare a new type-of-type directly using
    ["named constraints"](terminology.md#named-constraints). Named constraints
    can express requirements that multiple interfaces be implemented, and give
    you control over how name conflicts are handled.
-   Alternatively, you may resolve name conflicts by using a qualified member
    access expression to directly call a function from a specific interface
    using a qualified name.

## What are generics?

Generics are a mechanism for writing parameterized code that applies generally
instead of making near-duplicates for very similar situations, much like C++
templates. For example, instead of having one function per type-you-can-sort:

```
fn SortInt32Vector(a: Vector(i32)*) { ... }
fn SortStringVector(a: Vector(String)*) { ... }
...
```

You might have one generic function that could sort any array with comparable
elements:

```
fn SortVector(T:! Comparable, a: Vector(T)*) { ... }
```

The syntax above adds a `!` to indicate that the parameter named `T` is generic
and the caller will have to provide a value known at compile time.

Given an `i32` vector `iv`, `SortVector(i32, &iv)` is equivalent to
`SortInt32Vector(&iv)`. Similarly for a `String` vector `sv`,
`SortVector(String, &sv)` is equivalent to `SortStringVector(&sv)`. Thus, we can
sort any vector containing comparable elements using this single `SortVector`
function.

This ability to generalize makes `SortVector` a _generic_.

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

Interfaces, then, have a name and describe methods, functions, and other
entities for types to implement.

Example:

```
interface Comparable {
  // `Less` is an associated method.
  fn Less[self: Self](rhs: Self) -> bool;
}
```

Functions and methods may be given a default implementation by prefixing the
declaration with `default` and putting the function body in curly braces
`{`...`}` in place of the terminating `;` of the function declaration. To
prevent that implementation from being overridden, use `final` instead of
`default`.

Interfaces describe functionality, but not data; no variables may be declared in
an interface.

#### Contrast with templates

Contrast these generics with a C++ template, where the compiler may be able to
do some checking given a function definition, but more checking of the
definition is required after seeing the call sites once all the
[instantiations](terminology.md#instantiation) are known.

Note: [Generics terminology](terminology.md) goes into more detail about the
[differences between generics and templates](terminology.md#generic-versus-template-parameters).

### Implementing interfaces

Interfaces themselves only describe functionality by way of method descriptions.
A type needs to _implement_ an interface to indicate that it supports its
functionality. A given type may implement an interface at most once.

Consider this interface:

```
interface Printable {
  fn Print[self: Self]();
}
```

The `interface` keyword is used to define a
[_nominal interface_](terminology.md#nominal-interfaces). That means that types
need to explicitly implement them, using an `impl` block, such as here:

```
class Song {
  // ...

  // Implementing `Printable` for `Song` inside the definition of `Song`
  // without the keyword `external` means all names of `Printable`, such
  // as `F`, are included as a part of the `Song` API.
  impl as Printable {
    // Could use `Self` in place of `Song` here.
    fn Print[self: Song]() { ... }
  }
}

// Implement `Comparable` for `Song` without changing the API of `Song`
// using an `external impl` declaration. This may be defined in either
// the library defining `Song` or `Comparable`.
external impl Song as Comparable {
  // Could use either `Self` or `Song` here.
  fn Less[self: Self](rhs: Self) -> bool { ... }
}
```

Implementations may be defined within the class definition itself or
out-of-line. Implementations may optionally start with the `external` keyword to
say the members of the interface are not members of the class. Out-of-line
implementations must be external. External implementations may be defined in the
library defining either the class or the interface.

#### Accessing members of interfaces

The methods of an interface implemented internally within the class definition
may be called with the
[simple member access syntax](terminology.md#simple-member-access). Methods of
all implemented interfaces may be called with a
[qualified member access expression](terminology.md#qualified-member-access-expression),
whether they are defined internally or externally.

```
var song: Song;
// `song.Print()` is allowed, unlike `song.Play()`.
song.Print();
// `Less` is defined in `Comparable`, which is
// implemented externally for `Song`
song.(Comparable.Less)(song);
// Can also call `Print` using a qualified member
// access expression, using the compound member access
// syntax with the qualified name `Printable.Print`:
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
[**_type-of-type_**](terminology.md#type-of-type).

In general, a type-of-type describes the capabilities of a type, while a type
defines specific implementations of those capabilities. An interface, like
`Comparable`, may be used as a type-of-type. In that case, the constraint on the
type is that it must implement the interface `Comparable`.

A type-of-type also defines a set of names and a mapping to corresponding
qualified names. Those names are used for
[simple member lookup](terminology.md#simple-member-access) in scopes where the
value of the type is not known, such as when the type is a generic parameter.

You may combine interfaces into new type-of-types using
[the `&` operator](#combining-interfaces) or
[named constraints](#named-constraints).

### Generic functions

We want to be able to call generic functions just like ordinary functions, and
write generic function bodies like ordinary functions. There are only a few
differences, like that you can't take the address of generic functions.

#### Deduced parameters

This `SortVector` function is explicitly providing type information that is
already included in the type of the second argument. To eliminate the argument
at the call site, use a _deduced parameter_.

```
fn SortVectorDeduced[T:! Comparable](a: Vector(T)*) { ... }
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

and the compiler deduces that the `T` argument should be set to `i32` or
`String` from the type of the argument.

Deduced arguments are always determined from the call and its explicit
arguments. There is no syntax for specifying deduced arguments directly at the
call site.

```
// ERROR: can't determine `U` from explicit parameters
fn Illegal[T:! type, U:! type](x: T) -> U { ... }
```

#### Generic type parameters

A function with a generic type parameter can have the same function body as an
unparameterized one.

```
fn PrintIt[T:! Printable](p: T*) {
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
type-of-type using a
[qualified member access expression](#accessing-members-of-interfaces), as shown
in the following sections.

A function can have a mix of generic, template, and regular parameters.
Likewise, it's allowed to pass a template or generic value to a generic or
regular parameter. _However, passing a generic value to a template parameter is
future work._

### Requiring or extending another interface

Interfaces can require other interfaces be implemented:

```
interface Equatable {
  fn IsEqual[self: Self](rhs: Self) -> bool;
}

// `Iterable` requires that `Equatable` is implemented.
interface Iterable {
  impl as Equatable;
  fn Advance[addr self: Self*]();
}
```

The `extends` keyword is used to [extend](terminology.md#extending-an-interface)
another interface. If interface `Derived` extends interface `Base`, `Base`'s
interface is both required and all its methods are included in `Derived`'s
interface.

```
// `Hashable` extends `Equatable`.
interface Hashable {
  extends Equatable;
  fn Hash[self: Self]() -> u64;
}
// `Hashable` is equivalent to:
interface Hashable {
  impl as Equatable;
  alias IsEqual = Equatable.IsEqual;
  fn Hash[self: Self]() -> u64;
}
```

A type may implement the base interface implicitly by implementing all the
methods in the implementation of the derived interface.

```
class Key {
  // ...
  impl as Hashable {
    fn IsEqual[self: Key](rhs: Key) -> bool { ... }
    fn Hash[self: Key]() -> u64 { ... }
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
  fn GetCenter[self: Self]() -> (i32, i32);
  // Draw the object to the screen
  fn Draw[self: Self]();
}
interface EndOfGame {
  fn SetWinner[addr self: Self*](player: i32);
  // Indicate the game was a draw
  fn Draw[addr self: Self*]();
}

fn F[T:! Renderable & EndOfGame](game_state: T*) -> (i32, i32) {
  game_state->SetWinner(1);
  return game_state->Center();
}
```

Names with conflicts can be accessed using a
[qualified member access expression](#accessing-members-of-interfaces).

```
fn BothDraws[T:! Renderable & EndOfGame](game_state: T*) {
  game_state->(Renderable.Draw)();
  game_state->(GameState.Draw)();
}
```

#### Named constraints

You may also declare a new type-of-type directly using
["named constraints"](terminology.md#named-constraints). Named constraints can
express requirements that multiple interfaces be implemented, and give you
control over how name conflicts are handled. Named constraints have other
applications and capabilities not covered here.

```
constraint Combined {
  impl as Renderable;
  impl as EndOfGame;
  alias Draw_Renderable = Renderable.Draw;
  alias Draw_EndOfGame = EndOfGame.Draw;
  alias SetWinner = EndOfGame.SetWinner;
}

fn CallItAll[T:! Combined](game_state: T*, int winner) {
  if (winner > 0) {
    game_state->SetWinner(winner);
  } else {
    game_state->Draw_EndOfGame();
  }
  game_state->Draw_Renderable();
  // Can still use a qualified member access expression
  // for names not defined in the named constraint.
  return game_state->(Renderable.Center)();
}
```

#### Type erasure

Inside a generic function, the API of a type argument is
[erased](terminology.md#type-erasure) except for the names defined in the
type-of-type. An equivalent model is to say an
[archetype](terminology.md#archetype) is used for type checking and name lookup
when the actual type is not known in that scope. The archetype has members
dictated by the type-of-type.

For example: If there were a class `CDCover` defined this way:

```
class CDCover  {
  impl as Printable {
    ...
  }
}
```

it can be passed to this `PrintIt` function:

```
fn PrintIt[T:! Printable](p: T*) {
  p->Print();
}
```

Inside `PrintIt`, `T` is an archetype with the API of `Printable`. A call to
`PrintIt` with a value of type `CDCover` erases everything except the members or
`Printable`. This includes the type connection to `CDCover`, so it is illegal to
cast from `T` to `CDCover`.

### Adapting types

Carbon has a mechanism called [adapting types](terminology.md#adapting-a-type)
to create new types that are [compatible](terminology.md#compatible-types) with
existing types but with different interface implementations. This could be used
to add or replace implementations, or define implementations for reuse.

In this example, we have multiple ways of sorting a collection of `Song` values.

```
class Song { ... }

adapter SongByArtist extends Song {
  impl as Comparable { ... }
}

adapter SongByTitle extends Song {
  impl as Comparable { ... }
}
```

Values of type `Song` may be cast to `SongByArtist` or `SongByTitle` to get a
specific sort order.

### Interface input and output types

[Associated types and interface parameters](terminology.md#interface-type-parameters-and-associated-types)
allow function signatures to vary with the implementing type. The biggest
difference between these is that associated types ("output types") may be
deduced from a type, and types can implement the same interface multiple times
with different interface parameters ("input types").

#### Associated types

Expect types that vary in an interface to be associated types by default. Since
associated types may be deduced, they are more convenient to use. Imagine a
`Stack` interface. Different types implementing `Stack` will have different
element types:

```
interface Stack {
  let ElementType:! Movable;
  fn Push[addr self: Self*](value: ElementType);
  fn Pop[addr self: Self*]() -> ElementType;
  fn IsEmpty[addr self: Self*]() -> bool;
}
```

`ElementType` is an associated type of the interface `Stack`. Types that
implement `Stack` give `ElementType` a specific value of some type implementing
`Movable`. Functions that accept a type implementing `Stack` can deduce the
`ElementType` from the stack type.

```
// ✅ This is allowed, since the type of the stack will determine
// `ElementType`.
fn PeekAtTopOfStack[StackType:! Stack](s: StackType*)
    -> StackType.ElementType;
```

#### Parameterized interfaces

Parameterized interfaces are commonly associated with overloaded operators.
Imagine an interface for determining if two values are equivalent that allows
those types to be different. An element in a hash map might have type
`Pair(String, i64)` that implements both `Equatable(String)` and
`Equatable(Pair(String, i64))`.

```
interface Equatable(T:! type) {
  fn IsEqual[self: Self](compare_to: T) -> bool;
}
```

`T` is a parameter to interface `Equatable`. A type can implement `Equatable`
multiple times as long as each time it is with a different value of the `T`
parameter. Functions may accept types implementing `Equatable(i32)` or
`Equatable(f32)`. Functions can't accept types implementing `Equatable(T)` in
general, unless some other parameter determines `T`.

```
// ✅ This is allowed, since the value of `T` is determined by the
// `v` parameter.
fn FindInVector[T:! type, U:! Equatable(T)](v: Vector(T), needle: U)
    -> Optional(i32);

// ❌ This is forbidden. Since `U` could implement `Equatable`
// multiple times, there is no way to determine the value for `T`.
// Contrast with `PeekAtTopOfStack` in the associated type example.
fn CompileError[T:! type, U:! Equatable(T)](x: U) -> T;
```

### Constraints

Type-of-types can be further constrained using a `where` clause:

```
fn FindFirstPrime[T:! Container where .Element == i32]
    (c: T, i: i32) -> Optional(i32) {
  // The elements of `c` have type `T.Element`, which is `i32`.
  ...
}

fn PrintContainer[T:! Container where .Element is Printable](c: T) {
  // The type of the elements of `c` is not known, but we do know
  // that type satisfies the `Printable` interface.
  ...
}
```

Constraints limit the types that the generic function can operate on, but
increase the knowledge that may be used in the body of the function to operate
on values of those types.

Constraints are also used when implementing an interface to specify the values
of associated types (and other associated constants).

```
class Vector(T:! Movable) {
  impl as Stack where .ElementType = T { ... }
}
```

### Parameterized impls

Implementations can be parameterized to apply to multiple types. Those
parameters can have constraints to restrict when the implementation applies.
When multiple implementations apply, there is a rule to pick which one is
considered the most specific:

-   All type parameters in each `impl` declaration are replaced with question
    marks `?`. This is called the type structure of the `impl` declaration.
-   Given two type structures, find the first difference when read from
    left-to-right. The one with a `?` is less specific, the one with a concrete
    type name in that position is more specific.
-   If there is more than one `impl` declaration with the most specific type
    structure, pick the one listed first in the priority ordering.

To ensure [coherence](goals.md#coherence), an `impl` may only be declared in a
library defining some name from its type structure. If a library defines
multiple implementations with the same type structure, they must be listed in
priority order in a prioritization block.

### Operator overloading

To overload an operator, implement the corresponding interface from the standard
library. For example, to define how the unary `-` operator behaves for a type,
implement the `Negatable` interface for that type. The interfaces and rewrites
used for a given operator may be found in the
[expressions design](/docs/design/expressions/README.md).

As a convenience, there is a shortcut for defining an implementation that
supports any type implicitly convertible to a specified type, using `like`:

```
// Support multiplying values of type `Distance` with
// values of type `f64` or any type implicitly
// convertible to `f64`.
external impl Distance as MultipliableWith(like f64) ...
```

## Future work

-   Functions should have a way to accept types that vary at runtime.
-   You should have the ability to mark entities as `upcoming` or `deprecated`
    to support evolution.
-   There should be a way to define generic associated and higher-ranked/kinded
    types.

## References

-   [#524: Generics overview](https://github.com/carbon-language/carbon-lang/pull/524)
-   [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)
-   [#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818)
-   [#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920)
-   [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
-   [#1013: Generics: Set associated constants using `where` constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
-   [#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
