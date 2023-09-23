# Generics appendix: Witness tables

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Terminology](#terminology)
    -   [Witness tables](#witness-tables)
    -   [Dynamic-dispatch witness table](#dynamic-dispatch-witness-table)
    -   [Static-dispatch witness table](#static-dispatch-witness-table)
-   [Limitations of witness tables](#limitations-of-witness-tables)
    -   [Associated constants](#associated-constants)
    -   [Blanket implementations](#blanket-implementations)
    -   [Specialization](#specialization)
    -   [Calling templated functions](#calling-templated-functions)
-   [Implementing some Carbon generic features with witness tables](#implementing-some-carbon-generic-features-with-witness-tables)
    -   [Overview](#overview-1)
    -   [Example](#example)
    -   [Associated facets example](#associated-facets-example)

<!-- tocstop -->

## Overview

Witness tables are a strategy for implementing generics, specifically for
allowing the behavior of a generic function to vary with the values of generic
parameters. They have some nice properties:

-   They can be used both for runtime and compile-time dispatch.
-   They can support separate compilation even with compile-time dispatch.

However, it can be a challenge to implement some features of a generic system
with witness tables. This leads to limitations on the generic system, additional
runtime overhead, or both.

Swift uses witness tables for both static and dynamic dispatch, accepting both
limitations and overhead. Carbon and Rust only use witness tables for dynamic
dispatch, and apply limitations to control the runtime overhead when using that
feature. As an implementation detail, Carbon compilers might also use witness
tables for static dispatch, for example when the code conforms to the
limitations of what witness tables support. However, part of the point of this
document is to state the limitations and obstacles of doing that.

## Terminology

### Witness tables

[Witness tables](https://forums.swift.org/t/where-does-the-term-witness-table-come-from/54334/4)
are an implementation strategy where values passed to a compile-time type
binding are compiled into a table of required functionality. That table is then
filled in for a given passed-in type with references to the implementation on
the original type. The generic is implemented using calls into entries in the
witness table, which turn into calls to the original type. This doesn't
necessarily imply a runtime indirection: it may be a purely compile-time
separation of concerns. However, it insists on a full abstraction boundary
between the generic user of a type and the concrete implementation.

A simple way to imagine a witness table is as a struct of function pointers, one
per method in the interface. However, in practice, it's more complex because it
must model things like associated facets and interfaces.

Witness tables are called "dictionary passing" in Haskell. Outside of generics,
a [vtable](https://en.wikipedia.org/wiki/Virtual_method_table) is very similar
to a witness table, "witnessing" the specific descendant of a base class.
Vtables, however, are passed as part of the object instead of separately.

### Dynamic-dispatch witness table

For dynamic-dispatch witness tables, actual function pointers are formed and
used as a dynamic, runtime indirection. As a result, the generic code **will
not** be duplicated for different witness tables.

### Static-dispatch witness table

For static-dispatch witness tables, the implementation is required to collapse
the table indirections at compile time. As a result, the generic code **will**
be duplicated for different witness tables.

Static-dispatch may be implemented as a performance optimization for
dynamic-dispatch that increases generated code size. The final compiled output
may not retain the witness table.

## Limitations of witness tables

### Associated constants

An interface with associated constants can use that to allow the signature of a
function to vary. A similar issue arises with argument and return values
involving `Self`. This adds to the cost of calling such functions, for example
if they are not passed by pointer, then the generated code must support
arguments and return values with a size only known at runtime.

For this reason, Rust's dynamic trait dispatch system, trait objects, only works
with traits that are
["object safe,"](https://doc.rust-lang.org/reference/items/traits.html#object-safety)
which includes a requirement that
[all the associated types have specified values](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#trait-objects).
This reduces the expressivity of Rust traits to the subset that could be
supported by a C++ abstract base class.

Swift instead supports types with size only known at runtime for its
[ABI stability and dynamic linking features](https://faultlore.com/blah/swift-abi/#what-is-abi-stability-and-dynamic-linking),
and can use that to
[support more generic features with dynamic dispatch](https://faultlore.com/blah/swift-abi/#polymorphic-generics).
This comes with runtime overhead.

### Blanket implementations

[Blanket implementations](details.md#blanket-impl-declarations) allow you define
an implementation of interface `Y` for any type implementing interface `X`. This
allows a function to use the functionality of `Y` while only having a
requirement that `X` be implemented. This creates the problem of how to go from
a witness table for `X` to a witness table for `Y`.

Rust supports blanket implementations using monomorphization, but this only
works with static dispatch. Swift does not support blanket implementations. This
is possibly a result of the limitations of using witness tables to implement
generics.

### Specialization

Specialization compounds the difficulty of the previous two issues.

An interface with an associated facet might be implemented using witness tables
by including a reference to the associated facet's witness table in the witness
table for the interface. This doesn't, though, give you a witness table for
parameterized types using the associated facet as an argument. Synthesizing
those witness tables is particularly tricky if the implementation is different
for specific types due to specialization.

Similarly, a blanket implementation can guarantee that some implementation of an
interface exists. Specialization means that actual implementation of that
interface for specific types is not the one given by the blanket implementation.
Furthermore, that specialized implementation may be in an unrelated library.
They may be found anywhere in the program, not necessarily in the dependencies
of the code that needs to use a particular witness table.

As a result, specialization is not supported by Swift, which uses witness
tables. Specialization is being considered for Rust, and is compatible with its
monomorphization model used for static dispatch.

### Calling templated functions

Carbon's planned approach to support calling a templated function from a
checked-generic function, decided in
[issue #2153](https://github.com/carbon-language/carbon-lang/issues/2153),
relies on monomorphization. Trying to rely on witness tables would result in
different semantics for calling the same function with the same types, depending
on which witness tables were available at the callsite.

## Implementing some Carbon generic features with witness tables

### Overview

A possible model for generating code for a generic function is to use a
[witness table](#witness-tables) to represent how a type implements an
interface:

-   [Interfaces](details.md#interfaces) are types of witness tables.
-   An [impl](details.md#implementing-interfaces) is a witness table value.

We can think of the interface as defining a struct type with a field for every
interface member. An implementation of that interface for a type is a value of
that struct type, which we call a witness or witness table. For example, the
function and method members of an interface correspond to function pointer
fields. An implementation will have function pointer values pointing to the
functions defining the implementation of that interface for a given type. This
is like a [vtable](https://en.wikipedia.org/wiki/Virtual_method_table), except
stored separately from the object.

A witness might
[have references to other witness tables](#associated-facets-example), in order
to support these interface features and members:

-   [associated facets](details.md#associated-facets)
-   [type parameters](details.md#parameterized-interfaces)
-   [interface requirements](details.md#interface-requiring-other-interfaces)

It also could contain constants, to store the values of
[associated constants](details.md#associated-constants), or the type's size.

### Example

For example, this `Vector` interface:

```carbon
interface Vector {
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: f64) -> Self;
}
```

from [the generic details design](details.md#interfaces) could be thought of
defining a witness table type like:

```
class Vector {
  // `Self` is the representation type, which is only
  // known at compile time.
  var Self:! type;
  // `fnty` is placeholder syntax for a "function type",
  // so `Add` is a function that takes two `Self` parameters
  // and returns a value of type `Self`.
  var Add: fnty(a: Self, b: Self) -> Self;
  var Scale: fnty(a: Self, v: f64) -> Self;
}
```

The [`impl` definition of `Vector` for `Point_Inline`](details.md#inline-impl)
would be a value of this type:

```
var VectorForPoint_Inline: Vector  = {
    .Self = Point_Inline,
    // `lambda` is placeholder syntax for defining a
    // function value.
    .Add = lambda(a: Point_Inline, b: Point_Inline) -> Point_Inline {
      return {.x = a.x + b.x, .y = a.y + b.y};
    },
    .Scale = lambda(a: Point_Inline, v: f64) -> Point_Inline {
      return {.x = a.x * v, .y = a.y * v};
    },
};
```

Since generic arguments (where the parameter is declared using `:!`) are passed
at compile time, the actual value of `VectorForPoint_Inline` can be used to
generate the code for functions using that impl.

### Associated facets example

The associated facet can be modeled by a witness table field in the interface's
witness table.

```
interface Iterator {
  fn Advance[addr self: Self*]();
}

interface Container {
  let IteratorType:! Iterator;
  fn Begin[addr self: Self*]() -> IteratorType;
}
```

could be represented by:

```
class Iterator {
  var Self:! type;
  var Advance: fnty(this: Self*);
  ...
}
class Container {
  var Self:! type;

  // Witness that IteratorType implements Iterator.
  var IteratorType:! Iterator*;

  // Method
  var Begin: fnty (this: Self*) -> IteratorType->Self;
  ...
}
```
