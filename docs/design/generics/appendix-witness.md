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
-   [Examples of how you might implement Carbon's generics with witness tables](#examples-of-how-you-might-implement-carbons-generics-with-witness-tables)
    -   [FIXME: old text from details.md "Overview" section](#fixme-old-text-from-detailsmd-overview-section)
    -   [FIXME: Checked-generics implementation model](#fixme-checked-generics-implementation-model)
    -   [FIXME: Associated facets implementation model](#fixme-associated-facets-implementation-model)
        -   [FIXME: Sized type implementation model](#fixme-sized-type-implementation-model)

<!-- tocstop -->

## Overview

Witness tables are a strategy for implementing generics, specifically for
allowing the behavior of a generic function to vary with the values of generic
parameters. They have some nice properties:

-   They can be used both for runtime and compile-time dispatch.
-   They can support separate compilation even with compile-time dispatch.

However, it can be a challenge to implement some features of a generic system
with witness tables. This leads to either limitations on the generic system or
additional runtime overhead.

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
are an implementation strategy where values passed to a generic type parameter
are compiled into a table of required functionality. That table is then filled
in for a given passed-in type with references to the implementation on the
original type. The generic is implemented using calls into entries in the
witness table, which turn into calls to the original type. This doesn't
necessarily imply a runtime indirection: it may be a purely compile-time
separation of concerns. However, it insists on a full abstraction boundary
between the generic user of a type and the concrete implementation.

A simple way to imagine a witness table is as a struct of function pointers, one
per method in the interface. However, in practice, it's more complex because it
must model things like associated facets and interfaces.

Witness tables are called "dictionary passing" in Haskell. Outside of generics,
a [vtable](https://en.wikipedia.org/wiki/Virtual_method_table) is a witness
table that witnesses that a class is a descendant of an abstract base class, and
is passed as part of the object instead of separately.

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

FIXME

### Blanket implementations

FIXME

### Specialization

FIXME

## Examples of how you might implement Carbon's generics with witness tables

### FIXME: old text from details.md "Overview" section

We can think of the interface as defining a struct type whose members are
function pointers, and an implementation of an interface as a value of that
struct with actual function pointer values. An implementation is a table mapping
the interface's functions to function pointers. For more on this, see
[the implementation model section](#fixme-checked-generics-implementation-model).

For example, the [type's size](#fixme-sized-type-implementation-model)
(represented by an integer constant member of the type) could be a member of an
interface and its implementation. There are a few cases why we would include
another interface implementation as a member:

-   [associated facets](details.md#associated-facets)
-   [type parameters](details.md#parameterized-interfaces)
-   [interface requirements](details.md#interface-requiring-other-interfaces)

### FIXME: Checked-generics implementation model

A possible model for generating code for a generic function is to use a
[witness table](#witness-tables) to represent how a type implements an
interface:

-   [Interfaces](details.md#interfaces) are types of witness tables.
-   An [impl](details.md#implementing-interfaces) is a witness table value.

Type checking is done with just the interface. The impl is used during code
generation time, possibly using
[monomorphization](https://en.wikipedia.org/wiki/Monomorphization) to have a
separate instantiation of the function for each combination of the generic
argument values. The compiler is free to use other implementation strategies,
such as passing the witness table for any needed implementations, if that can be
predicted.

For the example above, [the Vector interface](details.md#interfaces) could be
thought of defining a witness table type like:

```
class Vector {
  // `Self` is the representation type, which is only
  // known at compile time.
  var Self:! type;
  // `fnty` is **placeholder** syntax for a "function type",
  // so `Add` is a function that takes two `Self` parameters
  // and returns a value of type `Self`.
  var Add: fnty(a: Self, b: Self) -> Self;
  var Scale: fnty(a: Self, v: f64) -> Self;
}
```

The [impl of Vector for Point_Inline](details.md#inline-impl) would be a value
of this type:

```
var VectorForPoint_Inline: Vector  = {
    .Self = Point_Inline,
    // `lambda` is **placeholder** syntax for defining a
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
at compile time, so the actual value of `VectorForPoint_Inline` can be used to
generate the code for functions using that impl. This is the
[static-dispatch witness table](#static-dispatch-witness-table) approach.

### FIXME: Associated facets implementation model

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

is represented by:

```
class Iterator(Self:! type) {
  var Advance: fnty(this: Self*);
  ...
}
class Container(Self:! type) {
  // Representation type for the iterator.
  let IteratorType:! type;
  // Witness that IteratorType implements Iterator.
  var iterator_impl: Iterator(IteratorType)*;

  // Method
  var Begin: fnty (this: Self*) -> IteratorType;
  ...
}
```

#### FIXME: Sized type implementation model

This requires a special integer field be included in the witness table type to
hold the size of the type. This field will only be known generically, so if its
value is used for type checking, we need some way of evaluating those type tests
symbolically.
