# Structs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Use cases](#use-cases)
    -   [Data types](#data-types)
    -   [Object types](#object-types)
    -   [Polymorphic types](#polymorphic-types)
    -   [Abstract base classes](#abstract-base-classes)
    -   [Other cases](#other-cases)
-   [Background](#background)
-   [Overview](#overview-1)
-   [Fields have an order](#fields-have-an-order)
-   [Anonymous structs](#anonymous-structs)
    -   [Literals](#literals)
    -   [Type declarations](#type-declarations)
    -   [Anonymous to named conversion](#anonymous-to-named-conversion)
    -   [Order is ignored on assignment](#order-is-ignored-on-assignment)
-   [Fields may have defaults](#fields-may-have-defaults)
-   [Member type](#member-type)
-   [Self](#self)
-   [Alias](#alias)
-   [Future work](#future-work)
    -   [Method syntax](#method-syntax)
    -   [Destructuring, pattern matching, and extract](#destructuring-pattern-matching-and-extract)
    -   [Access control](#access-control)
    -   [Operator overloading](#operator-overloading)
    -   [Inheritance](#inheritance)
    -   [Memory layout](#memory-layout)
    -   [No `static` variables](#no-static-variables)

<!-- tocstop -->

## Overview

A Carbon `struct` is a user-defined
[record type](<https://en.wikipedia.org/wiki/Record_(computer_science)>). This
is the primary mechanism for users to define new types in Carbon. A `struct` has
members that are referenced by their names, in contrast to a
[Carbon tuple](tuples.md) which defines a
[product type](https://en.wikipedia.org/wiki/Product_type) whose members are
referenced positionally.

Carbon supports both named, or "nominal", and unnamed, or "anonymous", struct
types. Named struct types are all distinct, but unnamed types are equal if they
have the same list of members. Unnamed struct literals may be used to initialize
or assign values to named struct variables.

## Use cases

The use cases for structs fall under three main categories: data types, object
types, and polymorphic types.

### Data types

Characterized by: public data fields, few if any methods, final.

Example: Key and value pair returned from a `SortedMap` or `HashMap`.

Properties:

FIXME

### Object types

FIXME

Characterized by: private data fields, non-overridable methods, final.

Examples: strings, containers, iterators, types with invariants like `Date`

Include:

-   RAII types that are movable but not copyable like C++'s `std::unique_ptr` or
    a file handle
-   non-movable types like `Mutex`

Extending this design to support object types is future work.

FIXME: public API and private helper API

### Polymorphic types

Characterized by: inheritance, private data in a base type, overridable/virtual
methods with dynamic dispatch, accessed through a pointer to "type extending
base"

Excluding complex multiple inheritance schemes, virtual inheritance, etc. One
base class for purposes of subtyping. Can have other parents as "mixins".

FIXME: Talk about the external API versus the APIs for communicating between
base and derived.

FIXME

Extending this design to support polymorphic types is future work.

### Abstract base classes

Can we use interfaces as the model for C++ ABCs to support implementing
multiple? Can we skip supporting derived-to-base conversions in this case?

How do you pass something implementing this from Carbon to C++?

Maybe we can say: you can inherit from multiple base classes but only the first
can have any data?

### Other cases

inheritance without virtual: intrusive linked list, need to be able to get from
the intrusive linked list type to the main type; type can be part of another
inheritance hierarchy. Could be handled in Carbon by giving mix-ins the
capability to go from their pointer to the pointer to the containing object.

iostream, virtual multiple inheritance, and base class has state; some uses of
streams are ifstream, ofstream; hard cases are the bidirectional ones: fstream,
stringstream. very few uses of fstream; one quarter are ostringstream, most
could be.

Do we want to expose Carbon interfaces to C++ as ABCs?

## Background

See how other languages tackle this problem:

-   [Swift](https://docs.swift.org/swift-book/LanguageGuide/ClassesAndStructures.html)
    -   has two different concepts: classes support
        [inheritance](https://docs.swift.org/swift-book/LanguageGuide/Inheritance.html)
        and use
        [reference counting](https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html)
        while structs have value semantics
    -   may have
        [constructor functions called "initializers"](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html)
        and
        [destructors called "deinitializers"](https://docs.swift.org/swift-book/LanguageGuide/Deinitialization.html)
    -   supports
        [properties](https://docs.swift.org/swift-book/LanguageGuide/Properties.html),
        including computed & lazy properties
    -   methods are const by default
        [unless marked mutating](https://docs.swift.org/swift-book/LanguageGuide/Methods.html#ID239)
    -   supports
        [extensions](https://docs.swift.org/swift-book/LanguageGuide/Extensions.html)
    -   has per-field
        [access control](https://docs.swift.org/swift-book/LanguageGuide/AccessControl.html)
-   [Rust](https://doc.rust-lang.org/book/ch05-01-defining-structs.html)
    -   has no support for inheritance
    -   has no special constructor functions, instead has literal syntax
    -   has some convenience syntax for common cases:
        [variable and field names matching](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#using-the-field-init-shorthand-when-variables-and-fields-have-the-same-name),
        [updating a subset of fields](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#creating-instances-from-other-instances-with-struct-update-syntax)
    -   [can have unnamed fields](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#using-tuple-structs-without-named-fields-to-create-different-types)
    -   [supports structs with size 0](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
-   [Zig](https://ziglang.org/documentation/0.6.0/#struct)
    -   [explicitly mark structs as packed to manually control layout](https://ziglang.org/documentation/0.6.0/#packed-struct)
    -   has a struct literal syntax,
        [including for anonymous structs](https://ziglang.org/documentation/0.6.0/#Anonymous-Struct-Literals)
    -   no special constructor functions
    -   supports fields with undefined values
    -   supports structs with size 0
    -   supports generics by way of memoized compile time functions accepting
        and returning types
    -   [supports default field values](https://ziglang.org/documentation/0.6.0/#toc-Default-Field-Values)
    -   [has no properties or operator overloading -- Zig does not like hidden control flow](https://ziglang.org/#Small-simple-language)

## Overview

Beyond simple tuples, Carbon of course allows defining
[record types](<https://en.wikipedia.org/wiki/Record_(computer_science)>). This
is the primary mechanism for users to extend the Carbon type system and
fundamentally is deeply rooted in C++ and its history (C and Simula). We simply
call them `struct`s rather than other terms as it is both familiar to existing
programmers and accurately captures their essence: they are a mechanism for
structuring data:

```
struct Widget {
  var x: Int;
  var y: Int;
  var z: Int;

  var payload: String;
}
```

The type itself is a compile-time constant value. All name access is done with
the `.` notation.

## Fields have an order

FIXME

## Anonymous structs

FIXME

### Literals

FIXME

```
{.key = "the", .value: 27}
```

### Type declarations

FIXME

```
struct {.key: String, .value: Int}
```

### Anonymous to named conversion

FIXME

### Order is ignored on assignment

FIXME

## Fields may have defaults

FIXME

## Member type

```
struct StringCounts {
  struct Node {
    var key: String;
    var count: Int;
  }
  var counts: Vector(Node);
}
```

The inner type is given the name `StringCounts.Node`.

## Self

Allowed to reference your own name inside a struct, but in limited ways, similar
to an incomplete type.

```
struct IntListNode {
  var data: Int;
  var next: IntListNode*;
}
```

`Self` is an alias for the current type:

```
struct IntListNode {
  var data: Int;
  var next: Self*;
}
```

`Self` refers to the innermost type declaration:

```
struct IntList {
  struct IntListNode {
    var data: Int;
    var next: Self*;
  }
  var first: IntListNode*;
}
```

## Alias

FIXME

## Future work

### Method syntax

We will need some way of defining methods on structs. The big concern is how we
designate the different ways the receiver can be passed into the method. This
question is being tracked in
[question-for-leads issue #494](https://github.com/carbon-language/carbon-lang/issues/494).
As an example, we need some way to distinguish methods that don't take a
receiver at all, like
[C++'s static methods](<https://en.wikipedia.org/wiki/Static_(keyword)#Static_method>)

We do not expect to have implicit member access in methods.

### Destructuring, pattern matching, and extract

FIXME

### Access control

We will need some way of controlling access to the members of structs. For now,
we assume all members are fully publicly accessible.

The default access control level, and the options for access control, are pretty
large open questions. Swift and C++ (especially w/ modules) provide a lot of
options and a pretty wide space to explore here. If the default isn't right most
of the time, access control runs the risk of becoming a significant ceremony
burden that we may want to alleviate with grouped access regions instead of
per-entity specifiers. Grouped access regions have some other advantages in
terms of pulling the public interface into a specific area of the type.

### Operator overloading

This includes destructors, copy and move operations, as well as other Carbon
operators such as `+` and `/`. We expect types to implement these operations by
implementing corresponding interfaces, see
[the generics overview](generics/overview.md).

### Inheritance

FIXME: limited multiple inheritance

FIXME:
[doc with constructor options for inheritance](https://docs.google.com/document/d/1GyrBIFyUbuLJGItmTAYUf9sqSDQjry_kjZKm5INl-84/edit)

### Memory layout

FIXME: Order, packing, alignment

### No `static` variables

FIXME: No `static` variables because there are no global variables.
