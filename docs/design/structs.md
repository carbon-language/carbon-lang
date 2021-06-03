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
-   [Background](#background)
-   [Overview](#overview-1)
-   [Future work](#future-work)
    -   [Method syntax](#method-syntax)
    -   [Constant members](#constant-members)
    -   [Access control](#access-control)
    -   [Operator overloading](#operator-overloading)
    -   [Inheritance](#inheritance)
    -   [Memory layout](#memory-layout)

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

FIXME

### Data types

FIXME

### Object types

FIXME

### Polymorphic types

FIXME

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

Other members and member functions needing an object parameter (or "methods")
must be accessed from an object of the type.

Some things in C++ are notably absent or orthogonally handled:

-   No need for `static` functions, they simply don't take an initial `self`
    parameter.
-   No `static` variables because there are no global variables. Instead, can
    have scoped constants.

## Future work

### Method syntax

We will need some way of defining methods on structs. The big concern is how we
designate the different ways the receiver can be passed into the method. This
question is being tracked in
[question-for-leads issue #494](https://github.com/carbon-language/carbon-lang/issues/494).

### Constant members

We need some syntax for defining constant members. These include:

-   Member types, like an iterator type
-   Constant member values, like C++'s `std::string::npos`
-   Functions that don't take a receiver, like
    [C++'s static methods](<https://en.wikipedia.org/wiki/Static_(keyword)#Static_method>)

All of these may be accessed as members of the type itself, without a value of
that type.

### Access control

We will need some way of controlling access to the members of structs. For now,
we assume all members are fully publicly accessible.

### Operator overloading

This includes destructors, copy and move operations, as well as other Carbon
operators such as `+` and `/`. We expect types to implement these operations by
implementing corresponding interfaces, see
[the generics overview](generics/overview.md).

### Inheritance

FIXME: no multiple inheritance

### Memory layout

FIXME: Order, packing, alignment
