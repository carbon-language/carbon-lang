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
    -   [Abstract base classes](#abstract-base-classes)
    -   [Polymorphic types](#polymorphic-types)
    -   [Mixins](#mixins)
    -   [Interop with C++ multiple inheritance](#interop-with-c-multiple-inheritance)
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
    -   [Abstract base classes interoperating with object-safe interfaces](#abstract-base-classes-interoperating-with-object-safe-interfaces)
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

The use cases for structs include both cases motivated by C++ interop, and cases
that we expect to be included in idiomatic Carbon-only code.

**This design currently only attempts to address the "data types" use case.**
Addressing the other use cases is future work.

### Data types

Data types consist of data fields that are publicly accessible and directly read
and manipulated by client code. They have few if any methods, and generally are
not involved in inheritance at all.

Examples include:

-   a key and value pair returned from a `SortedMap` or `HashMap`
-   a 2D point that might be used in a rendering API

Properties:

-   Operations like copy, move, destroy, unformed, and so on are defined
    field-wise.
-   Unnamed structs types and literals should match data type semantics.

Expected in idiomatic Carbon-only code.

**Background:** Kotlin has a dedicated concise syntax for defining
["data classes"](https://kotlinlang.org/docs/data-classes.html) that avoids
boilerplate. Python has a
[data class library](https://docs.python.org/3/library/dataclasses.html),
proposed in [PEP 557](https://www.python.org/dev/peps/pep-0557/), that fills a
similar role.

### Object types

An object type is a type that has
[encapsulation](<https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)>).
That is, its data fields are private and access and modification of values are
all done through methods defined on the type. An object type does not have a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table), and does not
support being inherited from -- they are
["final"](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Non-subclassable_classes>).

Examples: strings, containers, iterators, types with invariants such as `Date`.

Object types include:

-   RAII types that are movable but not copyable like C++'s `std::unique_ptr` or
    a file handle
-   non-movable types like `Mutex`

We expect two kinds of methods on object types: public methods defining the API
for accessing and manipulating values of the type, and private helper methods
used as an implementation detail of the public methods.

Object types are expected in idiomatic Carbon-only code. Extending this design
to support object types is future work.

### Abstract base classes

An [abstract base class](https://en.wikipedia.org/wiki/Abstract_type), or "ABC",
is a base type for use in inheritance with a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table) for dynamic
dispatch. The term "abstract" means that the base type can't be instantiated due
to methods without implementation, they are said to be "abstract" or "pure
virtual". Only child types that implement those methods may be instantiated.

Abstract base classes can't have data fields in the base type, which avoids the
main implementation difficulties and complexity of multiple inheritance. This
allows a type to inherit from multiple abstract base classes.

Abstract base classes are primarily used for
[subtyping](https://en.wikipedia.org/wiki/Subtyping). In practice that means
that if we have a type `Concrete` that is a concrete child of an abstract base
class named `ABC`, object of type `Concrete` will be accessed through pointers
of type `ABC*`, which means "a pointer to type inheriting from `ABC`." Such
types should only be allowed to be deleted by way of that pointer if a virtual
destructor is included in the abstract base class.

The use cases for abstract base classes almost entirely overlap with the
object-safe (as
[defined by Rust](https://doc.rust-lang.org/reference/items/traits.html#object-safety))
subset of [Carbon interfaces](generics/overview.md#interfaces). The main
difference is the representation in memory. A type extending an abstract base
class includes a pointer to the table of methods in the object value itself,
while a type implementing an interface would store the pointer alongside the
pointer to the value in a `DynPtr(MyInterface)`. Of course the interface option
also allows the method table to be passed at compile time.

We expect idiomatic Carbon-only code to use Carbon interfaces instead of
abstract base classes. Extending this design to support abstract base classes is
future work.

**Background:**
[Java interfaces](<https://en.wikipedia.org/wiki/Interface_(Java)>) model
abstract base classes.

### Polymorphic types

Polymorphic types support dynamic dispatch using a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table), but unlike
[abstract base classes](#abstract-base-classes) may also include private data in
a base type. Polymorphic types support traditional
[object-oriented single inheritance](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>),
a mix of [subtyping](https://en.wikipedia.org/wiki/Subtyping) and
[implementation and code reuse](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Code_reuse>).

We exclude complex multiple inheritance schemes, virtual inheritance, and so on
from this use case. This is to avoid the complexity and overhead they bring,
particularly since the use of these features in C++ is generally discouraged.
The rule is that every type has at most one parent type with data members for
subtyping purposes. Carbon will support additional parent types as long as they
are [abstract base classes](#abstract-base-classes) or [mixins](#mixins).

Characterized by: private data in a base type, a mix of concrete and
overridable/virtual methods with dynamic dispatch, accessed through a pointer to
"type extending base", virtual destructor.

**Background:** The
["Nothing is Something" talk by Sandi Metz](https://www.youtube.com/watch?v=OMPfEXIlTVE)
and
[the Composition Over Inheritance Principle](https://python-patterns.guide/gang-of-four/composition-over-inheritance/)
describe design patterns to use instead of multiple inheritance.

FIXME: Talk about the external API versus the APIs for communicating between
base and derived.
["The End Of Object Inheritance & The Beginning Of A New Modularity" talk by Augie Fackler and Nathaniel Manista](https://www.youtube.com/watch?v=3MNVP9-hglc)
discusses design patterns that split up types to reduce the number of kinds of
calls between child and parent types.

We expect polymorphic types in idiomatic Carbon-only code, at least for the
medium term. Extending this design to support polymorphic types is future work.

### Mixins

A [mixin](https://en.wikipedia.org/wiki/Mixin) is a declaration of data,
methods, and interface implementations that can be added to another type, called
the "main type". The methods of a mixin may also use data, methods, and
interface implementations provided by the main type. Mixins are designed around
implementation reuse rather than subtyping, and so don't need to use a vtable.

A mixin might be an implementation detail of a [data type](#data-types),
[object type](#object-types), or
[child of a polymorphictype](#polymorphic-types). A mixin might partially
implement an [abstract base class](#abstract-base-classes).

**Examples:**
[intrusive linked list](https://www.boost.org/doc/libs/1_63_0/doc/html/intrusive.html),
intrusive reference count

In both of these examples, the mixin needs the ability to convert between a
pointer to the mixin's data (like a "next" pointer or reference count) and a
pointer to the containing object with the main type.

Mixins are expected in idiomatic Carbon-only code. Extending this design to
support mixins is future work.

**Background:** Mixins are typically implemented using the
[curiously recurring template pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
in C++, but other languages support them directly.

-   In Dart, the mixin defines an interface that the destination type ends up
    implementing, which restores a form of subtyping. See
    [Dart: What are mixins?](https://medium.com/flutter-community/dart-what-are-mixins-3a72344011f3).
-   [Proposal to add mixin support to Swift](https://github.com/Anton3/swift-evolution/blob/mixins/proposals/NNNN-mixins.md).

### Interop with C++ multiple inheritance

iostream, virtual multiple inheritance, and base class has state; some uses of
streams are ifstream, ofstream; hard cases are the bidirectional ones: fstream,
stringstream. very few uses of fstream; one quarter are ostringstream, most
could be.

We do not expect idiomatic Carbon-only code to use multiple inheritance.
Extending this design to support interopating with C++ types using multiple
inheritance is future work.

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

We will need some way of defining
[methods](<https://en.wikipedia.org/wiki/Method_(computer_programming)>) on
structs. The big concern is how we designate the different ways the receiver can
be passed into the method. This question is being tracked in
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

### Abstract base classes interoperating with object-safe interfaces

We want four things:

-   Ability to convert an object-safe interface (a type-of-type) into an
    abstract base class (a base type), maybe using `AsBaseType(MyInterface)`.
-   Ability to convert an abstract base class (a base type) into an object-safe
    interface (a type-of-type), maybe using `AsInterface(MyABC)`.
-   Ability to convert a (thin) pointer to an abstract base class to a `DynPtr`
    of the corresponding interface.
-   We should arrange that `DynPtr(MyInterface)` should be a type extending the
    corresponding abstract base class.

### Memory layout

FIXME: Order, packing, alignment

### No `static` variables

FIXME: No `static` variables because there are no global variables.
