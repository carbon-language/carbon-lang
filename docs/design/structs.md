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
    -   [Data classes](#data-classes)
    -   [Encapsulated types](#encapsulated-types)
        -   [Without inheritance](#without-inheritance)
        -   [With inheritance and subtyping](#with-inheritance-and-subtyping)
            -   [Polymorphic types](#polymorphic-types)
                -   [Interface as base class](#interface-as-base-class)
            -   [Non-polymorphic inheritance](#non-polymorphic-inheritance)
            -   [Interop with C++ multiple inheritance](#interop-with-c-multiple-inheritance)
    -   [Mixins](#mixins)
-   [Background](#background)
-   [Overview](#overview-1)
-   [Members](#members)
    -   [Data members have an order](#data-members-have-an-order)
-   [Anonymous structs](#anonymous-structs)
    -   [Literals](#literals)
    -   [Type declarations](#type-declarations)
    -   [Option parameters](#option-parameters)
    -   [Order is ignored on assignment](#order-is-ignored-on-assignment)
    -   [Operations performed field-wise](#operations-performed-field-wise)
-   [Future work](#future-work)
    -   [Nominal struct types](#nominal-struct-types)
    -   [Construction](#construction)
    -   [Member type](#member-type)
    -   [Self](#self)
    -   [Let](#let)
    -   [Methods](#methods)
    -   [Destructuring, pattern matching, and extract](#destructuring-pattern-matching-and-extract)
    -   [Access control](#access-control)
    -   [Operator overloading](#operator-overloading)
    -   [Inheritance](#inheritance)
    -   [Abstract base classes interoperating with object-safe interfaces](#abstract-base-classes-interoperating-with-object-safe-interfaces)
    -   [Mixins](#mixins-1)
    -   [Non-virtual inheritance](#non-virtual-inheritance)
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
types. Nominal struct types are all distinct, but anonymous types are equal if
they have the same list of members. Anonymous struct literals may be used to
initialize or assign values to named struct variables.

## Use cases

The use cases for structs include both cases motivated by C++ interop, and cases
that we expect to be included in idiomatic Carbon-only code.

**This design currently only attempts to address the "data classes" use case.**
Addressing the other use cases is future work.

### Data classes

Data classes are types that consist of data fields that are publicly accessible
and directly read and manipulated by client code. They have few if any methods,
and generally are not involved in inheritance at all.

Examples include:

-   a key and value pair returned from a `SortedMap` or `HashMap`
-   a 2D point that might be used in a rendering API

Properties:

-   Operations like copy, move, destroy, unformed, and so on are defined
    field-wise.
-   Anonymous structs types and literals should match data class semantics.

Expected in idiomatic Carbon-only code.

**Background:** Kotlin has a dedicated concise syntax for defining
["data classes"](https://kotlinlang.org/docs/data-classes.html) that avoids
boilerplate. Python has a
[data class library](https://docs.python.org/3/library/dataclasses.html),
proposed in [PEP 557](https://www.python.org/dev/peps/pep-0557/), that fills a
similar role.

### Encapsulated types

There are several categories of types that support
[encapsulation](<https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)>).
This is done by making their data fields private so access and modification of
values are all done through methods defined on the type.

#### Without inheritance

The common case for encapsulated types are those that do not participate in
inheritance. These types neither extend other types nor do they support being
inherited from (they are
["final"](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Non-subclassable_classes>)).

Examples of this use case include:

-   strings, containers, iterators
-   types with invariants such as `Date`
-   RAII types that are movable but not copyable like C++'s `std::unique_ptr` or
    a file handle
-   non-movable types like `Mutex`

We expect two kinds of methods on these types: public methods defining the API
for accessing and manipulating values of the type, and private helper methods
used as an implementation detail of the public methods.

These types are expected in idiomatic Carbon-only code. Extending this design to
support these types is future work.

#### With inheritance and subtyping

FIXME

Generally needs encapsulation for subtyping

An object type does not have a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table), and does not
support being inherited from

##### Polymorphic types

Carbon will fully support single-inheritance type hierarchies with polymorphic
types.

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
The rule is that every type has at most one base type with data members for
subtyping purposes. Carbon will support additional base types as long as they
are [abstract base classes](#abstract-base-classes) or [mixins](#mixins).

While an abstract base class is an interface that allows decoupling, a
polymorphic type is a collaboration between a base and derived type to provide
some functionality. This is a bit like the difference between a library and a
framework, where you might use many of the former but only one of the latter.
However, there are some cases of overlap where there is an interface at the root
of a type hierarchy and polymorphic types as interior branches of the tree.

**Background:**
[The "Nothing is Something" talk by Sandi Metz](https://www.youtube.com/watch?v=OMPfEXIlTVE)
and
[the Composition Over Inheritance Principle](https://python-patterns.guide/gang-of-four/composition-over-inheritance/)
describe design patterns to use instead of multiple inheritance to support types
that vary over multiple axes.

Polymorphic types support a number of different kinds of methods:

-   Like abstract base classes, they will have virtual methods:
    -   Polymorphic types will always include virtual destructors.
    -   Polymorphic types may have pure-virtual methods, but in contrast to ABCs
        they aren't required.
    -   It is more common for polymorphic types to have a default implementation
        of virtual methods or have protected or
        [private](https://stackoverflow.com/questions/2170688/private-virtual-method-in-c)
        virtual methods intended to be called by methods in the base type but
        implemented in the descendant.
-   They may have non-virtual public or private helper methods, like
    encapsulated types without inheritance. These avoid the overhead of a
    virtual function call, and are more frequent in polymorphic types than
    abstract base classes due to the ability to reference some of the data
    members of the type.
-   They may have protected helper methods, typically non-virtual, provided by
    the base type to be called by the descendant.

Note that there are two uses for protected methods: those implemented in the
base and called in the descendant, and the other way around.
["The End Of Object Inheritance & The Beginning Of A New Modularity" talk by Augie Fackler and Nathaniel Manista](https://www.youtube.com/watch?v=3MNVP9-hglc)
discusses design patterns that split up types to reduce the number of kinds of
calls between base and derived types, and make sure calls only go in one
direction.

We expect polymorphic types in idiomatic Carbon-only code, at least for the
medium term. Extending this design to support polymorphic types is future work.

###### Interface as base class

**TODO:** rename to "interface base class/type" or "pure interface base
class/type", since ABCs in C++ are allowed to have data?

An [abstract base class](https://en.wikipedia.org/wiki/Abstract_type), or "ABC",
is a base type for use in inheritance with a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table) for dynamic
dispatch. The term "abstract" means that the base type can't be instantiated due
to methods without implementation, they are said to be "abstract" or "pure
virtual". Only derived types that implement those methods may be instantiated.

Abstract base classes can't have data fields in the base type, which avoids the
main implementation difficulties and complexity of multiple inheritance. This
allows a type to inherit from multiple abstract base classes.

Abstract base classes are primarily used for
[subtyping](https://en.wikipedia.org/wiki/Subtyping). In practice that means
that if we have a type `Concrete` that is a concrete type derived from an
abstract base class named `ABC`, objects of type `Concrete` will be accessed
through pointers of type `ABC*`, which means "a pointer to type inheriting from
`ABC`." Such types should only be allowed to be deleted by way of that pointer
if a virtual destructor is included in the abstract base class.

The use cases for abstract base classes almost entirely overlap with the
object-safe (as
[defined by Rust](https://doc.rust-lang.org/reference/items/traits.html#object-safety))
subset of [Carbon interfaces](generics/overview.md#interfaces). The main
difference is the representation in memory. A type extending an abstract base
class includes a pointer to the table of methods in the object value itself,
while a type implementing an interface would store the pointer alongside the
pointer to the value in a `DynPtr(MyInterface)`. Of course the interface option
also allows the method table to be passed at compile time.

The methods of an abstract base class will typically be pure virtual, with no
implementation. This is both because of the main use case is for defining an
interface without knowledge of how it is implemented, and because you typically
won't be able to define the implementation without reference to the data fields
of the object. In some cases, there may be non-abstract methods that are
implemented in terms of the pure virtual methods. In those cases, the pure
virtual methods may be
["protected"](https://en.wikipedia.org/wiki/Access_modifiers) to ensure they are
only called through the non-abstract API, but can still be implemented in
descendants.

We expect idiomatic Carbon-only code to generally use Carbon interfaces instead
of abstract base classes. We may still support abstract base classes long term
if we determine that the ability to put the pointer to the method
implementations in the object value is important for users, particularly with a
single parent as in the [polymorphic type case](#polymorphic-types). Extending
this design to support abstract base classes is future work.

**Background:**
[Java interfaces](<https://en.wikipedia.org/wiki/Interface_(Java)>) model
abstract base classes.

##### Non-polymorphic inheritance

While it is not common, there are cases where C++ code uses inheritance without
dynamic dispatch or a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table). Instead, methods
are never overridden, and derived types only add data and methods. There are
some cases where this is done in C++ but would be done differently in Carbon:

-   For implementation reuse without subtyping, in Carbon use mixins or
    composition instead of traditional inheritance. Carbon won't support private
    inheritance.
-   Carbon will allow data members to have size zero, so the
    [empty-base optimization](https://en.cppreference.com/w/cpp/language/ebo) is
    unnecessary.
-   For cases where the derived type does not add any data members, in Carbon
    you can use adapter types instead of inheritance.

However, there are still some cases where non-virtual inheritance makes sense.

**Examples:** LLVM
[red-black tree](https://github.com/llvm-mirror/libcxx/blob/master/include/__tree)
and doubly-linked lists **TODO**

##### Interop with C++ multiple inheritance

While Carbon won't support all the C++ forms of multiple inheritance, Carbon
code will still need to interoperate with C++ code that does. Of particular
concern are the `std::iostream` family of types. Most uses of those types are
the input and output variations or could be migrated to use those variations,
not the harder bidirectional cases.

Much of the complexity of this interoperation could be alleviated by adopting
the restriction that Carbon code can't directly access the fields of a virtual
base class. In the cases where such access is needed, the workaround is to
access them through C++ functions.

We do not expect idiomatic Carbon-only code to use multiple inheritance.
Extending this design to support interopating with C++ types using multiple
inheritance is future work.

### Mixins

A [mixin](https://en.wikipedia.org/wiki/Mixin) is a declaration of data,
methods, and interface implementations that can be added to another type, called
the "main type". The methods of a mixin may also use data, methods, and
interface implementations provided by the main type. Mixins are designed around
implementation reuse rather than subtyping, and so don't need to use a vtable.

A mixin might be an implementation detail of a [data class](#data-classes),
[object type](#object-types), or
[derived type of a polymorphic type](#polymorphic-types). A mixin might
partially implement an [abstract base class](#abstract-base-classes).

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
structuring data.

A `struct` type defines the interpretation of the bytes of a `struct` value,
including the size, data members, and layout. It defines the operations that may
be performed on those values, including what methods may be called on a `struct`
value. A `struct` type may directly have constant members. The type itself is a
compile-time constant value.

## Members

The members of a `struct` are named, and are accessed with the `.` notation. For
example:

```
var p: Point2D = ...;
// Data member access
p.x = 1;
p.y = 2;
// Method call
Print(p.DistanceFromOrigin());
```

[Tuples](tuples.md) are used for cases where accessing the members positionally
is more appropriate.

### Data members have an order

The data members of a struct, or "fields", have an order that matches the order
they are declared in. This affects the layout of those fields in memory, and the
order that the fields are destroyed when a value goes out of scope or is
deallocated.

## Anonymous structs

Anonymous structs are convenient for defining [data classes](#data-classes) in
an ad-hoc manner. They would commonly be used:

-   as the return type of a function that returns multiple values and wants
    those values to have names so a [tuple](tuples.md) is inappropriate
-   as a parameter to a function holding options with default values
-   as an initializer for other `struct` variables or values
-   as a type parameter to a container

**Future work:** We intend to support nominal [data classes](#data-classes) as
well.

### Literals

Anonymous struct literals are written using this syntax:

```
var kvpair: auto = {.key = "the", .value = 27};
```

This produces a struct value with two fields:

-   The first field is named "`key`" and has the value `"the"`. The type of the
    field is set to the type of the value, and so is `String`.
-   The second field is named "`value`" and has the value `27`. The type of the
    field is set to the type of the value, and so is `Int`.

**Open question:** To keep the literal syntax from being ambiguous with compound
statements, Carbon will adopt some combination of:

-   looking ahead after a `{` to see if it is followed by `.name`;
-   not allowing a literal struct at the beginning of a statement;
-   only allowing `{` to introduce a compound statement in contexts introduced
    by a keyword where they are required, like requiring `{ ... }` around the
    cases of an `if...else` statement.

### Type declarations

The type of `kvpair` in the last example would be declared:

```
struct {.key: String, .value: Int}
```

Anonymous struct may only have data members, so the type declaration is just a
list of field names, types, and optional defaults.

```
var with_defaults: struct {.x: Int = 0, .y: Int = 0} = {.y = 2};
Assert(with_defaults.x == 0);
Assert(with_defaults.y == 2);
```

Note that the type syntax uses an introducer to distinguish type declarations
from struct literals. For tuples, we can say the type of a tuple is the tuple of
the types. That policy wouldn't give a good option for defining field defaults
in `struct` types.

### Option parameters

Consider this function declaration:

```
fn SortIntVector(
    v: Vector(Int)*,
    options: struct {.stable: Bool = false, .descending: Bool = false} = {});
```

The `options` parameter defaults to `{}` which will result in the default value
for every field. So all of these calls are legal:

```
var v: Vector(Int) = ...;
// Uses defaults of `.stable` and `.descending` equal to `false`.
SortIntVector(&v);
SortIntVector(&v, {});
// Sets `.stable` option to `true`.
SortIntVector(&v, {.stable = true});
// Sets `.descending` option to `true`.
SortIntVector(&v, {.descending = true});
// Sets both `.stable` and `.descending` options to `true`.
SortIntVector(&v, {.stable = true, .descending = true});
```

### Order is ignored on assignment

When initializing or assigning a struct variable to an anonymous struct value on
the right hand side, the order of the fields do not have to match, just the
names.

```
var different_order: struct {.x: Int, .y: Int} = {.y = 2, .x = 3};
Assert(different_order.x == 3);
Assert(different_order.y == 2);

// Applicable for arguments as well.
SortIntVector(&v, {.descending = true, .stable = true});
```

### Operations performed field-wise

Assignment, destruction, and equality comparison is performed field-wise on
anonymous struct values.

```
var p: auto = {.x = 2, .y = 3};
Assert(p == {.x = 2, .y = 3});
Assert(p != {.x = 2, .y = 4});
p = {.x = 3, .y = 5};
```

Similarly, an anonymous struct has an unformed state if all its members do.

**Open question:** Should we define less-than comparison on anonymous struct
types if all its field types support it? We would have to forbid comparisons
between values with fields in different orders.

```
// Illegal
Assert({.x = 2, .y = 3} < {.y = 4, .x = 5});
```

## Future work

This includes features that need to be designed, questions to answer, and a
description of the provisional syntax in use until these decisions have been
made.

### Nominal struct types

The declarations for nominal `struct` types will have a different format.
Provisionally we have been using something like this:

```
struct TextLabel {
  var x: Int;
  var y: Int;

  var text: String;
}
```

It is an open question, though, how we will address the
[different use cases](#use-cases). For example, will we a different introducer
keyword like `class` for [polymorphic types](#polymorphic-types)?

### Construction

There are a variety of options for constructing `struct` values, we might choose
to support, including initializing from anonymous struct values:

```
var p1: Point2D = {.x = 1, .y = 2};
var p2: auto = {.x = 1, .y = 2} as Point2D;
var p3: auto = Point2D{.x = 1, .y = 2};
var p4: auto = Point2D(1, 2);
```

### Member type

Additional types may be defined in the scope of a `struct` definition.

```
struct StringCounts {
  struct Node {
    var key: String;
    var count: Int;
  }
  var counts: Vector(Node);
}
```

The inner type is a member of the type, and is given the name
`StringCounts.Node`.

### Self

A `struct` definition may provisionally include references to its own name in
limited ways, similar to an incomplete type. What is allowed and forbidden is an
open question.

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

### Let

Other type constants can provisionally be defined using a `let` declaration:

```
struct MyStruct {
  let Pi : Float32 = 3.141592653589793;
  let IndexType : Type = Int;
}
```

There are definite questions about this syntax:

-   Should these use the `:!` generic syntax decided in
    [issue #565](https://github.com/carbon-language/carbon-lang/issues/565)?
-   Would we also have `alias` declarations? How would they be different?

### Methods

A future proposal will incorporate
[method](<https://en.wikipedia.org/wiki/Method_(computer_programming)>)
declaration, definition, and calling into structs. The syntax for declaring
methods has been decided in
[question-for-leads issue #494](https://github.com/carbon-language/carbon-lang/issues/494).
Summarizing that issue:

-   Accessors are written: `fn Diameter[me: Self]() -> Float { ... }`
-   Mutators are written: `fn Expand[addr me: Self*](distance: Float) { ... }`
-   Associated functions that don't take a receiver at all, like
    [C++'s static methods](<https://en.wikipedia.org/wiki/Static_(keyword)#Static_method>),
    are written: `fn Create() -> Self { ... }`

We do not expect to have implicit member access in methods, so inside the method
body members will be accessed through the `me` parameter.

### Destructuring, pattern matching, and extract

It is an open question how we might destructure or match `struct` values.

```
var (key: String, value: Int) = {.key = "k", .value = 42};
var (key: String, value: Int) =
   {.key = "k", .value = 42}.extract(.key, .value);
```

Some discussion on this topic has occurred in:

-   [question-for-leads issue #505 on named parameters](https://github.com/carbon-language/carbon-lang/issues/505)
-   labeled params brainstorming docs
    [1](https://docs.google.com/document/d/1a1wI8SHGh3HYV8SUWPIKhg48ZW2glUlAMIIS3aec5dY/edit),
    [2](https://docs.google.com/document/d/1u6GORSkcgThMAiYKOqsgALcEviEtcghGb5TTVT-U-N0/edit)
-   ["match" in syntax choices doc](https://docs.google.com/document/d/1iuytei37LPg_tEd6xe-O6P_bpN7TIbEjNtFMLYW2Nno/edit#heading=h.y566d16ivoy2)

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

Carbon will need ways of saying:

-   this `struct` type has a virtual method table
-   this `struct` type extends a base type
-   this `struct` type is "final" and may not be extended further
-   this method is "virtual" and may be overridden in descendents
-   this method is "pure virtual" or "abstract" and must be overridden in
    descendants
-   this method overrides a method declared in a base type

Multiple inheritance will be limited in at least a couple of ways:

-   At most one supertype may define data members.
-   Carbon types can't access data members of C++ virtual base classes.

There is a
[document considering the options for constructing objects with inheritance](https://docs.google.com/document/d/1GyrBIFyUbuLJGItmTAYUf9sqSDQjry_kjZKm5INl-84/edit).

### Abstract base classes interoperating with object-safe interfaces

We want four things so that Carbon's object-safe interfaces may interoperate
with C++ abstract base classes:

-   Ability to convert an object-safe interface (a type-of-type) into an
    abstract base class (a base type), maybe using `AsBaseType(MyInterface)`.
-   Ability to convert an abstract base class (a base type) into an object-safe
    interface (a type-of-type), maybe using `AsInterface(MyABC)`.
-   Ability to convert a (thin) pointer to an abstract base class to a `DynPtr`
    of the corresponding interface.
-   We should arrange that `DynPtr(MyInterface)` should be a type extending the
    corresponding abstract base class.

### Mixins

We will need some way to declare mixins. This syntax will need a way to
distinguish defining versus requiring member variables. Methods may additionally
be given a default definition but may be overridden. Interface implementations
may only be partially provided by a mixin. Mixin methods will need to be able to
convert between pointers to the mixin type and the main type.

Mixins also complicate how constructors work.

### Non-virtual inheritance

We need some way of addressing two safety concerns created by non-virtual
inheritance:

-   Unclear how to safely run the right destructor.
-   [Object slicing](https://en.wikipedia.org/wiki/Object_slicing) is a danger
    when dereferencing a pointer of a base type.

These concerns would be resolved by distinguishing between pointers that point
to a specified type only and those that point to a type or any subtype. The
latter case would have restrictions to prevent misuse. This distinction may be
more complexity than is justified for a relatively rare use case.

### Memory layout

Carbon will need some way for users to specify the memory layout of `struct`
types, such as controlling the packing and alignment for the whole type or
individual members.

### No `static` variables

At the moment, there is no proposal to support
[`static` member variables](https://en.wikipedia.org/wiki/Class_variable#Static_member_variables_and_static_member_functions),
in line with avoiding global variables more generally.
