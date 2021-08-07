# Classes

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
-   [Struct types](#struct-types)
    -   [Literals](#literals)
    -   [Type expression](#type-expression)
    -   [Assignment and initialization](#assignment-and-initialization)
    -   [Operations performed field-wise](#operations-performed-field-wise)
-   [Future work](#future-work)
    -   [Nominal class types](#nominal-class-types)
    -   [Construction](#construction)
    -   [Member type](#member-type)
    -   [Self](#self)
    -   [Let](#let)
    -   [Methods](#methods)
    -   [Optional named parameters](#optional-named-parameters)
        -   [Field defaults for struct types](#field-defaults-for-struct-types)
        -   [Destructuring in pattern matching](#destructuring-in-pattern-matching)
        -   [Discussion](#discussion)
    -   [Access control](#access-control)
    -   [Operator overloading](#operator-overloading)
    -   [Inheritance](#inheritance)
    -   [C++ abstract base classes interoperating with object-safe interfaces](#c-abstract-base-classes-interoperating-with-object-safe-interfaces)
    -   [Mixins](#mixins-1)
    -   [Non-virtual inheritance](#non-virtual-inheritance)
    -   [Memory layout](#memory-layout)
    -   [No `static` variables](#no-static-variables)
    -   [Computed properties](#computed-properties)
    -   [Interfaces implemented for data classes](#interfaces-implemented-for-data-classes)

<!-- tocstop -->

## Overview

A Carbon `class` is a user-defined
[record type](<https://en.wikipedia.org/wiki/Record_(computer_science)>). This
is the primary mechanism for users to define new types in Carbon. A `class` has
members that are referenced by their names, in contrast to a
[Carbon tuple](tuples.md) which defines a
[product type](https://en.wikipedia.org/wiki/Product_type) whose members are
referenced positionally.

Carbon supports both named, or "nominal", and unnamed, anonymous, or
"structural", class types. Nominal class types are all distinct, but structural
types are equal if they have the same sequence of member types and names.
Structural class literals may be used to initialize or assign values to nominal
class variables.

## Use cases

The use cases for classes include both cases motivated by C++ interop, and cases
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
-   Anonymous classes types and literals should match data class semantics.

Expected in idiomatic Carbon-only code.

**Background:** Kotlin has a dedicated concise syntax for defining
[_data classes_](https://kotlinlang.org/docs/data-classes.html) that avoids
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
inheritance. These types neither support being inherited from (they are
["final"](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Non-subclassable_classes>))
nor do they extend other types.

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

The [subtyping](https://en.wikipedia.org/wiki/Subtyping) you get with
inheritance is that you may assign the address of an object of a derived type to
a pointer to its base type. For this to work, the compiler needs implementation
strategies that allow operations performed through the pointer to the base type
work independent of which derived type it actually points to. These strategies
include:

-   Arranging for the the data layout of derived types to start with the data
    layout of the base type as a prefix.
-   Putting a pointer to a table of function pointers, a
    [_vtable_](https://en.wikipedia.org/wiki/Virtual_method_table), as the first
    data member of the object. This allows methods to be
    [_virtual_](https://en.wikipedia.org/wiki/Virtual_function) and have a
    derived-type-specific implementation, an _override_, that is used even when
    invoking the method on a pointer to a base type.
-   Non-virtual methods implemented on a base type should be applicable to all
    derived types. In general, derived types should not attempt to overload or
    override non-virtual names defined in the base type.

Note that these subtyping implementation strategies generally rely on
encapsulation, but encapsulation is not a strict requirement in all cases.

This subtyping relationship also creates safety concerns, which Carbon should
protect against.
[Slicing problems](https://en.wikipedia.org/wiki/Object_slicing) can arise when
the source or target of an assignment is a dereferenced pointer to the base
type. It is also incorrect to delete an object with a non-virtual destructor
through a pointer to a base type.

##### Polymorphic types

Carbon will fully support single-inheritance type hierarchies with polymorphic
types.

Polymorphic types support
[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch) using a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table), and data members,
but only single inheritance. Individual methods opt in to using dynamic
dispatch, so types will have a mix of
["virtual"](https://en.wikipedia.org/wiki/Virtual_function) and non-virtual
methods. Polymorphic types support traditional
[object-oriented single inheritance](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>),
a mix of [subtyping](https://en.wikipedia.org/wiki/Subtyping) and
[implementation and code reuse](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Code_reuse>).

We exclude complex multiple inheritance schemes, virtual inheritance, and so on
from this use case. This is to avoid the complexity and overhead they bring,
particularly since the use of these features in C++ is generally discouraged.
The rule is that every type has at most one base type with data members for
subtyping purposes. Carbon will support additional base types as long as they
[don't have data members](#interface-as-base-class) or
[don't support subtyping](#mixins).

**Background:**
[The "Nothing is Something" talk by Sandi Metz](https://www.youtube.com/watch?v=OMPfEXIlTVE)
and
[the Composition Over Inheritance Principle](https://python-patterns.guide/gang-of-four/composition-over-inheritance/)
describe design patterns to use instead of multiple inheritance to support types
that vary over multiple axes.

In rare cases where the complex multiple inheritance schemes of C++ are truly
needed, they can be effectively approximated using a combination of these
simpler building blocks.

Polymorphic types support a number of different kinds of methods:

-   They will have virtual methods:
    -   Polymorphic types will typically include virtual destructors.
    -   The virtual methods types may have default implementations or be
        [_abstract_](<https://en.wikipedia.org/wiki/Method_(computer_programming)#Abstract_methods>)
        (or
        [_pure virtual_](https://en.wikipedia.org/wiki/Virtual_function#Abstract_classes_and_pure_virtual_functions)).
        In the latter case, they must be implemented in any derived class that
        can be instantiated.
    -   Virtual methods may be
        [_protected_](https://en.wikipedia.org/wiki/Access_modifiers) or
        [_private_](https://stackoverflow.com/questions/2170688/private-virtual-method-in-c),
        intended to be called by methods in the base type but implemented in the
        descendant.
-   They may have non-virtual public or private helper methods, like
    [encapsulated types without inheritance](#without-inheritance). These avoid
    the overhead of a virtual function call, and can be written when the base
    class has sufficient data members.
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

We distinguish the specific case of polymorphic base classes that have no data
members:

-   From an implementation perspective, the lack of data members removes most of
    the problems with supporting multiple inheritance.
-   They are about decoupling two pieces of code instead of collaborating.
-   As a use case, they are used primarily for subtyping and much less
    implementation reuse than other polymorphic types.
-   This case overlaps with the
    [interface](/docs/design/generics/terminology.md#interface) concept
    introduced for [Carbon generics](/docs/design/generics/overview.md).

Removing support for data fields greatly simplifies supporting multiple
inheritance. For example, it removes the need for a mechanism to figure out the
offset of those data fields in the object. Similarly we don't need
[C++'s virtual inheritance](https://en.wikipedia.org/wiki/Virtual_inheritance)
to avoid duplicating those fields. Some complexities still remain, such as
pointers changing values when casting to a secondary parent type, but these seem
manageable given the benefits of supporting this useful case of multiple
inheritance.

While an interface base class is generally for providing an API that allows
decoupling two pieces of code, a polymorphic type is a collaboration between a
base and derived type to provide some functionality. This is a bit like the
difference between a library and a framework, where you might use many of the
former but only one of the latter.

Interface base classes are primarily used for subtyping. The extent of
implementation reuse is generally limited by the lack of data members, and the
decoupling role they play is usually about defining an API as a set of public
pure-virtual methods. Compared to other polymorphic types, they more rarely have
methods with implementations (virtual or not), or have methods with restricted
access. The main use case is when there is a method that is implemented in terms
of pure-virtual methods. Those pure-virtual methods may be marked as protected
to ensure they are only called through the non-abstract API, but can still be
implemented in descendants.

While it is typical for this case to be associated with single-level inheritance
hierarchies, there are some cases where there is an interface at the root of a
type hierarchy and polymorphic types as interior branches of the tree. The case
of generic interfaces extending or requiring other interface would also be
modeled by deeper inheritance hierarchies.

An interface as base class needs to either have a virtual destructor or forbid
deallocation.

There is significant overlap between interface base classes and
[Carbon interfaces](generics/overview.md#interfaces). Both represent APIs as a
collection of method names and signatures to implement. The subset of interfaces
that support dynamic dispatch are called _object-safe_, following
[Rust](https://doc.rust-lang.org/reference/items/traits.html#object-safety):

-   They don't have a `Self` in the signature of a method in a contravariant
    position like a parameter.
-   They don't have free associated types or other associated items used in a
    method signature.

The restrictions on object-safe interfaces match the restrictions on base class
methods. The main difference is the representation in memory. A type extending a
base class with virtual methods includes a pointer to the table of methods in
the object value itself, while a type implementing an interface would store the
pointer alongside the pointer to the value in a `DynPtr(MyInterface)`. Of
course, the interface option also allows the method table to be passed at
compile time.

**Note:** This presumes that we include some concept of `final` methods in
interfaces to match non-virtual functions in base classes.

We expect idiomatic Carbon-only code to generally use Carbon interfaces instead
of interface base classes. We may still support interface base classes long term
if we determine that the ability to put the pointer to the method
implementations in the object value is important for users, particularly with a
single parent as in the [polymorphic type case](#polymorphic-types). Extending
this design to support interface base classes is future work.

**Background:**
[C++ abstract base classes](https://en.wikipedia.org/wiki/Abstract_type) that
don't have data members and
[Java interfaces](<https://en.wikipedia.org/wiki/Interface_(Java)>) model this
case.

##### Non-polymorphic inheritance

While it is not common, there are cases where C++ code uses inheritance without
dynamic dispatch or a
[vtable](https://en.wikipedia.org/wiki/Virtual_method_table). Instead, methods
are never overridden, and derived types only add data and methods. There are
some cases where this is done in C++ but would be done differently in Carbon:

-   For implementation reuse without subtyping, Carbon code should use mixins or
    composition. Carbon won't support private inheritance.
-   Carbon will allow data members to have size zero, so the
    [empty-base optimization](https://en.cppreference.com/w/cpp/language/ebo) is
    unnecessary.
-   For cases where the derived type does not add any data members, in Carbon
    you can potentially use adapter types instead of inheritance.

However, there are still some cases where non-virtual inheritance makes sense.
One is a parameterized type where a prefix of the data is the same independent
of the parameter. An example of this is containers with a
[small-buffer optimization](https://akrzemi1.wordpress.com/2014/04/14/common-optimizations/#sbo),
as described in the talk
[CppCon 2016: Chandler Carruth "High Performance Code 201: Hybrid Data Structures"](https://www.youtube.com/watch?v=vElZc6zSIXM).
By moving the data and methods that don't depend on the buffer size to a base
class, we reduce the instantiation overhead for monomorphization. The base type
is also useful for reducing instantiation for consumers of the container, as
long as they only need to access methods defined in the base.

Another case for non-virtual inheritance is for different node types within a
data structure that have some data members in common. This is done in LLVM's
map,
[red-black tree](https://github.com/llvm-mirror/libcxx/blob/master/include/__tree),
and list data structure types. In a linked list, the base type might have the
next and previous pointers, which is enough for a sentinel node, and there would
also be a derived type with the actual data member. The base type can define
operations like "splice" that only operate on the pointers not the data, and
this is in fact enforced by the type system. Only the derived node type needs to
be parameterized by the element type, saving on instantiation costs as before.

Many of the concerns around non-polymorphic inheritance are the same as for the
non-virtual methods of [polymorphic types](#polymorphic-types). Assignment and
destruction are examples of operations that need particular care to be sure they
are only done on values of the correct type, rather than through a subtyping
relationship. This means having some extrinsic way of knowing when it is safe to
downcast before performing one of those operations, or performing them on
pointers that were never upcast to the base type.

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
partially implement an [interface as base class](#interface-as-base-class).

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
-   Swift is considering
    [a proposal to add mixin support](https://github.com/Anton3/swift-evolution/blob/mixins/proposals/NNNN-mixins.md).

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

Beyond tuples, Carbon allows defining
[record types](<https://en.wikipedia.org/wiki/Record_(computer_science)>). This
is the primary mechanism for users to extend the Carbon type system and is
deeply rooted in C++ and its history (C and Simula). We call them _classes_
rather than other terms as that is both familiar to existing programmers and
accurately captures their essence: they define the types of objects with
(optional) support for methods, encapsulation, and so on.

A class type defines the interpretation of the bytes of a value of that type,
including the size, data members, and layout. It defines the operations that may
be performed on those values, including what methods may be called. A class type
may directly have constant members. The type itself is a compile-time immutable
constant value.

## Members

The members of a class are named, and are accessed with the `.` notation. For
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

The data members of a class, or _fields_, have an order that matches the order
they are declared in. This determines the order of those fields in memory, and
the order that the fields are destroyed when a value goes out of scope or is
deallocated.

## Struct types

_Structural data classes_, or _struct types_, are convenient for defining
[data classes](#data-classes) in an ad-hoc manner. They would commonly be used:

-   as the return type of a function that returns multiple values and wants
    those values to have names so a [tuple](tuples.md) is inappropriate
-   as an initializer for other `class` variables or values
-   as a type parameter to a container

Note that struct types are examples of _data class types_ and are still classes,
but we expect later to support more ways to define data class types. Also note
that there is no `struct` keyword, "struct" is just convenient shorthand
terminology for a structural data class.

**Future work:** We intend to support nominal [data classes](#data-classes) as
well.

### Literals

_Structural data class literals_, or _struct literals_, are written using this
syntax:

```
var kvpair: auto = {.key = "the", .value = 27};
```

This produces a struct value with two fields:

-   The first field is named "`key`" and has the value `"the"`. The type of the
    field is set to the type of the value, and so is `String`.
-   The second field is named "`value`" and has the value `27`. The type of the
    field is set to the type of the value, and so is `Int`.

Note: A comma may optionally be included before the closing curly brace `}`:

```
var kvpair: auto = {.key = "the", .value = 27,};
```

**Open question:** To keep the literal syntax from being ambiguous with compound
statements, Carbon will adopt some combination of:

-   looking ahead after a `{` to see if it is followed by `.name`;
-   not allowing a struct literal at the beginning of a statement;
-   only allowing `{` to introduce a compound statement in contexts introduced
    by a keyword where they are required, like requiring `{ ... }` around the
    cases of an `if...else` statement.

### Type expression

The type of `kvpair` in the last example would be represented by this
expression:

```
{.key: String, .value: Int}
```

This syntax is intended to parallel the literal syntax, and so uses commas (`,`)
to separate fields instead of a semicolon (`;`) terminator. This choice also
reflects the expected use inline in function signature declarations.

Struct types may only have data members, so the type declaration is just a list
of field names and types. The result of a struct type expression is an immutable
compile-time type value.

Note: Like with struct literal expressions, a comma may optionally be included
before the closing curly brace `}`:

```
{.key: String, .value: Int,}
```

Also note that `{}` represents both the empty struct literal and its type.

### Assignment and initialization

When initializing or assigning a variable with a data class such as a struct
type to a struct value on the right hand side, the order of the fields does not
have to match, just the names.

```
var different_order: {.x: Int, .y: Int} = {.y = 2, .x = 3};
Assert(different_order.x == 3);
Assert(different_order.y == 2);
```

**Open question:** What operations and in what order happen for assignment and
initialization?

-   Is assignment just destruction followed by initialization? Is that
    destruction completed for the whole object before initializing, or is it
    interleaved field-by-field?
-   When initializing to a literal value, is a temporary containing the literal
    value constructed first or are the fields initialized directly? The latter
    approach supports types that can't be moved or copied, such as mutex.
-   Perhaps some operations are _not_ ordered with respect to each other?

When initializing or assigning, the order of fields is determined from the
target on the left side of the `=`. This rule matches what we expect for classes
with encapsulation more generally.

### Operations performed field-wise

Generally speaking, the operations that are available on a data class value,
such as a value with a struct type, are dependent on those operations being
available for all the types of the fields.

For example, two values of the same data class type may be compared for equality
if equality is supported for every member of the type:

```
var p: auto = {.x = 2, .y = 3};
Assert(p == {.x = 2, .y = 3});
Assert(p != {.x = 2, .y = 4});
```

Similarly, a data class has an unformed state if all its members do. Treatment
of unformed state follows
[#257](https://github.com/carbon-language/carbon-lang/pull/257).

`==` and `!=` are defined on a data class type if all its field types support
it:

```
Assert({.x = 2, .y = 4} != {.x = 5, .y = 3});
```

**Open question:** Which other comparisons are supported is the subject of
[question-for-leads issue #710](https://github.com/carbon-language/carbon-lang/issues/710).

```
// Illegal
Assert({.x = 2, .y = 3} != {.y = 4, .x = 5});
```

Destruction is performed field-wise in reverse order.

Extending user-defined operations on the fields to an operation on an entire
data class is [future work](#interfaces-implemented-for-data-classes).

## Future work

This includes features that need to be designed, questions to answer, and a
description of the provisional syntax in use until these decisions have been
made.

### Nominal class types

The declarations for nominal class types will have a different format.
Provisionally we have been using something like this:

```
class TextLabel {
  var x: Int;
  var y: Int;

  var text: String;
}
```

It is an open question, though, how we will address the
[different use cases](#use-cases). For example, we might mark
[data classes](#data-classes) with an `impl as Data {}` line.

### Construction

There are a variety of options for constructing class values, we might choose to
support, including initializing from struct values:

```
var p1: Point2D = {.x = 1, .y = 2};
var p2: auto = {.x = 1, .y = 2} as Point2D;
var p3: auto = Point2D{.x = 1, .y = 2};
var p4: auto = Point2D(1, 2);
```

### Member type

Additional types may be defined in the scope of a class definition.

```
class StringCounts {
  class Node {
    var key: String;
    var count: Int;
  }
  var counts: Vector(Node);
}
```

The inner type is a member of the type, and is given the name
`StringCounts.Node`.

### Self

A `class` definition may provisionally include references to its own name in
limited ways, similar to an incomplete type. What is allowed and forbidden is an
open question.

```
class IntListNode {
  var data: Int;
  var next: IntListNode*;
}
```

An equivalent definition of `IntListNode`, since `Self` is an alias for the
current type, is:

```
class IntListNode {
  var data: Int;
  var next: Self*;
}
```

`Self` refers to the innermost type declaration:

```
class IntList {
  class IntListNode {
    var data: Int;
    var next: Self*;
  }
  var first: IntListNode*;
}
```

### Let

Other type constants can provisionally be defined using a `let` declaration:

```
class MyClass {
  let Pi: Float32 = 3.141592653589793;
  let IndexType: Type = Int;
}
```

There are definite questions about this syntax:

-   Should these use the `:!` generic syntax decided in
    [issue #565](https://github.com/carbon-language/carbon-lang/issues/565)?
-   Would we also have `alias` declarations? They would only be used for names,
    not other constant values.

### Methods

A future proposal will incorporate
[method](<https://en.wikipedia.org/wiki/Method_(computer_programming)>)
declaration, definition, and calling into classes. The syntax for declaring
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

### Optional named parameters

Structs are being considered as a possible mechanism for implementing optional
named parameters. We have three main candidate approaches: allowing struct types
to have field defaults, having dedicated support for destructuring struct values
in pattern contexts, or having a dedicated optional named parameter syntax.

#### Field defaults for struct types

If struct types could have field defaults, you could write a function
declaration with all of the optional parameters in an option struct:

```
fn SortIntVector(
    v: Vector(Int)*,
    options: {.stable: Bool = false,
              .descending: Bool = false} = {}) {
  // Code using `options.stable` and `options.descending`.
}

// Uses defaults of `.stable` and `.descending` equal to `false`.
SortIntVector(&v);
SortIntVector(&v, {});
// Sets `.stable` option to `true`.
SortIntVector(&v, {.stable = true});
// Sets `.descending` option to `true`.
SortIntVector(&v, {.descending = true});
// Sets both `.stable` and `.descending` options to `true`.
SortIntVector(&v, {.stable = true, .descending = true});
// Order can be different for arguments as well.
SortIntVector(&v, {.descending = true, .stable = true});
```

#### Destructuring in pattern matching

We might instead support destructuring struct patterns with defaults:

```
fn SortIntVector(
    v: Vector(Int)*,
    {stable: Bool = false, descending: Bool = false}) {
  // Code using `stable` and `descending`.
}
```

This would allow the same syntax at the call site, but avoids
[some concerns with field defaults](https://github.com/carbon-language/carbon-lang/pull/561#discussion_r683856715)
and allows some other use cases such as destructuring return values.

#### Discussion

We might support destructuring directly:

```
var {key: String, value: Int} = ReturnKeyValue();
```

or by way of a mechanism that converts a struct into a tuple:

```
var (key: String, value: Int) =
    ReturnKeyValue().extract(.key, .value);
// or maybe:
var (key: String, value: Int) =
    ReturnKeyValue()[(.key, .value)];
```

Similarly we might support optional named parameters directly instead of by way
of struct types.

Some discussion on this topic has occurred in:

-   [question-for-leads issue #505 on named parameters](https://github.com/carbon-language/carbon-lang/issues/505)
-   labeled params brainstorming docs
    [1](https://docs.google.com/document/d/1a1wI8SHGh3HYV8SUWPIKhg48ZW2glUlAMIIS3aec5dY/edit),
    [2](https://docs.google.com/document/d/1u6GORSkcgThMAiYKOqsgALcEviEtcghGb5TTVT-U-N0/edit)
-   ["match" in syntax choices doc](https://docs.google.com/document/d/1iuytei37LPg_tEd6xe-O6P_bpN7TIbEjNtFMLYW2Nno/edit#heading=h.y566d16ivoy2)

### Access control

We will need some way of controlling access to the members of classes. By
default, all members are fully publicly accessible, as decided in
[issue #665](https://github.com/carbon-language/carbon-lang/issues/665).

The set of access control options Carbon will support is an open question. Swift
and C++ (especially w/ modules) provide a lot of options and a pretty wide space
to explore here.

### Operator overloading

This includes destructors, copy and move operations, as well as other Carbon
operators such as `+` and `/`. We expect types to implement these operations by
implementing corresponding interfaces, see
[the generics overview](generics/overview.md).

### Inheritance

Carbon will need ways of saying:

-   this `class` type has a virtual method table
-   this `class` type extends a base type
-   this `class` type is "final" and may not be extended further
-   this method is "virtual" and may be overridden in descendents
-   this method is "pure virtual" or "abstract" and must be overridden in
    descendants
-   this method overrides a method declared in a base type

Multiple inheritance will be limited in at least a couple of ways:

-   At most one supertype may define data members.
-   Carbon types can't access data members of C++ virtual base classes.

There is a
[document considering the options for constructing objects with inheritance](https://docs.google.com/document/d/1GyrBIFyUbuLJGItmTAYUf9sqSDQjry_kjZKm5INl-84/edit).

### C++ abstract base classes interoperating with object-safe interfaces

We want four things so that Carbon's object-safe interfaces may interoperate
with C++ abstract base classes without data members, matching the
[interface as base class use case](#interface-as-base-class):

-   Ability to convert an object-safe interface (a type-of-type) into an
    C++-compatible base class (a base type), maybe using
    `AsBaseClass(MyInterface)`.
-   Ability to convert a C++ base class without data members (a base type) into
    an object-safe interface (a type-of-type), maybe using `AsInterface(MyIBC)`.
-   Ability to convert a (thin) pointer to an abstract base class to a `DynPtr`
    of the corresponding interface.
-   Ability to convert `DynPtr(MyInterface)` values to a proxy type that extends
    the corresponding base class `AsBaseType(MyInterface)`.

Note that the proxy type extending `AsBaseType(MyInterface)` would be a
different type than `DynPtr(MyInterface)` since the receiver input to the
function members of the vtable for the former does not match those in the
witness table for the latter.

### Mixins

We will need some way to declare mixins. This syntax will need a way to
distinguish defining versus requiring member variables. Methods may additionally
be given a default definition but may be overridden. Interface implementations
may only be partially provided by a mixin. Mixin methods will need to be able to
convert between pointers to the mixin type and the main type.

Open questions include whether a mixin is its own type that is a member of the
containing type, and whether mixins are templated on the containing type. Mixins
also complicate how constructors work.

### Non-virtual inheritance

We need some way of addressing two safety concerns created by non-virtual
inheritance:

-   Unclear how to safely run the right destructor.
-   [Object slicing](https://en.wikipedia.org/wiki/Object_slicing) is a danger
    when dereferencing a pointer of a base type.

These concerns would be resolved by distinguishing between pointers that point
to a specified type only and those that point to a type or any subtype. The
latter case would have restrictions to prevent misuse. This distinction may be
more complexity than is justified for a relatively rare use case. An alternative
approach would be to forbid destruction of non-final types without virtual
destructors, and forbid assignment of non-final types entirely.

This open question is being considered in
[question-for-leads issue #652](https://github.com/carbon-language/carbon-lang/issues/652).

### Memory layout

Carbon will need some way for users to specify the memory layout of class types
beyond simple ordering of fields, such as controlling the packing and alignment
for the whole type or individual members.

### No `static` variables

At the moment, there is no proposal to support
[`static` member variables](https://en.wikipedia.org/wiki/Class_variable#Static_member_variables_and_static_member_functions),
in line with avoiding global variables more generally. Carbon may need some
support in this area, though, for parity with and migration from C++.

### Computed properties

Carbon might want to support members of a type that are accessed like a data
member but return a computed value like a function. This has a number of
implications:

-   It would be a way of publicly exposing data members for
    [encapsulated types](#encapsulated-types), allowing for rules that otherwise
    forbid mixing public and private data members.
-   It would provide a more graceful evolution path from a
    [data class](#data-classes) to an [encapsulated type](#encapsulated-types).
-   It would give an option to start with a [data class](#data-classes) instead
    of writing all the boilerplate to create an
    [encapsulated type](#encapsulated-types) preemptively to allow future
    evolution.
-   It would let you take a variable away and put a property in its place with
    no other code changes. The number one use for this is so you can put a
    breakpoint in the property code, then later go back to public variable once
    you understand who was misbehaving.
-   We should have some guidance for when to use a computed property instead of
    a function with no arguments. One possible criteria is when it is a pure
    function of the state of the object and executes in an amount of time
    similar to ordinary member access.

However, there are likely to be differences between computed properties and
other data members, such as the ability to take the address of them. We might
want to support "read only" data members, that can be read through the public
api but only modified with private access, for data members which may need to
evolve into a computed property. There are also questions regarding how to
support assigning or modifying computed properties, such as using `+=`.

### Interfaces implemented for data classes

We should define a way for defining implementations of interfaces for struct
types. To satisfy coherence, these implementations would have to be defined in
the library with the interface definition. The syntax might look like:

```
interface ConstructWidgetFrom {
  fn Construct(Self) -> Widget;
}

external impl {.kind: WidgetKind, .size: Int}
    as ConstructWidgetFrom { ... }
```

In addition, we should define a way for interfaces to define templated blanket
implementations for [data classes](#data-classes) more generally. These
implementations will typically subject to the criteria that all the data fields
of the type must implement the interface. An example use case would be to say
that a data class is serializable if all of its fields were. For this we will
need a type-of-type for capturing that criteria, maybe something like
`DataFieldsImplement(MyInterface)`. The templated implementation will need some
way of iterating through the fields so it can perform operations fieldwise. This
feature should also implement the interfaces for any tuples whose fields satisfy
the criteria.

It is an open question how define implementations for binary operators. For
example, if `Int` is comparable to `Float32`, then `{.x = 3, .y = 2.72}` should
be comparable to `{.x = 3.14, .y = 2}`. The trick is how to declare the criteria
that "`T` is comparable to `U` if they have the same field names in the same
order, and for every field `x`, the type of `T.x` implements `ComparableTo` for
the type of `U.x`."
