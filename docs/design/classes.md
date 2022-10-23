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
-   [Members](#members)
    -   [Data members have an order](#data-members-have-an-order)
-   [Struct types](#struct-types)
    -   [Literals](#literals)
    -   [Type expression](#type-expression)
    -   [Assignment and initialization](#assignment-and-initialization)
    -   [Operations performed field-wise](#operations-performed-field-wise)
-   [Nominal class types](#nominal-class-types)
    -   [Forward declaration](#forward-declaration)
    -   [`Self`](#self)
    -   [Construction](#construction)
        -   [Assignment](#assignment)
    -   [Member functions](#member-functions)
        -   [Class functions](#class-functions)
        -   [Methods](#methods)
        -   [Name lookup in member function definitions](#name-lookup-in-member-function-definitions)
    -   [Nominal data classes](#nominal-data-classes)
    -   [Member type](#member-type)
    -   [Let](#let)
    -   [Alias](#alias)
    -   [Inheritance](#inheritance)
        -   [Virtual methods](#virtual-methods)
            -   [Virtual override keywords](#virtual-override-keywords)
        -   [Subtyping](#subtyping)
        -   [`Self` refers to the current type](#self-refers-to-the-current-type)
        -   [Constructors](#constructors)
            -   [Partial facet](#partial-facet)
            -   [Usage](#usage)
        -   [Assignment with inheritance](#assignment-with-inheritance)
    -   [Destructors](#destructors)
    -   [Access control](#access-control)
        -   [Private access](#private-access)
        -   [Protected access](#protected-access)
        -   [Friends](#friends)
        -   [Test friendship](#test-friendship)
        -   [Access control for construction](#access-control-for-construction)
    -   [Operator overloading](#operator-overloading)
-   [Future work](#future-work)
    -   [Struct literal shortcut](#struct-literal-shortcut)
    -   [Optional named parameters](#optional-named-parameters)
        -   [Field defaults for struct types](#field-defaults-for-struct-types)
        -   [Destructuring in pattern matching](#destructuring-in-pattern-matching)
        -   [Discussion](#discussion)
    -   [Inheritance](#inheritance-1)
        -   [C++ abstract base classes interoperating with object-safe interfaces](#c-abstract-base-classes-interoperating-with-object-safe-interfaces)
        -   [Overloaded methods](#overloaded-methods)
        -   [Interop with C++ inheritance](#interop-with-c-inheritance)
            -   [Virtual base classes](#virtual-base-classes)
    -   [Mixins](#mixins-1)
    -   [Memory layout](#memory-layout)
    -   [No `static` variables](#no-static-variables)
    -   [Computed properties](#computed-properties)
    -   [Interfaces implemented for data classes](#interfaces-implemented-for-data-classes)
-   [References](#references)

<!-- tocstop -->

## Overview

A Carbon _class_ is a user-defined
[record type](<https://en.wikipedia.org/wiki/Record_(computer_science)>). A
class has members that are referenced by their names, in contrast to a
[Carbon tuple](tuples.md) which defines a
[product type](https://en.wikipedia.org/wiki/Product_type) whose members are
referenced positionally.

Classes are the primary mechanism for users to extend the Carbon type system and
are deeply rooted in C++ and its history (C and Simula). We call them classes
rather than other terms as that is both familiar to existing programmers and
accurately captures their essence: they define the types of objects with
(optional) support for methods, encapsulation, and so on.

Carbon supports both named, or "nominal", and unnamed, anonymous, or
"structural", class types. Nominal class types are all distinct, but structural
types are equal if they have the same sequence of member types and names.
Structural class literals may be used to initialize or assign values to nominal
class variables.

A class type defines the interpretation of the bytes of a value of that type,
including the size, data members, and layout. It defines the operations that may
be performed on those values, including what methods may be called. A class type
may directly have constant members. The type itself is a compile-time immutable
constant value.

## Use cases

The use cases for classes include both cases motivated by C++ interop, and cases
that we expect to be included in idiomatic Carbon-only code.

**This design currently only attempts to address the "data classes" and
"encapsulated types" use cases.** Addressing the "interface as base class",
"interop with C++ multiple inheritance" and "mixin" use cases is future work.

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

These types are expected in idiomatic Carbon-only code.

#### With inheritance and subtyping

The [subtyping](https://en.wikipedia.org/wiki/Subtyping) you get with
inheritance is that you may assign the address of an object of a derived type to
a pointer to its base type. For this to work, the compiler needs implementation
strategies that allow operations performed through the pointer to the base type
work independent of which derived type it actually points to. These strategies
include:

-   Arranging for the data layout of derived types to start with the data layout
    of the base type as a prefix.
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
Extending this design to support interoperating with C++ types using multiple
inheritance is future work.

### Mixins

A [mixin](https://en.wikipedia.org/wiki/Mixin) is a declaration of data,
methods, and interface implementations that can be added to another type, called
the "main type". The methods of a mixin may also use data, methods, and
interface implementations provided by the main type. Mixins are designed around
implementation reuse rather than subtyping, and so don't need to use a vtable.

A mixin might be an implementation detail of a [data class](#data-classes), or
[encapsulated type](#encapsulated-types). A mixin might partially implement an
[interface as base class](#interface-as-base-class).

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

Note that struct types are examples of _data class types_ and are still classes.
The ["nominal data classes" section](#nominal-data-classes) describes another
way to define a data class type. Also note that there is no `struct` keyword,
"struct" is just convenient shorthand terminology for a structural data class.

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
    field is set to the type of the value, and so is `i32`.

Note: A comma `,` may optionally be included after the last field:

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
{.key: String, .value: i32}
```

This syntax is intended to parallel the literal syntax, and so uses commas (`,`)
to separate fields instead of a semicolon (`;`) terminator. This choice also
reflects the expected use inline in function signature declarations.

Struct types may only have data members, so the type declaration is just a list
of field names and types. The result of a struct type expression is an immutable
compile-time type value.

Note: Like with struct literal expressions, a comma `,` may optionally be
included after the last field:

```
{.key: String, .value: i32,}
```

Also note that `{}` represents both the empty struct literal and its type.

### Assignment and initialization

When initializing or assigning a variable with a data class such as a struct
type to a struct value on the right hand side, the order of the fields does not
have to match, just the names.

```
var different_order: {.x: i32, .y: i32} = {.y = 2, .x = 3};
Assert(different_order.x == 3);
Assert(different_order.y == 2);
```

Initialization and assignment occur field-by-field. The order of fields is
determined from the target on the left side of the `=`. This rule matches what
we expect for classes with encapsulation more generally.

**Open question:** What operations and in what order happen for assignment and
initialization?

-   Is assignment just destruction followed by initialization? Is that
    destruction completed for the whole object before initializing, or is it
    interleaved field-by-field?
-   When initializing to a literal value, is a temporary containing the literal
    value constructed first or are the fields initialized directly? The latter
    approach supports types that can't be moved or copied, such as mutex.
-   Perhaps some operations are _not_ ordered with respect to each other?

### Operations performed field-wise

Generally speaking, the operations that are available on a data class value,
such as a value with a struct type, are dependent on those operations being
available for all the types of the fields.

For example, two values of the same data class type may be compared for equality
or inequality if equality is supported for every member of the type:

```
var p: auto = {.x = 2, .y = 3};
Assert(p == {.x = 2, .y = 3});
Assert(p != {.x = 2, .y = 4});
Assert({.x = 2, .y = 4} != {.x = 5, .y = 3});
```

Equality and inequality comparisons are also allowed between different data
class types when:

-   At least one is a struct type.
-   They have the same set of field names, though the order may be different.
-   Equality comparison is defined between the pairs of member types with the
    same field names.

For example, since
[comparison between `i32` and `u32` is defined](/proposals/p0702.md#built-in-comparisons-and-implicit-conversions),
equality comparison between values of types `{.x: i32, .y: i32}` and
`{.y: u32, .x: u32}` is as well. Equality and inequality comparisons compare
fields using the field order of the left-hand operand and stop once the outcome
of the comparison is determined. However, the comparison order and
short-circuiting are generally expected to affect only the performance
characteristics of the comparison and not its meaning.

Ordering comparisons, such as `<` and `<=`, use the order of the fields to do a
[lexicographical comparison](https://en.wikipedia.org/wiki/Lexicographic_order).
The argument types must have a matching order of the field names. Otherwise, the
restrictions on ordering comparisons between different data class types are
analogous to equality comparisons:

-   At least one is a struct type.
-   Ordering comparison is defined between the pairs of member types with the
    same field names.

Implicit conversion from a struct type to a data class type is allowed when the
set of field names is the same and implicit conversion is defined between the
pairs of member types with the same field names. So calling a function
effectively performs an assignment from each of the caller's arguments to the
function's parameters, and will be valid when those assignments are all valid.

A data class has an unformed state if all its members do. Treatment of unformed
state follows proposal
[#257](https://github.com/carbon-language/carbon-lang/pull/257).

Destruction is performed field-wise in reverse order.

Extending user-defined operations on the fields to an operation on an entire
data class is [future work](#interfaces-implemented-for-data-classes).

**References:** The rules for assignment, comparison, and implicit conversion
for argument passing were decided in
[question-for-leads issue #710](https://github.com/carbon-language/carbon-lang/issues/710).

## Nominal class types

The declarations for nominal class types will have:

-   an optional `abstract` or `base` prefix
-   `class` introducer
-   the name of the class
-   an optional `extends` followed by the name of the immediate base class
-   `{`, an open curly brace
-   a sequence of declarations
-   `}`, a close curly brace

Declarations should generally match declarations that can be declared in other
contexts, for example variable declarations with `var` will define
[instance variables](https://en.wikipedia.org/wiki/Instance_variable):

```
class TextLabel {
  var x: i32;
  var y: i32;

  var text: String = "default";
}
```

The main difference here is that `"default"` is a default instead of an
initializer, and will be ignored if another value is supplied for that field
when constructing a value. Defaults must be constants whose value can be
determined at compile time.

### Forward declaration

To support circular references between class types, we allow
[forward declaration](https://en.wikipedia.org/wiki/Forward_declaration) of
types. Forward declarations end with semicolon `;` after the name of the class,
instead of any `extends` clause and the block of declarations in curly braces
`{`...`}`. A type that is forward declared is considered incomplete until the
end of a definition with the same name.

```
// Forward declaration of `GraphNode`.
class GraphNode;

class GraphEdge {
  var head: GraphNode*;
  var tail: GraphNode*;
}

class GraphNode {
  var edges: Vector(GraphEdge*);
}
// `GraphNode` is first complete here.
```

**Open question:** What is specifically allowed and forbidden with an incomplete
type has not yet been decided.

### `Self`

A `class` definition may provisionally include references to its own name in
limited ways. These limitations arise from the type not being complete until the
end of its definition is reached.

```
class IntListNode {
  var data: i32;
  var next: IntListNode*;
}
```

An equivalent definition of `IntListNode`, since the `Self` keyword is an alias
for the current type, is:

```
class IntListNode {
  var data: i32;
  var next: Self*;
}
```

`Self` refers to the innermost type declaration:

```
class IntList {
  class IntListNode {
    var data: i32;
    // `Self` is `IntListNode`, not `IntList`.
    var next: Self*;
  }
  var first: IntListNode*;
}
```

### Construction

Any function with access to all the data fields of a class can construct one by
converting a [struct value](#struct-types) to the class type:

```
var tl1: TextLabel = {.x = 1, .y = 2};
var tl2: auto = {.x = 1, .y = 2} as TextLabel;

Assert(tl1.x == tl2.x);

fn ReturnsATextLabel() -> TextLabel {
  return {.x = 1, .y = 2};
}
var tl3: TextLabel = ReturnsATextLabel();

fn AcceptsATextLabel(tl: TextLabel) -> i32 {
  return tl.x + tl.y;
}
Assert(AcceptsATextLabel({.x = 2, .y = 4}) == 6);
```

Note that a nominal class, unlike a [struct type](#type-expression), can define
default values for fields, and so may be initialized with a
[struct value](#literals) that omits some or all of those fields.

#### Assignment

Assignment to a struct value is also allowed in a function with access to all
the data fields of a class. Assignment always overwrites all of the field
members.

```
var tl: TextLabel = {.x = 1, .y = 2};
Assert(tl.text == "default");

// ✅ Allowed: assigns all fields
tl = {.x = 3, .y = 4, .text = "new"};

// ✅ Allowed: This statement is evaluated in two steps:
// 1. {.x = 5, .y = 6} is converted into a new TextLabel value,
//    using default for field `text`.
// 2. tl is assigned to a TextLabel, which has values for all
//    fields.
tl = {.x = 5, .y = 6};
Assert(tl.text == "default");
```

**Open question:** This behavior might be surprising because there is an
ambiguity about whether to use the default value or the previous value for a
field. We could require all fields to be specified when assigning, and only use
field defaults when initializing a new value.

```
// ❌ Forbidden: should tl.text == "default" or "new"?
tl = {.x = 5, .y = 6};
```

### Member functions

Member functions can either be class functions or methods. Class functions are
members of the type, while methods can only be called on instances.

#### Class functions

A class function is like a
[C++ static member function](https://en.cppreference.com/w/cpp/language/static#Static_member_functions),
and is declared like a function at file scope. The declaration can include a
definition of the function body, or that definition can be provided out of line
after the class definition is finished. A common use is for constructor
functions.

```
class Point {
  fn Origin() -> Self {
    return {.x = 0, .y = 0};
  }
  fn CreateCentered() -> Self;

  var x: i32;
  var y: i32;
}

fn Point.CreateCentered() -> Self {
  return {.x = ScreenWidth() / 2, .y = ScreenHeight() / 2};
}
```

Class functions are members of the type, and may be accessed as using dot `.`
member access either the type or any instance.

```
var p1: Point = Point.Origin();
var p2: Point = p1.CreateCentered();
```

#### Methods

[Method](<https://en.wikipedia.org/wiki/Method_(computer_programming)>)
declarations are distinguished from [class function](#class-functions)
declarations by having a `me` parameter in square brackets `[`...`]` before the
explicit parameter list in parens `(`...`)`. There is no implicit member access
in methods, so inside the method body members are accessed through the `me`
parameter. Methods may be written lexically inline or after the class
declaration.

```carbon
class Circle {
  fn Diameter[me: Self]() -> f32 {
    return me.radius * 2;
  }
  fn Expand[addr me: Self*](distance: f32);

  var center: Point;
  var radius: f32;
}

fn Circle.Expand[addr me: Self*](distance: f32) {
  me->radius += distance;
}

var c: Circle = {.center = Point.Origin(), .radius = 1.5 };
Assert(Math.Abs(c.Diameter() - 3.0) < 0.001);
c.Expand(0.5);
Assert(Math.Abs(c.Diameter() - 4.0) < 0.001);
```

-   Methods are called using the dot `.` member syntax, `c.Diameter()` and
    `c.Expand(`...`)`.
-   `Diameter` computes and returns the diameter of the circle without modifying
    the `Circle` instance. This is signified using `[me: Self]` in the method
    declaration.
-   `c.Expand(`...`)` does modify the value of `c`. This is signified using
    `[addr me: Self*]` in the method declaration.

The pattern '`addr` _patt_' means "first take the address of the argument, which
must be an
[l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>), and
then match pattern _patt_ against it".

If the method declaration also includes
[deduced generic parameters](/docs/design/generics/overview.md#deduced-parameters),
the `me` parameter must be in the same list in square brackets `[`...`]`. The
`me` parameter may appear in any position in that list, as long as it appears
after any names needed to describe its type.

#### Name lookup in member function definitions

When defining a member function lexically inline, we delay type checking of the
function body until the definition of the current type is complete. This means
that name lookup _for members of objects_ is also delayed. That means that you
can reference `me.F()` in a lexically inline method definition even before the
declaration of `F` in that class definition. However, other names still need to
be declared before they are used. This includes unqualified names, names within
namespaces, and names _for members of types_.

```
class Point {
  fn Distance[me: Self]() -> f32 {
    // ✅ Allowed: `x` and `y` are names for members of an object,
    // and so lookup is delayed until `type_of(me) == Self` is complete.
    return Math.Sqrt(me.x * me.x + me.y * me.y);
  }

  fn CreatePolarInvalid(r: f32, theta: f32) -> Point {
    // ❌ Forbidden: unqualified name used before declaration.
    return Create(r * Math.Cos(theta), r * Math.Sin(theta));
  }
  fn CreatePolarValid1(r: f32, theta: f32) -> Point {
    // ❌ Forbidden: `Create` is not yet declared.
    return Point.Create(r * Math.Cos(theta), r * Math.Sin(theta));
  }
  fn CreatePolarValid2(r: f32, theta: f32) -> Point {
    // ❌ Forbidden: `Create` is not yet declared.
    return Self.Create(r * Math.Cos(theta), r * Math.Sin(theta));
  }

  fn Create(x: f32, y: f32) -> Point {
    // ✅ Allowed: checking that conversion of `{.x: f32, .y: f32}`
    // to `Point` is delayed until `Point` is complete.
    return {.x = x, .y = y};
  }

  fn CreateXEqualsY(xy: f32) -> Point {
    // ✅ Allowed: `Create` is declared earlier.
    return Create(xy, xy);
  }

  fn CreateXAxis(x: f32) -> Point;

  fn Angle[me: Self]() -> f32;

  var x: f32;
  var y: f32;
}

fn Point.CreateXAxis(x: f32) -> Point {
  // ✅ Allowed: `Point` type is complete.
  // Members of `Point` like `Create` are in scope.
  return Create(x, 0);
}

fn Point.Angle[me: Self]() -> f32 {
  // ✅ Allowed: `Point` type is complete.
  // Function is checked immediately.
  return Math.ATan2(me.y, me.x);
}
```

**Note:** The details of name lookup are still being decided in issue
[#472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472).

### Nominal data classes

We will mark [data classes](#data-classes) with an `impl as Data {}` line.

```
class TextLabel {
  var x: i32;
  var y: i32;

  var text: String;

  // This line makes `TextLabel` a data class, which defines
  // a number of operations field-wise.
  impl as Data {}
}
```

The fields of data classes must all be public. That line will add
[field-wise implementations and operations of all interfaces that a struct with the same fields would get by default](#operations-performed-field-wise).

The word `Data` here refers to an empty interface in the Carbon prologue. That
interface would then be part of our
[strategy for defining how other interfaces are implemented for data classes](#interfaces-implemented-for-data-classes).

**References:** Rationale for this approach is given in proposal
[#722](/proposals/p0722.md#nominal-data-class).

### Member type

Additional types may be defined in the scope of a class definition.

```
class StringCounts {
  class Node {
    var key: String;
    var count: i32;
  }
  var counts: Vector(Node);
}
```

The inner type is a member of the type, and is given the name
`StringCounts.Node`. This case is called a _member class_ since the type is a
class, but other kinds of type declarations, like choice types, are allowed.

### Let

Other type constants can be defined using a `let` declaration:

```
class MyClass {
  let Pi:! f32 = 3.141592653589793;
  let IndexType:! Type = i32;
}
```

The `:!` indicates that this is defining a compile-time constant, and so does
not affect the storage of instances of that class.

### Alias

You may declare aliases of the names of class members. This is to allow them to
be renamed in multiple steps or support alternate names.

```
class StringPair {
  var key: String;
  var value: String;
  alias first = key;
  alias second = value;
}

var sp1: StringPair = {.key = "K", .value = "1"};
var sp2: StringPair = {.first = "K", .second = "2"};
Assert(sp1.first == sp2.key);
Assert(&sp1.first == &sp1.key);
```

**Future work:** This needs to be connected to the broader design of aliases,
once that lands.

### Inheritance

Carbon supports
[inheritance](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>)
using a
[class hierarchy](<https://en.wikipedia.org/wiki/Class_(computer_programming)#Hierarchical>),
on an opt-in basis. Classes by default are
[_final_](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Non-subclassable_classes>),
which means they may not be extended. To declare a class as allowing extension,
use either the `base class` or `abstract class` introducer:

```
base class MyBaseClass { ... }
```

A _base class_ may be _extended_ to get a _derived class_:

```
base class MiddleDerived extends MyBaseClass { ... }
class FinalDerived extends MiddleDerived { ... }
// ❌ Forbidden: class Illegal extends FinalDerived { ... }
```

An _[abstract class](https://en.wikipedia.org/wiki/Abstract_type)_ or _abstract
base class_ is a base class that may not be instantiated.

```
abstract class MyAbstractClass { ... }
// ❌ Forbidden: var a: MyAbstractClass = ...;
```

**Future work:** For now, the Carbon design only supports single inheritance. In
the future, Carbon will support multiple inheritance with limitations on all
base classes except the one listed first.

**Terminology:** We say `MiddleDerived` and `FinalDerived` are _derived
classes_, transitively extending or _derived from_ `MyBaseClass`. Similarly
`FinalDerived` is derived from or extends `MiddleDerived`. `MiddleDerived` is
`FinalDerived`'s _immediate base class_, and both `MiddleDerived` and
`MyBaseClass` are base classes of `FinalDerived`. Base classes that are not
abstract are called _extensible classes_.

A derived class has all the members of the class it extends, including data
members and methods, though it may not be able to access them if they were
declared `private`.

#### Virtual methods

A base class may define
[virtual methods](https://en.wikipedia.org/wiki/Virtual_function). These are
methods whose implementation may be overridden in a derived class.

Only methods defined in the scope of the class definition may be virtual, not
any defined in
[external interface impls](/docs/design/generics/details.md#external-impl).
Interface methods may be implemented using virtual methods when the
[impl is internal](/docs/design/generics/details.md#implementing-interfaces),
and calls to those methods by way of the interface will do virtual dispatch just
like a direct call to the method does.

[Class functions](#class-functions) may not be declared virtual.

##### Virtual override keywords

A method is declared as virtual by using a _virtual override keyword_ in its
declaration before `fn`.

```
base class MyBaseClass {
  virtual fn Overridable[me: Self]() -> i32 { return 7; }
}
```

This matches C++, and makes it relatively easy for authors of derived classes to
find the functions that can be overridden.

If no keyword is specified, the default for methods is that they are
_non-virtual_. This means:

-   they can't override methods in bases of this class;
-   they can't be overridden in derived classes; and
-   they have an implementation in the current class, and that implementation
    must work for all derived classes.

There are three virtual override keywords:

-   `virtual` - This marks a method as not present in bases of this class and
    having an implementation in this class. That implementation may be
    overridden in derived classes.
-   `abstract` - This marks a method that must be overridden in a derived class
    since it has no implementation in this class. This is short for "abstract
    virtual" but is called
    ["pure virtual" in C++](https://en.wikipedia.org/wiki/Virtual_function#Abstract_classes_and_pure_virtual_functions).
    Only abstract classes may have unimplemented abstract methods.
-   `impl` - This marks a method that overrides a method marked `virtual` or
    `abstract` in the base class with an implementation specific to -- and
    defined within -- this class. The method is still virtual and may be
    overridden again in subsequent derived classes if this is a base class. See
    [method overriding in Wikipedia](https://en.wikipedia.org/wiki/Method_overriding).
    Requiring a keyword when overriding allows the compiler to diagnose when the
    derived class accidentally uses the wrong signature or spelling and so
    doesn't match the base class. We intentionally use the same keyword here as
    for implementing interfaces, to emphasize that they are similar operations.

| Keyword on<br />method in `C` | Allowed in<br />`abstract class C` | Allowed in<br />`base class C` | Allowed in<br />final `class C` | in `B` where<br />`C extends B`                          | in `D` where<br />`D extends C`                                                  |
| ----------------------------- | ---------------------------------- | ------------------------------ | ------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `virtual`                     | ✅                                 | ✅                             | ❌                              | _not present_                                            | `abstract`<br />`impl`<br />_not mentioned_                                      |
| `abstract`                    | ✅                                 | ❌                             | ❌                              | _not present_<br />`virtual`<br />`abstract`<br />`impl` | `abstract`<br />`impl`<br />_may not be<br />mentioned if<br />`D` is not final_ |
| `impl`                        | ✅                                 | ✅                             | ✅                              | `virtual`<br />`abstract`<br />`impl`                    | `abstract`<br />`impl`                                                           |

#### Subtyping

A pointer to a base class, like `MyBaseClass*` is actually considered to be a
pointer to that type or any derived class, like `MiddleDerived` or
`FinalDerived`. This means that a `FinalDerived*` value may be implicitly cast
to type `MiddleDerived*` or `MyBaseClass*`.

This is accomplished by making the data layout of a type extending `MyBaseClass`
have `MyBaseClass` as a prefix. In addition, the first class in the inheritance
chain with a virtual method will include a virtual pointer, or _vptr_, pointing
to a [virtual method table](https://en.wikipedia.org/wiki/Virtual_method_table),
or _vtable_. Any calls to virtual methods will perform
[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch) by calling
the method using the function pointer in the vtable, to get the overridden
implementation from the most derived class that implements the method.

Since a final class may not be extended, the compiler can bypass the vtable and
use [static dispatch](https://en.wikipedia.org/wiki/Static_dispatch). In
general, you can use a combination of an abstract base class and a final class
instead of an extensible class if you need to distinguish between exactly a type
and possibly a subtype.

```
base class Extensible { ... }

// Can be replaced by:

abstract class ExtensibleBase { ... }
class ExactlyExtensible extends ExtensibleBase { ... }
```

#### `Self` refers to the current type

Note that `Self` in a class definition means "the current type being defined"
not "the type implementing this method." To implement a method in a derived
class that uses `Self` in the declaration in the base class, only the type of
`me` should change:

```
base class B1 {
  virtual fn F[me: Self](x: Self) -> Self;
  // Means exactly the same thing as:
  //   virtual fn F[me: B1](x: B1) -> B1;
}

class D1 extends B1 {
  // ❌ Illegal:
  //   impl fn F[me: Self](x: Self) -> Self;
  // since that would mean the same thing as:
  //   impl fn F[me: Self](x: D1) -> D1;
  // and `D1` is a different type than `B1`.

  // ✅ Allowed: Parameter and return types
  //  of `F` match declaration in `B1`.
  impl fn F[me: Self](x: B1) -> B1;
  // Or: impl fn F[me: D1](x: B1) -> B1;
}
```

The exception is when there is a [subtyping relationship](#subtyping) such that
it would be legal for a caller using the base classes signature to actually be
calling the derived implementation, as in:

```
base class B2 {
  virtual fn Clone[me: Self]() -> Self*;
  // Means exactly the same thing as:
  //   virtual fn Clone[me: B2]() -> B2*;
}

class D2 extends B2 {
  // ✅ Allowed
  impl fn Clone[me: Self]() -> Self*;
  // Means the same thing as:
  //   impl fn Clone[me: D2]() -> D2*;
  // which is allowed since `D2*` is a
  // subtype of `B2*`.
}
```

#### Constructors

Like for classes without inheritance, constructors for a derived class are
ordinary functions that return an instance of the derived class. Generally
constructor functions should return the constructed value without copying, as in
proposal
[#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257).
This means either
[creating the object in the return statement itself](/proposals/p0257.md#function-returns-and-initialization),
or in
[a `returned var` declaration](/proposals/p0257.md#declared-returned-variable).
As before, instances can be created using by casting a struct value into the
class type, this time with a `.base` member to initialize the members of the
immediate base type.

```
class MyDerivedType extends MyBaseType {
  fn Create() -> MyDerivedType {
    return {.base = MyBaseType.Create(), .derived_field = ...};
  }
}
```

There are two cases that aren't well supported with this pattern:

-   Users cannot create a value of an abstract class, which is necessary when it
    has private fields or otherwise requires initialization.
-   Users may want to reduce the chance of mistakes from calling a method on a
    partially constructed object. Of particular concern is calling a virtual
    method prior to forming the derived class and so it uses the base class
    implementation.

While expected to be relatively rarely needed, we will address both of these
concerns with a specialized type just used during construction of base classes,
called the partial facet type for the class.

##### Partial facet

The partial facet for a base class type like `MyBaseType` is written
`partial MyBaseType`.

-   Only methods that take the partial facet type may be called on the partial
    facet type, so methods have to opt in to being called on an object that
    isn't fully constructed.
-   No virtual methods may take the partial facet type, so there is no way to
    transitively call a virtual method on an object that isn't fully
    constructed.
-   `partial MyBaseClass` and `MyBaseClass` have the same fields in the same
    order with the same data layout. The only difference is that
    `partial MyBaseClass` doesn't use (look into) its hidden vptr slot. To
    reliably catch any bugs where virtual function calls occur in this state,
    both fast and hardened release builds will initialize the hidden vptr slot
    to a null pointer. Debug builds will initialize it to an alternate vtable
    whose functions will abort the program with a clear diagnostic.
-   Since `partial MyBaseClass` has the same data layout but only uses a subset,
    there is a subtyping relationship between these types. A `MyBaseClass` value
    is a `partial MyBaseClass` value, but not the other way around. So you can
    cast `MyBaseClass*` to `partial MyBaseClass*`, but the other direction is
    not safe.
-   When `MyBaseClass` may be instantiated, there is a conversion from
    `partial MyBaseClass` to `MyBaseClass`. It changes the value by filling in
    the hidden vptr slot. If `MyBaseClass` is abstract, then attempting that
    conversion is an error.
-   `partial MyBaseClass` is considered final, even if `MyBaseClass` is not.
    This is despite the fact that from a data layout perspective,
    `partial MyDerivedClass` will have `partial MyBaseClass` as a prefix if
    `MyDerivedClass` extends `MyBaseClass`. The type `partial MyBaseClass`
    specifically means "exactly this and no more." This means we don't need to
    look at the hidden vptr slot, and we can instantiate it even if it doesn't
    have a virtual [destructor](#destructors).
-   The keyword `partial` may only be applied to a base class. For final
    classes, there is no need for a second type.

##### Usage

The general pattern is that base classes can define constructors returning the
partial facet type.

```
base class MyBaseClass {
  fn Create() -> partial Self {
    return {.base_field_1 = ..., .base_field_2 = ...};
  }
  // ...
}
```

Extensible classes can be instantiated even from a partial facet value:

```
var mbc: MyBaseClass = MyBaseClass.Create();
```

The conversion from `partial MyBaseClass` to `MyBaseClass` only fills in the
vptr value and can be done in place. After the conversion, all public methods
may be called, including virtual methods.

The partial facet is required for abstract classes, since otherwise they may not
be instantiated. Constructor functions for abstract classes should be marked
[protected](#protected-access) so they may only be accessed in derived classes.

```
abstract class MyAbstractClass {
  protected fn Create() -> partial Self {
    return {.base_field_1 = ..., .base_field_2 = ...};
  }
  // ...
}
// ❌ Error: can't instantiate abstract class
var abc: MyAbstractClass = ...;
```

If a base class wants to store a pointer to itself somewhere in the constructor
function, there are two choices:

-   An extensible class could use the plain type instead of the partial facet.

    ```
    base class MyBaseClass {
      fn Create() -> Self {
        returned var result: Self = {...};
        StoreMyPointerSomewhere(&result);
        return var;
      }
    }
    ```

-   The other choice is to explicitly cast the type of its address. This pointer
    should not be used to call any virtual method until the object is finished
    being constructed, since the vptr will be null.

    ```
    abstract class MyAbstractClass {
      protected fn Create() -> partial Self {
        returned var result: partial Self = {...};
        // Careful! Pointer to object that isn't fully constructed!
        StoreMyPointerSomewhere(&result as Self*);
        return var;
      }
    }
    ```

The constructor for a derived class may construct values from a partial facet of
the class' immediate base type or the full type:

```
abstract class MyAbstractClass {
  protected fn Create() -> partial Self { ... }
}

// Base class returns a partial type
base class Derived extends MyAbstractClass {
  protected fn Create() -> partial Self {
    return {.base = MyAbstractClass.Create(), .derived_field = ...};
  }
  ...
}

base class MyBaseClass {
  fn Create() -> Self { ... }
}

// Base class returns a full type
base class ExtensibleDerived extends MyBaseClass {
  fn Create() -> Self {
    return {.base = MyBaseClass.Create(), .derived_field = ...};
  }
  ...
}
```

And final classes will return a type that does not use the partial facet:

```
class FinalDerived extends MiddleDerived {
  fn Create() -> Self {
    return {.base = MiddleDerived.Create(), .derived_field = ...};
  }
  ...
}
```

Observe that the vptr is only assigned twice in release builds if you use
partial facets:

-   The first class value created, by the factory function creating the base of
    the class hierarchy, initialized the vptr field to nullptr. Every derived
    type transitively created from that value will leave it alone.
-   Only when the value has its most-derived class and is converted from the
    partial facet type to its final type is the vptr field set to its final
    value.

In the case that the base class can be instantiated, tooling could optionally
recommend that functions returning `Self` that are used to initialize a derived
class be changed to return `partial Self` instead. However, the consequences of
returning `Self` instead of `partial Self` when the value will be used to
initialize a derived class are fairly minor:

-   The vptr field will be assigned more than necessary.
-   The types won't protect against calling methods on a value while it is being
    constructed, much like the situation in C++ currently.

#### Assignment with inheritance

Since the assignment operator method should not be virtual, it is only safe to
implement it for final types. However, following the
[maxim that Carbon should "focus on encouraging appropriate usage of features rather than restricting misuse"](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
we allow users to also implement assignment on extensible classes, even though
it can lead to [slicing](https://en.wikipedia.org/wiki/Object_slicing).

### Destructors

Every non-abstract type is _destructible_, meaning has a defined destructor
function called when the lifetime of a value of that type ends, such as when a
variable goes out of scope. The destructor for a class may be customized using
the `destructor` keyword:

```carbon
class MyClass {
  destructor [me: Self] { ... }
}
```

or:

```carbon
class MyClass {
  // Can modify `me` in the body.
  destructor [addr me: Self*] { ... }
}
```

If a class has no `destructor` declaration, it gets the default destructor,
which is equivalent to `destructor [me: Self] { }`.

The destructor for a class is run before the destructors of its data members.
The data members are destroyed in reverse order of declaration. Derived classes
are destroyed before their base classes, so the order of operations is:

-   derived class' destructor runs,
-   the data members of the derived class are destroyed, in reverse order of
    declaration,
-   the immediate base class' destructor runs,
-   the data members of the immediate base class are destroyed, in reverse order
    of declaration,
-   and so on.

Destructors may be declared in class scope and then defined out-of-line:

```carbon
class MyClass {
  destructor [addr me: Self*];
}
destructor MyClass [addr me: Self*] { ... }
```

It is illegal to delete an instance of a derived class through a pointer to one
of its base classes unless it has a
[virtual destructor](https://en.wikipedia.org/wiki/Virtual_function#Virtual_destructors).
An abstract or base class' destructor may be declared virtual using the
`virtual` introducer, in which case any derived class destructor declaration
must be `impl`:

```carbon
base class MyBaseClass {
  virtual destructor [addr me: Self*] { ... }
}

class MyDerivedClass extends MyBaseClass {
  impl destructor [addr me: Self*] { ... }
}
```

The properties of a type, whether type is abstract, base, or final, and whether
the destructor is virtual or non-virtual, determines which
[type-of-types](/docs/design/generics/terminology.md#type-of-type) it satisfies.

-   Non-abstract classes are `Concrete`. This means you can create local and
    member variables of this type. `Concrete` types have destructors that are
    called when the local variable goes out of scope or the containing object of
    the member variable is destroyed.
-   Final classes and classes with a virtual destructor are `Deletable`. These
    may be safely deleted through a pointer.
-   Classes that are `Concrete`, `Deletable`, or both are `Destructible`. These
    are types that may be deleted through a pointer, but it might not be safe.
    The concerning situation is when you have a pointer to a base class without
    a virtual destructor. It is unsafe to delete that pointer when it is
    actually pointing to a derived class.

**Note:** The names `Deletable` and `Destructible` are
[**placeholders**](/proposals/p1154.md#type-of-type-naming) since they do not
conform to the decision on
[question-for-leads issue #1058: "How should interfaces for core functionality be named?"](https://github.com/carbon-language/carbon-lang/issues/1058).

| Class    | Destructor  | `Concrete` | `Deletable` | `Destructible` |
| -------- | ----------- | ---------- | ----------- | -------------- |
| abstract | non-virtual | no         | no          | no             |
| abstract | virtual     | no         | yes         | yes            |
| base     | non-virtual | yes        | no          | yes            |
| base     | virtual     | yes        | yes         | yes            |
| final    | any         | yes        | yes         | yes            |

The compiler automatically determines which of these
[type-of-types](/docs/design/generics/terminology.md#type-of-type) a given type
satisfies. It is illegal to directly implement `Concrete`, `Deletable`, or
`Destructible` directly. For more about these constraints, see
["destructor constraints" in the detailed generics design](/docs/design/generics/details.md#destructor-constraints).

A pointer to `Deletable` types may be passed to the `Delete` method of the
`Allocator` [interface](/docs/design/generics/terminology.md#interface). To
deallocate a pointer to a base class without a virtual destructor, which may
only be done when it is not actually pointing to a value with a derived type,
call the `UnsafeDelete` method instead. Note that you may not call
`UnsafeDelete` on abstract types without virtual destructors, it requires
`Destructible`.

```
interface Allocator {
  // ...
  fn Delete[T:! Deletable, addr me: Self*](p: T*);
  fn UnsafeDelete[T:! Destructible, addr me: Self*](p: T*);
}
```

To pass a pointer to a base class without a virtual destructor to a generic
function expecting a `Deletable` type, use the `UnsafeAllowDelete`
[type adapter](/docs/design/generics/details.md#adapting-types).

```
adapter UnsafeAllowDelete(T:! Concrete) extends T {
  impl as Deletable {}
}

// Example usage:
fn RequiresDeletable[T:! Deletable](p: T*);
var x: MyExtensible;
RequiresDeletable(&x as UnsafeAllowDelete(MyExtensible)*);
```

If a virtual method is transitively called from inside a destructor, the
implementation from the current class is used, not any overrides from derived
classes. It will abort the execution of the program if that method is abstract
and not implemented in the current class.

**Future work:** Allow or require destructors to be declared as taking
`partial Self` in order to prove no use of virtual methods.

Types satisfy the
[`TrivialDestructor`](/docs/design/generics/details.md#destructor-constraints)
type-of-type if:

-   the class declaration does not define a destructor or the class defines the
    destructor with an empty body `{ }`,
-   all data members implement `TrivialDestructor`, and
-   all base classes implement `TrivialDestructor`.

For example, a [struct type](#struct-types) implements `TrivialDestructor` if
all its members do.

`TrivialDestructor` implies that their destructor does nothing, which may be
used to generate optimized specializations.

There is no provision for handling failure in a destructor. All operations that
could potentially fail must be performed before the destructor is called.
Unhandled failure during a destructor call will abort the program.

**Future work:** Allow or require destructors to be declared as taking
`[var me: Self]`.

**Alternatives considered:**

-   [Types implement destructor interface](/proposals/p1154.md#types-implement-destructor-interface)
-   [Prevent virtual function calls in destructors](/proposals/p1154.md#prevent-virtual-function-calls-in-destructors)
-   [Allow functions to act as destructors](/proposals/p1154.md#allow-functions-to-act-as-destructors)
-   [Allow private destructors](/proposals/p1154.md#allow-private-destructors)
-   [Allow multiple conditional destructors](/proposals/p1154.md#allow-multiple-conditional-destructors)
-   [Don't distinguish safe and unsafe delete operations](/proposals/p1154.md#dont-distinguish-safe-and-unsafe-delete-operations)
-   [Don't allow unsafe delete](/proposals/p1154.md#dont-allow-unsafe-delete)
-   [Allow final destructors](/proposals/p1154.md#allow-final-destructors)

### Access control

By default, all members of a class are fully publicly accessible. Access can be
restricted by adding a keyword, called an
[access modifier](https://en.wikipedia.org/wiki/Access_modifiers), prior to the
declaration. Access modifiers are how Carbon supports
[encapsulation](#encapsulated-types).

The [access modifier](https://en.wikipedia.org/wiki/Access_modifiers) is written
before any [virtual override keyword](#virtual-override-keywords).

**Rationale:** Carbon makes members public by default for a few reasons:

-   The readability of public members is the most important, since we expect
    most readers to be concerned with the public API of a type.
-   The members that are most commonly private are the data fields, which have
    relatively less complicated definitions that suffer less from the extra
    annotation.

Additionally, there is precedent for this approach in modern object-oriented
languages such as
[Kotlin](https://kotlinlang.org/docs/visibility-modifiers.html) and
[Python](https://docs.python.org/3/tutorial/classes.html), both of which are
well regarded for their usability.

Keywords controlling visibility are attached to individual declarations instead
of C++'s approach of labels controlling the visibility for all following
declarations to
[reduce context sensitivity](/docs/project/principles/low_context_sensitivity.md).
This matches
[Rust](https://doc.rust-lang.org/reference/visibility-and-privacy.html),
[Swift](https://docs.swift.org/swift-book/LanguageGuide/AccessControl.html),
[Java](http://rosettacode.org/wiki/Classes#Java),
[C#](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/access-modifiers),
[Kotlin](https://kotlinlang.org/docs/visibility-modifiers.html#classes-and-interfaces),
and [D](https://wiki.dlang.org/Access_specifiers_and_visibility).

**References:** Proposal
[#561: Basic classes](https://github.com/carbon-language/carbon-lang/pull/561)
included the decision that
[members default to publicly accessible](/proposals/p0561.md#access-control)
originally asked in issue
[#665](https://github.com/carbon-language/carbon-lang/issues/665).

#### Private access

As in C++, `private` means only accessible to members of the class and any
[friends](#friends).

```carbon
class Point {
  fn Distance[me: Self]() -> f32;
  // These are only accessible to members of `Point`.
  private var x: f32;
  private var y: f32;
}
```

A `private virtual` or `private abstract` method may be implemented in derived
classes, even though it may not be called. This allows derived classes to
customize the behavior of a function called by a method of the base class, while
still preventing the derived class from calling it. This matches the behavior of
C++ and is more orthogonal.

**Future work:** `private` will give the member internal linkage unless it needs
to be external because it is used in an inline method or template. We may in the
future
[add a way to specify internal linkage explicitly](/proposals/p0722.md#specifying-linkage-as-part-of-the-access-modifier).

**Open questions:** Using `private` to mean "restricted to this class" matches
C++. Other languages support restricting to different scopes:

-   Swift supports "restrict to this module" and "restrict to this file".
-   Rust supports "restrict to this module and any children of this module", as
    well as "restrict to this crate", "restrict to parent module", and "restrict
    to a specific ancestor module".

**Comparison to other languages:** C++, Rust, and Swift all make class members
private by default. C++ offers the `struct` keyword that makes members public by
default.

#### Protected access

Protected members may only be accessed by members of this class, members of
derived classes, and any [friends](#friends).

```
base class MyBaseClass {
  protected fn HelperClassFunction(x: i32) -> i32;
  protected fn HelperMethod[me: Self](x: i32) -> i32;
  protected var data: i32;
}

class MyDerivedClass extends MyBaseClass {
  fn UsesProtected[addr me: Self*]() {
    // Can access protected members in derived class
    var x: i32 = HelperClassFunction(3);
    me->data = me->HelperMethod(x);
  }
}
```

#### Friends

Classes may have a _friend_ declaration:

```
class Buddy { ... }

class Pal {
  private var x: i32;
  friend Buddy;
}
```

This declares `Buddy` to be a friend of `Pal`, which means that `Buddy` can
access all members of this class, even the ones that are declared `private` or
`protected`.

The `friend` keyword is followed by the name of an existing function, type, or
parameterized family of types. Unlike C++, it won't act as a forward declaration
of that name. The name must be resolvable by the compiler, and so may not be a
member of a template.

#### Test friendship

**Future work:** There should be a convenient way of allowing tests in the same
library as the class definition to access private members of the class. Ideally
this could be done without changing the class definition itself, since it
doesn't affect the class' public API.

#### Access control for construction

A function may construct a class, by casting a struct value to the class type,
if it has access to (write) all of its fields.

**Future work:** There should be a way to limit which code can construct a class
even when it only has public fields. This will be resolved in question-for-leads
issue [#803](https://github.com/carbon-language/carbon-lang/issues/803).

### Operator overloading

Developers may define how standard Carbon operators, such as `+` and `/`, apply
to custom types by implementing the
[interface](generics/terminology.md#interface) that corresponds to that operator
for the types of the operands. See the
["operator overloading" section](generics/details.md#operator-overloading) of
the [generics design](generics/overview.md). The specific interface used for a
given operator may be found in the
[expressions design](/docs/design/expressions/README.md).

## Future work

This includes features that need to be designed, questions to answer, and a
description of the provisional syntax in use until these decisions have been
made.

### Struct literal shortcut

We could allow you to write `{x, y}` as a short hand for `{.x = x, .y = y}`.

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
    v: Vector(i32)*,
    options: {.stable: bool = false,
              .descending: bool = false} = {}) {
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
    v: Vector(i32)*,
    {stable: bool = false, descending: bool = false}) {
  // Code using `stable` and `descending`.
}
```

This would allow the same syntax at the call site, but avoids
[some concerns with field defaults](https://github.com/carbon-language/carbon-lang/pull/561#discussion_r683856715)
and allows some other use cases such as destructuring return values.

#### Discussion

We might support destructuring directly:

```
var {key: String, value: i32} = ReturnKeyValue();
```

or by way of a mechanism that converts a struct into a tuple:

```
var (key: String, value: i32) =
    ReturnKeyValue().extract(.key, .value);
// or maybe:
var (key: String, value: i32) =
    ReturnKeyValue()[(.key, .value)];
```

Similarly we might support optional named parameters directly instead of by way
of struct types.

Some discussion on this topic has occurred in:

-   [question-for-leads issue #505 on named parameters](https://github.com/carbon-language/carbon-lang/issues/505)
-   labeled params brainstorming docs
    [1](https://docs.google.com/document/d/1Ui2OEHLwa9LZ6ktc1joJqE7_N-ZHX2gBvBpaFw6DUy8/edit?usp=sharing&resourcekey=0-6bEnyc03QePVcttPRSFoew),
    [2](https://docs.google.com/document/d/1kK_tti4DwPqa3Oh5CgA5pWSx0g3bKlZG1yREMpq9uiU/edit?usp=sharing&resourcekey=0-oFV6tXtCVu1bcHz4oCMyMQ)
-   ["match" in syntax choices doc](https://docs.google.com/document/d/1EhZA3AlY9TaCMho9jz2ynFxK-6eS6BwMAkE5jNYQzEA/edit?usp=sharing&resourcekey=0-QXEoh-b4_sQG2u636gIa1A#heading=h.y566d16ivoy2)

### Inheritance

#### C++ abstract base classes interoperating with object-safe interfaces

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

#### Overloaded methods

We allow a derived class to define a [class function](#class-functions) with the
same name as a class function in the base class. For example, we expect it to be
pretty common to have a constructor function named `Create` at all levels of the
type hierarchy.

Beyond that, we may want some rules or restrictions about defining methods in a
derived class with the same name as a base class method without overriding it.
There are some opportunities to improve on and simplify the C++ story:

-   We don't want to silently hide methods in the base class because of a method
    with the same name in a derived class. There are uses for this in C++, but
    it also causes problems and without multiple inheritance there isn't the
    same need in Carbon.
-   Overload resolution should happen before virtual dispatch.
-   For evolution purposes, you should be able to add private members to a base
    class that have the same name as member of a derived class without affecting
    overload resolution on instances of the derived class, in functions that
    aren't friends of the base class.

**References:** This was discussed in
[the open discussion on 2021-07-12](https://docs.google.com/document/d/14vAcURDKeH6LZ_TQCMRGpNJrXSZCACQqDy29YH19XGo/edit#heading=h.40jlsrcgp8mr).

#### Interop with C++ inheritance

This design directly supports Carbon classes inheriting from a single C++ class.

```
class CarbonClass extends C++.CPlusPlusClass {
  fn Create() -> Self {
    return {.base = C++.CPlusPlusClass(...), .other_fields = ...};
  }
  ...
}
```

To allow C++ classes to extend Carbon classes, there needs to be some way for
C++ constructors to initialize their base class:

-   There could be some way to export a Carbon class that identifies which
    factory functions may be used as constructors.
-   We could explicitly call the Carbon factory function, as in:

    ```
    // `Base` is a Carbon class which gets converted to a
    // C++ class for interop purposes:
    class Base {
    public:
        virtual ~Base() {}
        static auto Create() -> Base;
    };

    // In C++
    class Derived : public Base {
    public:
        virtual ~Derived() override {}
        // This isn't currently a case where C++ guarantees no copy,
        // and so it currently still requires a notional copy and
        // there appear to be implementation challenges with
        // removing them. This may require an extension to make work
        // reliably without an extraneous copy of the base subobject.
        Derived() : Base(Base::Create()) {}
    };
    ```

    However, this doesn't work in the case where `Base` can't be instantiated,
    or `Base` does not have a copy constructor, even though it shouldn't be
    called due to RVO.

##### Virtual base classes

TODO: Ask zygoloid to fill this in.

Carbon won't support declaring virtual base classes, and the C++ interop use
cases Carbon needs to support are limited. This will allow us to simplify the
C++ interop by allowing Carbon to delegate initialization of virtual base
classes to the C++ side.

This requires that we enforce two rules:

-   No multiple inheritance of C++ classes with virtual bases
-   No C++ class extending a Carbon class that extends a C++ class with a
    virtual base

### Mixins

We will need some way to declare mixins. This syntax will need a way to
distinguish defining versus requiring member variables. Methods may additionally
be given a default definition but may be overridden. Interface implementations
may only be partially provided by a mixin. Mixin methods will need to be able to
convert between pointers to the mixin type and the main type.

Open questions include whether a mixin is its own type that is a member of the
containing type, and whether mixins are templated on the containing type. Mixins
also complicate how constructors work.

### Memory layout

Carbon will need some way for users to specify the memory layout of class types
beyond simple ordering of fields, such as controlling the packing and alignment
for the whole type or individual members.

We may allow members of a derived class like to put data members in the final
padding of its base class prefix. Tail-padding reuse has both advantages and
disadvantages, so we may have some way for a class to explicitly mark that its
tail padding is available for use by a derived class,

Advantages:

-   Tail-padding reuse is sometimes a nice layout optimization (eg, in Clang we
    save 8 bytes per `Expr` by reusing tail padding).
-   No class size regressions when migrating from C++.
-   Special case of reusing the tail padding of a class that is empty other than
    its tail padding is very important, to the extent that we will likely need
    to support either zero-sized types or tail-padding reuse in order to have
    acceptable class layouts.

Disadvantages:

-   Cannot use `memcpy(p, q, sizeof(Base))` to copy around base class subobjects
    if the destination is an in-lifetime, because they might overlap other
    objects' representations.
-   Somewhat more complex model.
-   We need some mechanism for disabling tail-padding reuse in "standard layout"
    types.
-   We may also have to use narrowed loads for the last member of a base class
    to avoid accidentally creating a race condition.

However, we can still use `memcpy` and `memset` to initialize a base class
subobject, even if its tail padding might be reused, so long as we guarantee
that no other object lives in the tail padding and is initialized before the
base class. In C++, that happens only due to virtual base classes getting
initialized early and laid out at the end of the object; if we disallow virtual
base classes then we can guarantee that initialization order is address order,
removing most of the downside of tail-padding reuse.

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

external impl {.kind: WidgetKind, .size: i32}
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

It is an open question how to define implementations for binary operators. For
example, if `i32` is comparable to `f64`, then `{.x = 3, .y = 2.72}` should be
comparable to `{.x = 3.14, .y = 2}`. The trick is how to declare the criteria
that "`T` is comparable to `U` if they have the same field names in the same
order, and for every field `x`, the type of `T.x` implements `ComparableTo` for
the type of `U.x`."

## References

-   [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)
-   [#561: Basic classes: use cases, struct literals, struct types, and future wor](https://github.com/carbon-language/carbon-lang/pull/561)
-   [#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722)
-   [#777: Inheritance](https://github.com/carbon-language/carbon-lang/pull/777)
-   [#981: Implicit conversions for aggregates](https://github.com/carbon-language/carbon-lang/pull/981)
-   [#1154: Destructors](https://github.com/carbon-language/carbon-lang/pull/1154)
-   [#2107: Clarify rules around `Self` and `.Self`](https://github.com/carbon-language/carbon-lang/pull/2107)
