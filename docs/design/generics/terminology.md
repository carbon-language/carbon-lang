# Generics: Terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Generic means compile-time parameterized](#generic-means-compile-time-parameterized)
-   [Checked versus template parameters](#checked-versus-template-parameters)
    -   [Polymorphism](#polymorphism)
        -   [Parametric polymorphism](#parametric-polymorphism)
        -   [Compile-time duck typing](#compile-time-duck-typing)
        -   [Ad-hoc polymorphism](#ad-hoc-polymorphism)
    -   [Constrained genericity](#constrained-genericity)
    -   [Dependent names](#dependent-names)
    -   [Definition checking](#definition-checking)
        -   [Complete definition checking](#complete-definition-checking)
        -   [Early versus late type checking](#early-versus-late-type-checking)
-   [Bindings](#bindings)
-   [Types and `type`](#types-and-type)
-   [Facet type](#facet-type)
-   [Facet](#facet)
-   [Type expression](#type-expression)
-   [Facet binding](#facet-binding)
-   [Deduced parameter](#deduced-parameter)
-   [Interface](#interface)
    -   [Structural interfaces](#structural-interfaces)
    -   [Nominal interfaces](#nominal-interfaces)
    -   [Named constraints](#named-constraints)
-   [Associated entity](#associated-entity)
-   [Impl: Implementation of an interface](#impl-implementation-of-an-interface)
    -   [Extending an impl](#extending-an-impl)
-   [Member access](#member-access)
    -   [Simple member access](#simple-member-access)
    -   [Qualified member access expression](#qualified-member-access-expression)
-   [Compatible types](#compatible-types)
-   [Subtyping and casting](#subtyping-and-casting)
-   [Coherence](#coherence)
-   [Adapting a type](#adapting-a-type)
-   [Type erasure](#type-erasure)
-   [Archetype](#archetype)
-   [Extending an interface](#extending-an-interface)
-   [Dynamic-dispatch witness table](#dynamic-dispatch-witness-table)
-   [Instantiation](#instantiation)
-   [Specialization](#specialization)
    -   [Template specialization](#template-specialization)
    -   [Checked-generic specialization](#checked-generic-specialization)
-   [Conditional conformance](#conditional-conformance)
-   [Interface parameters and associated constants](#interface-parameters-and-associated-constants)
-   [Type constraints](#type-constraints)
-   [References](#references)

<!-- tocstop -->

## Generic means compile-time parameterized

Generally speaking, when we talk about _generics_, either
[checked or template](#checked-versus-template-parameters), we are talking about
generalizing some language construct by adding a _compile-time parameter_, also
called a _generic parameter_, to it. So:

-   a _generic function_ is a function with at least one compile-time parameter,
    which could be an explicit argument to the function or
    [deduced](#deduced-parameter);
-   a _generic type_ is a type with a compile-time parameter, for example a
    container type parameterized by the type of the contained elements;
-   a _generic interface_ is an [interface](#interface) with
    [a compile-time parameter](#interface-parameters-and-associated-constants).

This parameter broadens the scope of the language construct on an axis defined
by that parameter, for example it could define a family of functions instead of
a single one.

Note that different languages allow different things to be parameterized; for
example, Rust supports
[generic associated types](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html).

## Checked versus template parameters

When we distinguish between checked and template generics in Carbon, it is on a
parameter by parameter basis. A single function can take a mix of regular,
checked, and template parameters.

-   **Regular parameters**, or "dynamic parameters", are designated using the
    "\<name>`:` \<type>" syntax (or "\<value>").
-   **Checked parameters** are designated using `:!` between the name and the
    type (so it is "\<name>`:!` \<type>").
-   **Template parameters** are designated using "`template` \<name>`:!`
    \<type>".

The syntax for checked and template parameters was decided in
[questions-for-leads issue #565](https://github.com/carbon-language/carbon-lang/issues/565).

Expected difference between checked and template parameters:

<table>
  <tr>
   <td><strong>Checked</strong>
   </td>
   <td><strong>Template</strong>
   </td>
  </tr>
  <tr>
   <td>bounded parametric polymorphism
   </td>
   <td>compile-time duck typing and ad-hoc polymorphism
   </td>
  </tr>
  <tr>
   <td>constrained genericity
   </td>
   <td>optional constraints
   </td>
  </tr>
  <tr>
   <td>name lookup resolved for definitions in isolation ("early")
   </td>
   <td>name lookup can use information from arguments (name lookup may be "late")
   </td>
  </tr>
  <tr>
   <td>sound to typecheck definitions in isolation ("early")
   </td>
   <td>complete type checking may require information from calls (may be "late")
   </td>
  </tr>
  <tr>
   <td>supports separate type checking; may also support separate compilation
   </td>
   <td>separate compilation only to the extent that C++ supports it
   </td>
  </tr>
  <tr>
   <td>allowed but not required to be implemented using dynamic dispatch
   </td>
   <td>does not support implementation by way of dynamic dispatch, just static by way of <a href="#instantiation">instantiation</a>
   </td>
  </tr>
  <tr>
   <td>monomorphization is an optional optimization that cannot render the program invalid
   </td>
   <td>monomorphization is mandatory and can fail, resulting in the program being invalid
   </td>
  </tr>
</table>

### Polymorphism

Generics provide different forms of
[polymorphism](<https://en.wikipedia.org/wiki/Polymorphism_(computer_science)>)
than object-oriented programming with inheritance. That uses
[subtype polymorphism](https://en.wikipedia.org/wiki/Subtyping) where different
descendants, or "subtypes", of a base class can provide different
implementations of a method, subject to some compatibility restrictions on the
signature.

#### Parametric polymorphism

Parametric polymorphism
([Wikipedia](https://en.wikipedia.org/wiki/Parametric_polymorphism)) is when a
function or a data type can be written generically so that it can handle values
_identically_ without depending on their type.
[Bounded parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism#Bounded_parametric_polymorphism)
is where the allowed types are restricted to satisfy some constraints. Within
the set of allowed types, different types are treated uniformly.

#### Compile-time duck typing

Duck typing ([Wikipedia](https://en.wikipedia.org/wiki/Duck_typing)) is when the
legal types for arguments are determined implicitly by the usage of the values
of those types in the body of the function. Compile-time duck typing is when the
usages in the body of the function are checked at compile-time, along all code
paths. Contrast this with ordinary duck typing in a dynamic language such as
Python where type errors are only diagnosed at runtime when a usage is reached
dynamically.

#### Ad-hoc polymorphism

Ad-hoc polymorphism
([Wikipedia](https://en.wikipedia.org/wiki/Ad_hoc_polymorphism)), also known as
"overloading", is when a single function name has multiple implementations for
handling different argument types. There is no enforcement of any consistency
between the implementations. For example, the return type of each overload can
be arbitrary, rather than being the result of some consistent rule being applied
to the argument types.

Templates work with ad-hoc polymorphism in two ways:

-   A function with template parameters can be
    [specialized](#template-specialization) in
    [C++](https://en.cppreference.com/w/cpp/language/template_specialization) as
    a form of ad-hoc polymorphism.
-   A function with template parameters can call overloaded functions since it
    will only resolve that call after the types are known.

In Carbon, we expect there to be a compile error if overloading of some name
prevents a checked-generic function from being typechecked from its definition
alone. For example, let's say we have some overloaded function called `F` that
has two overloads:

```
fn F[template T:! type](x: T*) -> T;
fn F(x: Int) -> bool;
```

A checked generic function `G` can call `F` with a type like `T*` that cannot
possibly call the `F(Int)` overload for `F`, and so it can consistently
determine the return type of `F`. But `G` can't call `F` with an argument that
could match either overload.

**Note:** It is undecided what to do in the situation where `F` is overloaded,
but the signatures are consistent and so callers could still typecheck calls to
`F`. This still poses problems for the dynamic strategy for compiling generics,
in a similar way to impl specialization.

### Constrained genericity

We will allow some way of specifying constraints as part of a function (or type
or other parameterized language construct). These constraints are a limit on
what callers are allowed to pass in. The distinction between constrained and
unconstrained genericity is whether the body of the function is limited to just
those operations that are guaranteed by the constraints.

With templates using unconstrained genericity, you may perform any operation in
the body of the function, and they will be checked against the specific types
used in calls. You can still have constraints, but they are optional in that
they could be removed and the function would still have the same capabilities.
Constraints only affect the caller, which will use them to resolve overloaded
calls to the template and provide clearer error messages.

With checked generics using constrained genericity, the function body can be
checked against the signature at the time of definition. Note that it is still
perfectly permissible to have no constraints on a type; that just means that you
can only perform operations that work for all types (such as manipulate pointers
to values of that type) in the body of the function.

### Dependent names

A name is said to be _dependent_ if it depends on some checked or template
parameter. Note: this matches
[the use of the term "dependent" in C++](https://www.google.com/search?q=c%2B%2B+dependent+name),
not as in [dependent types](https://en.wikipedia.org/wiki/Dependent_type).

### Definition checking

Definition checking is the process of semantically checking the definition of
parameterized code for correctness _independently_ of any particular argument
values. It includes type checking and other semantic checks. It is possible,
even with templates, to check semantics of expressions that are not
[dependent](#dependent-names) on any template parameter in the definition.
Adding constraints to template parameters and/or switching them to be checked
allows the compiler to increase how much of the definition can be checked. Any
remaining checks are delayed until [instantiation](#instantiation), which can
fail.

#### Complete definition checking

Complete definition checking is when the definition can be _fully_ semantically
checked, including type checking. It is an especially useful property because it
enables _separate_ semantic checking of the definition, a prerequisite to
separate compilation. It also is a requirement for implementation strategies
that don’t instantiate the implementation (for example,
[type erasure](#type-erasure) or
[dynamic-dispatch witness tables](#dynamic-dispatch-witness-table)).

#### Early versus late type checking

Early type checking is where expressions and statements are type checked when
the definition of the function body is compiled, as part of definition checking.
This occurs for regular and checked-generic values.

Late type checking is where expressions and statements may only be fully
typechecked once calling information is known. Late type checking delays
complete definition checking. This occurs for
[template-dependent](#dependent-names) values.

## Bindings

_Binding patterns_ associate a name with a type and a value. This is used to
declare function parameters, in `let` and `var` declarations, as well as to
declare [compile-time parameters](#generic-means-compile-time-parameterized) for
classes, interfaces, and so on. There are three kinds of binding patterns,
corresponding to
[the three expression phases](/docs/design/README.md#expression-phases):

-   A _runtime binding pattern_ binds to a dynamic value at runtime, and is
    written using a `:`, as in `x: i32`.
-   A _symbolic binding pattern_ binds to a compile-time value that is not known
    when type checking, and is used to declare
    [checked generic](#checked-versus-template-parameters) parameters. These
    binding use `:!`, as in `T:! type`.
-   A _template binding pattern_ binds to a compile-time value that is known
    when type checking, and is used to declare
    [template](#checked-versus-template-parameters) parameters. These bindings
    use the keyword `template` in addition to `:!`, as in `template T:! type`.

The last two binding patterns, which are about binding a compile-time value, are
called _compile-time binding patterns_, and correspond to those binding patterns
that use `:!`.

The name being declared, which is the identifier to the left of the `:` or `:!`,
is called a _binding_, or more specifically a _runtime binding_, _compile-time
binding_, _symbolic binding_, or _template binding_. The expression to the right
defining the type of the binding pattern is called the _binding type
expression_, a kind of [type expression](#type-expression). For example, in
`T:! Hashable`, `T` is the binding (a symbolic binding in this case), and
`Hashable` is the binding type expression.

## Types and `type`

A _type_ is a value of type `type`. Conversely, `type` is the type of all types.

Expressions in type position, for example a [binding type expression](#bindings)
or the return type of a function, are implicitly cast to type `type`. This means
that it is legal to put a value that is not a type where a type is expected, as
long as it has an implicit conversion to `type` that may be performed at compile
time.

## Facet type

A _facet type_ is a [type](#types-and-type) whose values are some subset of the
values of `type`, determined by a set of [type constraints](#type-constraints):

-   [Interfaces](#interface) and [named constraints](#named-constraints) are
    facet types whose constraints are that the interface or named constraint is
    satisfied by the type.
-   The values produced by `&` operations between facet types and by `where`
    expressions are facet types, whose set of constraints are determined by the
    `&` or `where` expression.
-   `type` is a facet type whose set of constraints is empty.

A facet type is the type used when declaring some type parameter. It foremost
determines which types are legal arguments for that type parameter. For template
parameters, that is all a facet type does. For checked parameters, it also
determines the API that is available in the body of the definition of the
[generic function, class, or other entity](#generic-means-compile-time-parameterized).

## Facet

A _facet_ is a value of a [facet type](#facet-type). For example,
`i32 as Hashable` is a facet, and `Hashable` is a facet type. Note that all
types are facets, since [`type`](#types-and-type) is considered a facet type.
Not all facets are types, though: `i32 as Hashable` is of type `Hashable` not
`type`, so it is a facet that is not a type. However, in places where a type is
expected, for example in a [binding type expression](#bindings) or after the
`->` in a function declaration, there is an automatic implicit conversion to
`type`. This means that a facet may be used in those positions. For example, the
facet `i32 as Hashable` will implicitly convert to `(i32 as Hashable) as type`,
which is `i32`, in those contexts.

## Type expression

A _type expression_ is an expression that can be used as a type. In some cases,
what is written in the source code is a value, like a [facet](#facet) or tuple
of types, that is not a type but has an implicit conversion to `type`. In those
cases, we are concerned with the type value after the implicit conversion.

## Facet binding

We use the term _facet binding_ to refer to the name introduced by a
[compile-time binding pattern](#bindings) (using `:!` with or without the
`template` modifier) where the declared type is a [facet type](#facet-type). In
the binding pattern `T:! Hashable`, `T` is a facet binding, and the value of `T`
is a [facet](#facet).

## Deduced parameter

A deduced parameter is listed in the optional `[` `]` section right after the
function name in a function signature:

`fn` \<name> `[` \<deduced parameters> `](` \<explicit parameters> `) ->`
\<return type>

Deduced arguments are determined as a result of pattern matching the explicit
argument values (usually the types of those values) to the explicit parameters.

See more [here](overview.md#deduced-parameters).

## Interface

An interface is an API constraint used in a function signature to provide
encapsulation. Encapsulation here means that callers of the function only need
to know about the interface requirements to call the function, not anything
about the implementation of the function body, and the compiler can check the
function body without knowing anything more about the caller. Callers of the
function provide a value that has an implementation of the API and the body of
the function may then use that API. In the case of a checked generic, the
function may _only_ use that API.

### Structural interfaces

A "structural" interface is one where we say a type satisfies the interface as
long as it has members with a specific list of names, and for each name it must
have some type or signature. A type can satisfy a structural interface without
ever naming that interface, just by virtue of having members with the right
form.

### Nominal interfaces

A "nominal" interface is one where we say a type can only satisfy an interface
if there is some explicit statement saying so, for example by defining an
[impl](#impl-implementation-of-an-interface). This allows "satisfies the
interface" to have additional semantic meaning beyond what is directly checkable
by the compiler. For example, knowing whether the `Draw` function means "render
an image to the screen" or "take a card from the top of a deck of cards"; or
that a `+` operator is commutative (and not, say, string concatenation).

We use the "structural" versus "nominal" terminology as a generalization of the
same terms being used in a
[subtyping context](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes).

### Named constraints

Named constraints are "structural" in the sense that they match a type based on
meeting some criteria rather than an explicit statement in the type's
definition. The criteria for a named constraint, however, are less focused on
the type's API and instead might include a set of nominal interfaces that the
type must implement and constraints on the
[associated entities](#associated-entity) and
[interface parameters](#interface-parameters-and-associated-constants).

## Associated entity

An _associated entity_ is a requirement in an interface that a type's
implementation of the interface must satisfy by having a matching definition. A
requirement that the type define a value for a member constant is called an
_associated constant_. If the type of the associated constant is a
[facet type](#facet-type), then it is called an _associated [facet](#facet)_,
which corresponds to what is called an "associated type" in other languages
([Swift](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/generics/#Associated-Types),
[Rust](https://doc.rust-lang.org/reference/items/associated-items.html#associated-types)).
Similarly, an interface can have _associated function_, _associated method_, or
_associated class function_.

Different types can satisfy an interface with different definitions for a given
member. These definitions are _associated_ with what type is implementing the
interface. An [impl](#impl-implementation-of-an-interface) defines what is
associated with the type for that interface.

Rust uses the term
["associated item"](https://doc.rust-lang.org/reference/items/associated-items.html)
instead of associated entity.

## Impl: Implementation of an interface

An _impl_ is an implementation of an interface for a specific type, called the
_implementing type_. It is the place where the function bodies are defined,
values for associated constants, etc. are given. Implementations are needed for
[nominal interfaces](#nominal-interfaces);
[structural interfaces](#structural-interfaces) and
[named constraints](#named-constraints) define conformance implicitly instead of
by requiring an impl to be defined. In can still make sense to explicitly
implement a named constraint as a way to implement all of the interfaces it
requires.

### Extending an impl

A type that _extends_ the implementation of an interface has all the named
members of the interface as named members of the type. This means that the
members of the interface are available by way of both
[simple member access and qualified member access expressions](#member-access).

If a type implements an interface without extending, the members of the
interface may only be accessed using
[qualified member access expressions](#qualified-member-access-expression).

## Member access

There are two different kinds of member access: _simple_ and _compound_. See the
[member access design document](/docs/design/expressions/member_access.md) for
the details. The application to generics combines compound member access with
qualified names, which we call a _qualified member access expression_.

### Simple member access

Simple member access has the from `object.member`, where `member` is a word
naming a member of `object`. This form may be used to access members of
interfaces when the type of `object`
[extends the implementation](#extending-an-impl) of that interface.

If `String` extends its implementation of `Printable`, then `s1.Print()` calls
the `Print` method of `Printable` using simple member access. In this case, the
name `Print` is used without qualifying it with the name of the interface it is
a member of since it is recognized as a member of the type itself as well.

### Qualified member access expression

Compound member access has the form `object.(expression)`, where `expression` is
resolved in the containing scope. A compound member access where the member
expression is a simple member access expression, as in `a.(context.b)`, is
called a _qualified member access expression_. The member expression `context.b`
may be the _qualified member name_ of an interface member, that consists of the
name of the interface, possibly qualified with a package or namespace name, a
dot `.` and the name of the member.

For example, if the `Comparable` interface has a `Less` member method, then the
qualified name of that member is `Comparable.Less`. So if `String` implements
`Comparable`, and `s1` and `s2` are variables of type `String`, then the `Less`
method may be called using the qualified member name by writing the qualified
member access expression `s1.(Comparable.Less)(s2)`.

This form may be used to access any member of an interface implemented for a
type, whether or not it [extends the implementation](#extending-an-impl).

## Compatible types

Two types are compatible if they have the same notional set of values and
represent those values in the same way, even if they expose different APIs. The
representation of a type describes how the values of that type are represented
as a sequence of bits in memory. The set of values of a type includes properties
that the compiler can't directly see, such as invariants that the type
maintains.

We can't just say two types are compatible based on structural reasons. Instead,
we have specific constructs that create compatible types from existing types in
ways that encourage preserving the programmer's intended semantics and
invariants, such as implementing the API of the new type by calling (public)
methods of the original API, instead of accessing any private implementation
details.

## Subtyping and casting

Both subtyping and casting are different names for changing the type of a value
to a compatible type.

[Subtyping](https://en.wikipedia.org/wiki/Subtyping) is a relationship between
two types where you can safely operate on a value of one type using a variable
of another. For example, using C++'s object-oriented features, you can operate
on a value of a derived class using a pointer to the base class. In most cases,
you can pass a more specific type to a function that can handle a more general
type. Return types work the opposite way, a function can return a more specific
type to a caller prepared to handle a more general type. This determines how
function signatures can change from base class to derived class, see
[covariance and contravariance in Wikipedia](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>).

In a generics context, we are specifically interested in the subtyping
relationships between [facet types](#facet-type). In particular, a facet type
encompasses a set of [type constraints](#type-constraints), and you can convert
a type from a more-restrictive facet type to another facet type whose
constraints are implied by the first. C++ concepts terminology uses the term
["subsumes"](https://en.cppreference.com/w/cpp/language/constraints#Partial_ordering_of_constraints)
to talk about this partial ordering of constraints, but we avoid that term since
it is at odds with the use of the term in
[object-oriented subtyping terminology](https://en.wikipedia.org/wiki/Subtyping#Subsumption).

Note that subtyping is a bit like
[coercion](https://en.wikipedia.org/wiki/Type_conversion), except we want to
make it clear that the data representation of the value is not changing, just
its type as reflected in the API available to manipulate the value.

Casting is indicated explicitly by way of some syntax in the source code. You
might use a cast to switch between [type adaptations](#adapting-a-type), or to
be explicit where an implicit conversion would otherwise occur. For now, we are
saying "`x as y`" is the provisional syntax in Carbon for casting the value `x`
to the type `y`. Note that outside of generics, the term "casting" includes any
explicit type change, including those that change the data representation.

In contexts where an expression of one type is provided and a different type is
required, an [implicit conversion](../expressions/implicit_conversions.md) is
performed if it is considered safe to do so. Such an implicit conversion, if
permitted, always has the same meaning as an explicit cast.

## Coherence

A generics or interface system has the _implementation coherence_ property, or
simply _coherence_, if there is a single answer to the question "what is the
implementation of this interface for this type, if any?" independent of context,
such as the libraries imported into a given file. Coherence is
[a goal of Carbon checked generics](goals.md#coherence).

This is enforced using two kinds of rules:

-   An _orphan rule_ is a restriction on which files may declare a particular
    implementation. This is to ensure that the implementation is imported any
    time it could be used. For example, if neither the type nor the interface is
    parameterized, the orphan rule requires that the implementation must be in
    the same library as the interface or type. The rule is we don't allow an
    _orphan_ implementation that is not with either of its parents (parent type
    or parent interface).
-   An _overlap rule_ is a way to _consistently_ select a single implementation
    when multiple implementations apply. In Carbon, overlap is resolved by
    picking a single implementation using a rule that picks the one that is
    considered most specialized. In Rust, by contrast, the
    [overlap rule or overlap check](https://rust-lang.github.io/chalk/book/clauses/coherence.html#chalk-overlap-check)
    instead produces an error if two implementations apply at once.

The rationale for Carbon choosing coherence and alternatives considered may be
found in [this appendix](appendix-coherence.md)

## Adapting a type

A type can be adapted by creating a new type that is
[compatible](#compatible-types) with an existing type, but has a different API.
In particular, the new type might implement different interfaces or provide
different implementations of the same interfaces.

Unlike extending a type (as with C++ class inheritance), you are not allowed to
add new data fields onto the end of the representation -- you may only change
the API. This means that it is safe to [cast](#subtyping-and-casting) a value
between those two types without any dynamic checks or danger of
[object slicing](https://en.wikipedia.org/wiki/Object_slicing).

This is called "newtype" in Rust, and is used for capturing additional
information in types to improve type safety by moving some checking to compile
time ([1](https://doc.rust-lang.org/rust-by-example/generics/new_types.html),
[2](https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction),
[3](https://www.worthe-it.co.za/blog/2020-10-31-newtype-pattern-in-rust.html))
and as a workaround for
[Rust's orphan rules for coherence](https://github.com/Ixrec/rust-orphan-rules#why-are-the-orphan-rules-controversial).

## Type erasure

"Type erasure" is where a type's API is replaced by a subset. Everything outside
of the preserved subset is said to have been "erased". This can happen in a
variety of contexts including both checked generics and runtime polymorphism.
For checked generics, type erasure restricts a type to just the API required by
the constraints on that type stated in the signature of the function.

An example of type erasure in runtime polymorphism in C++ is casting from a
pointer of a derived type to a pointer to an abstract base type. Only the API of
the base type is available on the result, even though the implementation of
those methods still come from the derived type.

The term "type erasure" can also refer to
[the specific strategy used by Java to implement generics](https://en.wikipedia.org/wiki/Generics_in_Java).
which includes erasing the identity of type parameters. This is not the meaning
of "type erasure" used in Carbon.

## Archetype

A placeholder type is used when type checking a function in place of a generic
type parameter. This allows type checking when the specific type to be used is
not known at type-checking time. The type satisfies just its constraint and no
more, so it acts as the most general type satisfying the interface. In this way
the archetype is the supertype of all types satisfying the interface.

In addition to satisfying all the requirements of its constraint, the archetype
also has the member names of its constraint. Effectively it is considered to
[extend the implementation of the constraint](#extending-an-impl).

## Extending an interface

An interface can be extended by defining an interface that includes the full API
of another interface, plus some additional API. Types implementing the extended
interface should automatically be considered to have implemented the narrower
interface.

## Dynamic-dispatch witness table

Dynamic-dispatch
[witness tables](https://forums.swift.org/t/where-does-the-term-witness-table-come-from/54334/4)
are an implementation strategy that uses a table accessed at runtime to allow
behavior of a function to vary. This allows a function to work with any type
implementing a facet type (such as an interface). For example, the witness table
might contain pointers to the implementations of the functions of the interface.
This can be done to reduce the size of generated code, at the expense of
additional indirection at runtime.

It can also allow a function to dynamically dispatch when the runtime type of a
value is not known. This is the implementation strategy for
[boxed protocol types in Swift](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/opaquetypes/#Boxed-Protocol-Types)
and
[trait objects in Rust](https://doc.rust-lang.org/book/ch17-02-trait-objects.html).
Note that this often comes with limitations, since for example it is much more
difficult to support when the associated constants of the interface are not
known.

Typically a reference to the witness table will be passed separately from the
object, unlike a
[virtual method table](https://en.wikipedia.org/wiki/Virtual_method_table),
which otherwise is very similar to a witness table, "witnessing" the specific
descendant of a base class.

Carbon's approach to using witness tables is detailed in an
[appendix](appendix-witness.md).

## Instantiation

Instantiation is the implementation strategy for templates in both C++ and
Carbon. Instantiation explicitly creates a copy of the template code and
replaces the template components with the concrete type and its implementation
operations. It allows duck typing and lazy binding. Instantiation implies
template code **will** be duplicated.

Unlike static-dispatch witness tables (as in Swift) and
[monomorphization (as in Rust)](https://doc.rust-lang.org/book/ch10-01-syntax.html#performance-of-code-using-generics),
this is done **before** type checking completes. Only when the template is used
with a concrete type is the template fully type checked, and it type checks
against the actual concrete type after substituting it into the template. This
means that different instantiations may interpret the same construct in
different ways, and that templates can include constructs that are not valid for
some possible instantiations. However, it also means that some errors in the
template implementation may not produce errors until the instantiation occurs,
and other errors may only happen for **some** instantiations.

## Specialization

### Template specialization

Specialization in C++ is essentially overloading, or
[ad-hoc polymorphism](#ad-hoc-polymorphism), in the context of a template. The
template is overloaded to have a different definition for some subset of the
possible template argument values. For example, the C++ type `std::vector<T>`
might have a specialization `std::vector<T*>` that is implemented in terms of
`std::vector<void*>` to reduce code size. In C++, even the interface of a
templated type can be changed in a specialization, as happens for
`std::vector<bool>`.

### Checked-generic specialization

Specialization of checked generics, or types used by checked generics, is
restricted to changing the implementation _without_ affecting the interface.
This restriction is needed to preserve the ability to perform type checking of
generic definitions that reference a type that can be specialized, without
statically knowing which specialization will be used.

## Conditional conformance

Conditional conformance is when you have a parameterized type that has one API
that it always supports, but satisfies additional interfaces under some
conditions on the type argument. For example: `Array(T)` might implement
`Comparable` if `T` itself implements `Comparable`, using lexicographical order.

## Interface parameters and associated constants

_Interface parameters_ and [associated constants](#associated-entity) are both
ways of allowing the types in function signatures in an interface to vary. For
example, different
[stacks](<https://en.wikipedia.org/wiki/Stack_(abstract_data_type)>) will have
different element types. That element type would be used as the parameter type
of the `Push` function and the return type of the `Pop` function. As
[in Rust](https://rust-lang.github.io/rfcs/0195-associated-items.html#clearer-trait-matching),
we can distinguish these by whether they are input parameters or output
parameters:

-   An interface parameter is a parameter or input to the interface. That means
    they must be specified before an implementation of the interface may be
    determined.
-   In contrast, associated constants are outputs. This means that they are
    determined by the implementation, and need not be specified in a
    [type constraint](#type-constraints).

Functions using an interface as a constraint need not specify the value of its
associated constants.

```
// Stack using associated facets
interface Stack {
  let ElementType:! type;
  fn Push[addr self: Self*](value: ElementType);
  fn Pop[addr self: Self*]() -> ElementType;
}

// Works on any type implementing `Stack`. Return type
// is determined by the type's implementation of `Stack`.
fn PeekAtTopOfStack[T:! Stack](s: T*) -> T.ElementType {
  let ret: T.ElementType = s->Pop();
  s->Push(ret);
  return ret;
}

class Fruit;
class FruitStack {
  // Implement `Stack` for `FruitStack`
  // with `ElementType` set to `Fruit`.
  extend impl as Stack where .ElementType == Fruit { ... }
}
```

Associated constants are particularly called for when the implementation of the
interface determines the value, not the caller. For example, the iterator type
for a container is specific to the container and not something you would expect
a user of the interface to specify.

If you have an interface with parameters, a type can have multiple matching
`impl` declarations for different combinations of argument values. As a result,
interface parameters may not be deduced in a function call. However, if the
interface parameters are specified, a type can only have a single implementation
of the given interface. This unique implementation choice determines the values
of associated constants.

For example, we might have an interface that says how to perform addition with
another type:

```
interface AddWith(T:! type) {
  let ResultType:! type;
  fn Add[self: Self](rhs: T) -> ResultType;
}
```

An `i32` value might support addition with `i32`, `u16`, and `f64` values.

```
impl i32 as AddWith(i32) where .ResultType = i32 { ... }
impl i32 as AddWith(u16) where .ResultType = i32 { ... }
impl i32 as AddWith(f64) where .ResultType = f64 { ... }
```

To write a generic function requiring a parameter to be `AddWith`, there needs
to be some way to determine the type to add to:

```
// ✅ This is allowed, since the value of `T` is determined by the
// `y` parameter.
fn DoAdd[T:! type, U:! AddWith(T)](x: U, y: T) -> U.ResultType {
  return x.Add(y);
}

// ❌ This is forbidden, can't uniquely determine `T`.
fn CompileError[T:! type, U:! AddWith(T)](x: U) -> T;
```

Once the interface parameters can be determined, that determines the values for
associated constants, such as `ResultType` in the example. As always, calls with
types for which no implementation exists will be rejected at the call site:

```
// ❌ This is forbidden, no implementation of `AddWith(Orange)`
// for `Apple`.
DoAdd(apple, orange);
```

The type of an interface parameters and associated constants is commonly a
[facet type](#facet-type), but not always. For example, an interface parameter
that specifies an array bound might have an integer type.

## Type constraints

Type constraints restrict which types are legal for a
[facet binding](#facet-binding), like a facet parameter or associated facet.
They help define semantics under which they should be called, and prevent
incorrect calls.

In general there are a number of different type relationships we would like to
express, for example:

-   This function accepts two containers. The container types may be different,
    but the element types need to match.
-   For this container interface we have associated facets for iterators and
    elements. The iterator type's element type needs to match the container's
    element type.
-   An interface may define an associated facet that needs to be constrained to
    implement some interfaces.
-   This type must be [compatible](#compatible-types) with another type. You
    might use this to define alternate implementations of a single interfaces,
    such as sorting order, for a single type.

Note that type constraints can be a restriction on one facet parameter or
associated facet, or can define a relationship between multiple facets.

## References

-   [#447: Generics terminology](https://github.com/carbon-language/carbon-lang/pull/447)
-   [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)
-   [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
-   [#1013: Generics: Set associated constants using where constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
-   [#2138: Checked and template generic terminology](https://github.com/carbon-language/carbon-lang/pull/2138)
-   [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
-   [#2760: Consistent `class` and `interface` syntax](https://github.com/carbon-language/carbon-lang/pull/2760)
