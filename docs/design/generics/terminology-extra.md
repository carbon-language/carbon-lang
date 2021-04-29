# Carbon: Generics - Terminology, extra content

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Basics](#basics)
-   [Interface](#interface)
    -   [What kind of values are interfaces?](#what-kind-of-values-are-interfaces)
        -   [Interfaces are concrete types](#interfaces-are-concrete-types)
        -   [Interfaces are type-types](#interfaces-are-type-types)
            -   [Facet type-types](#facet-type-types)
            -   [Type-types parameterized by reprs](#type-types-parameterized-by-reprs)
        -   [Interfaces are opaque](#interfaces-are-opaque)
-   [Impls: Implementations of interfaces](#impls-implementations-of-interfaces)
    -   [Default impl](#default-impl)
    -   [Named impl](#named-impl)
    -   [Templated impl](#templated-impl)
-   [Invoking interface methods](#invoking-interface-methods)
    -   [Facet types](#facet-types)
    -   [Union Types](#union-types)
    -   [Separate Impls](#separate-impls)
-   [Implementation strategies for generics](#implementation-strategies-for-generics)
    -   [Witness tables (for example, Swift and Carbon Generics)](#witness-tables-for-example-swift-and-carbon-generics)
        -   [Dynamic-dispatch witness table](#dynamic-dispatch-witness-table)
        -   [Static-dispatch witness table](#static-dispatch-witness-table)
    -   [Type erasure (for example, Java)](#type-erasure-for-example-java)
    -   [Monomorphization (for example, Rust)](#monomorphization-for-example-rust)
    -   [Instantiation (for example, C++ and Carbon Templates)](#instantiation-for-example-c-and-carbon-templates)
-   [Specialization](#specialization)
-   [Conditional conformance](#conditional-conformance)
-   [Interface type parameters versus associated types](#interface-type-parameters-versus-associated-types)
-   [Type constraints](#type-constraints)
-   [Dependent types (or more generally, values)](#dependent-types-or-more-generally-values)
-   [Broken links footnote](#broken-links-footnote)

<!-- tocstop -->

## Basics

This document is an extension of the [main terminology doc](terminology.md) that
covers the breadth of alternatives considered for generics in Carbon and the
process for choosing between them.

Please see the
[Carbon principle: Generics](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md)
and
[Templates and generics: distinctions (TODO)](#broken-links-footnote)<!-- T:Templates and generics: distinctions -->
docs for a description of generics in Carbon, specifically the goals and
differences from templates. Additionally, the "Carbon Generic" doc has
"[What are generics?](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/overview.md#what-are-generics)"
and
"[Goals: Generics](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/overview.md#goals-generics)"
sections.

TODO: Runtime versus compile-time trade offs, worry about things like size of
value unknown producing much slower code -- needs guardrails, explicit opt-in?

## Interface

An interface is an API constraint used in a function signature to provide
encapsulation. Encapsulation here means that callers of the function only need
to know about the interface requirements to call the function, not anything
about the implementation of the function body, and the compiler can check the
function body without knowing anything more about the caller. Callers of the
function provide a value that has an implementation of the API and the body of
the function may then use that API (and nothing else).

There are a few different possible interface programming models.

### What kind of values are interfaces?

If you use the name of an interface after it has been defined, what role does it
play grammatically? What kind of value is it?

#### Interfaces are concrete types

In one programming model, an interface is thought of as a forward declaration of
a concrete type. For example, "`interface Comparable(Type:$ T) for T`" means
that there is a type `Comparable(T)` defined for some types `T`, and for those
`T` where it is defined, you may implicitly cast between `T` and `Comparable(T)`
(they are [compatible types](terminology.md#compatible-types)). Here,
`Comparable(T)` is a [facet type](#facet-types) for `T`, so `Comparable(T)` has
the API defined by the interface. You might define a generic function like so:

```
fn F[Type:$ T](Comparable(T): x) ...
```

or

```
fn F(Comparable(Type:$ T): x) ...
```

The `$` means that `T` is known generically -- it must be known to the caller,
but the specific value will not be used in type checking the body of the
function. A caller passing a value of type `T` for which `Comparable(T)` is
defined will implicitly cast the argument from `T` to `Comparable(T)`. Inside
the body of the function `x` will have type `Comparable(T)` and therefore will
have the API reflecting the interface defined by `Comparable` rather than the
API of `T`.

See
["Interfaces are concrete facet types"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/overview.md#interfaces-are-concrete-types)
in the Generics proposal for more on this programming model.

#### Interfaces are type-types

In other programming models, an interface is a type-type, that is, an interface
is a type whose values are also types (satisfying some properties). For example,
we might define a generic function like so:

```
fn F[InterfaceName:$ T](T: x) ...
```

Here we say `T` is a value whose type is `InterfaceName`, which means that `T`
is a type that implements the interface with name `InterfaceName` (in some way),
and that `x` is a value of the type `T`. Again, the `$` means that `T` is known
generically -- it must be known to the caller, but the specific value will not
be used in type checking the body of the function.

There are different variations of type-types; below are ones that have been
actively proposed.

##### Facet type-types

Type `T` satisfies `InterfaceName` if it is a [facet type](#facet-types) of
`InterfaceName` for some `NativeType`. That means that `NativeType` provides an
implementation of `InterfaceName` and `T` is the type compatible with
`NativeType` that exposes that specific implementation, so the API of `T`
matches `InterfaceName` instead of `NativeType`. TODO: deep dive describing this
programming model; currently the best we have is
[Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.3kqiirqlj97f -->.

##### Type-types parameterized by reprs

Type `T` satisfies `InterfaceName(R)` if it is a facet type (as above) of
`InterfaceName` that is compatible with (shares a representation with) type `R`.
This is good for representing multiple implementations of a single interface and
implementations of multiple interfaces (in both cases sharing the type `R`).
However it creates problems when you try and infer `R` from the types of the
values passed in by the user, since it is unclear which `R` to pick from the set
of types sharing the same representation.

See
["Type-types parameterized by reprs"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/overview.md#type-types-parameterized-by-reprs)
in the Generics proposal for more on this programming model.

#### Interfaces are opaque

Another possibility is that an interface is an opaque key. This has the
advantage that it doesn't privilege any one use of an interface. Any use of an
interface is in the context of a language construct that explicitly takes an
interface as one of its arguments and the use of the construct makes it clear
what role the interface is playing.

An example of a programming model like this is presented in
[Carbon closed function overloading proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#heading=h.qkop7hrup3jx -->.

## Impls: Implementations of interfaces

An _impl_ is an implementation of an interface for a specific type. It is the
place where the function bodies are defined, values for associated types, etc.
are given. A given generics programming model may support default impls, named
impls, or both. Impls are mostly associated with nominal interfaces; structural
interfaces define conformance implicitly instead of by requiring an impl to be
defined.

### Default impl

A default impl is an implementation associated with a (type, interface) pair
without a separate name.

Generally speaking, in order to make sure that every place looking up the
default impl for a particular interface for a type sees the same result, we
require default impls to be defined either with the interface or the type.

The main benefit of default impls is that it allows
[subsumption](terminology.md#subtyping-and-casting) to work. That is, by
allowing the compiler to look up the impl for a (type, interface) combination
automatically, you allow passing of values to generic functions transparently,
without any explicit statement of how to satisfy the parameter's interface
requirements. This of course means that any given type can have at most one
default impl for any given interface.

### Named impl

A named impl is an implementation of a (type, interface) pair that is given its
own name. The impl won't be used by default/implicitly; it will only be used as
a result of looking up its name, which means it can be defined anywhere. There
is no restriction that a (type, interface) pair be limited to one named impl.

### Templated impl

This is where there is a templated definition of an impl, defined along with an
interface, that specifies a common implementation for an interface for a variety
of types. For example, it could be used to say that any type implementing these
other two interfaces automatically gets an implementation for this interface, or
to say that you can satisfy this interface structurally.

## Invoking interface methods

Given a function with a generic parameter whose type is known to satisfy some
number of interfaces, how do you actually call methods defined in one of those
interfaces in the body of the function?

For the following possibilities we will use a running example with a struct `S`
implementing two interfaces `A` and `B` defining functions `F` and `G`
respectively (using [facet type-type](#facet-type-types) syntax):

```
interface A {
  fn F(Self*: this);
}
interface B {
  fn G(Self*: this);
}
struct S {
  fn H(Self*: this) { ... }
  impl A { fn F... }
  impl B { fn G... }
}
```

Note: There are other possibilities in this space, but these are the ones that
have come up so far.

### Facet types

One approach is to cast values to a facet type. A facet type is a
[compatible type](terminology.md#compatible-types) to the original type written
by the user, but has an API exactly matching one interface.

A facet type corresponds to a single
[impl](#impls-implementations-of-interfaces), and so you would have one facet
type for every interface that a type implements. Impls, though, are typically
discussed as part of saying how types define implementations for interfaces,
whereas facet types are about how code invokes API functions or uses other
members defined in an impl.

For our example, you might write a function that takes values with a type
implementing both `A` and `B` like so:

```
// "facet type-type" syntax
fn Z[TypeImplements(A, B):$ T](T*: x) {
  (x as (T as A)*)->F();
  (x as (T as B)*)->G();
}
// "interfaces are concrete types" syntax
fn Z[Type:$ T, A(T), B(T)](T*: x) {
  (x as A(T)*)->F();
  (x as B(T)*)->G();
}
```

Facet types have the advantage that they separate name lookup for interfaces
from each other and from the names defined in the native type. You know that
when you are using a value cast to a facet type it definitely is going to use
the function associated with a specific interface. The disadvantage is, if you
are operating on a type implementing multiple interfaces, you have to be
specific about which interface you are using at any given time. This creates a
difference in how code would be written from templates using duck typing.

Facet types may be most useful with nominal interfaces. You could also imagine
facet types in a programming model with structural interfaces and no impls. In
this case, the facet type would be a projection of a type's API onto the subset
defined by the interface. However, in this situation you would have no need to
maintain separate namespaces, and so union types would make more sense.

### Union Types

A union type would concatenate the APIs of multiple interfaces together. For our
example, you might write a function that takes values with a type implementing
both `A` and `B` like so:

```
// "facet type-type" syntax
fn Z[A & B:$ T](T*: x) {
  x->F();
  x->G();
}
```

This would reduce the code delta from templated code, but introduces the
possibility that there could be name conflicts between the members of interfaces
being combined, and new conflicts could be introduced as interfaces evolve. This
would need some mechanisms for picking some subset of the functions, renaming
functions, and marking new additions to an interface as "upcoming" to create a
transition time where conflicts can be dealt with before they become errors.

### Separate Impls

Separate impls use non-type objects that can be passed around as values. The
type of an impl value is parameterized by both an interface and a compatible
type. For example, if `x` is the name of an impl value for interface `I` and
compatible type `T` and `F` is the name of function defined in `I`, and `y` is a
value of type `T`, you might have an explicit syntax where you call the `F` from
`x` on `y` using `x.F(y)`, or an implicit syntax like `with (x) { y.F(); }`.

For our example, you might write a function that takes values with a type
implementing both `A` and `B` like so:

```
fn Z[Type:$ T](T*: x,
               Impl(A, T): a = T.DefaultImpl(A),
               Impl(B, T): b = T.DefaultImpl(B)) {
  a.F(x);
  b.G(x);
  // Or
  with (a) { x.F(); }
  with (b) { x.G(); }
}
```

These were proposed in
[Carbon: Impls are values passed as arguments with defaults (TODO)](#broken-links-footnote)<!-- T:Carbon: Impls are values passed as arguments with defaults -->,
but so far have not proven popular.

## Implementation strategies for generics

Witness tables, type erasure, monomorphization and instantiation all describe
methods for implementing generics. They are each trying to address how generics
perform operations on the type provided by a caller.

### Witness tables (for example, Swift and Carbon Generics)

For witness tables, values passed to a generic parameter are compiled into a
table of required functionality. That table is then filled in for a given
passed-in type with references to the implementation on the original type. The
generic is implemented using calls into entries in the witness table, which turn
into calls to the original type. This doesn't necessarily imply a runtime
indirection: it may be a purely compile-time separation of concerns. However, it
insists on a full abstraction boundary between the generic user of a type and
the concrete implementation.

A simple way to imagine a witness table is as a struct of function pointers, one
per method in the interface. However, in practice, it's more complex because it
must model things like associated types and interfaces.

Witness tables are called "dictionary passing" in Haskell. Outside of generics,
a [vtable](https://en.wikipedia.org/wiki/Virtual_method_table) is a witness
table that witnesses that a class is a descendant of an abstract base class, and
is passed as part of the object instead of separately.

#### Dynamic-dispatch witness table

For dynamic-dispatch witness tables, actual function pointers are formed and
used as a dynamic, runtime indirection. As a result, the generic code **will
not** be duplicated for different witness tables.

#### Static-dispatch witness table

For static-dispatch witness tables, the implementation is required to collapse
the table indirections at compile time. As a result, the generic code **will**
be duplicated for different witness tables.

Static-dispatch may be implemented as a performance optimization for
dynamic-dispatch that increases generated code size. The final compiled output
may not retain the witness table.

### Type erasure (for example, Java)

Type erasure is similar to dynamic-dispatch witness tables, but it goes further
and pushes the abstraction all the way to runtime. The actual type is completely
unknown at compile time. Type erasure implies generic code **will not** be
duplicated.

A fundamental distinction from a witness table is that the actual type cannot be
recovered. When dynamic-dispatch witness tables are used, they may still model
the actual type through some dynamic system rather than ensuring it is fully
opaque. Type erasure removes that option, which can
[cause problems](https://en.wikipedia.org/wiki/Generics_in_Java#Problems_with_type_erasure).

### Monomorphization (for example, Rust)

Monomorphization explicitly creates a copy of the generic code and replaces the
generic components with the concrete type and its implementation operations.
Monomorphization implies generic code **will** be duplicated.

Monomorphization is similar to instantiation, except that it's done **after**
type checking. This allows monomorphization to do type checks to be done on the
generic in isolation, meaning errors apply to **all** possible instantiations.

Static-dispatch witness table output looks similar to monomorphization output.
However, monomorphization does not require a witness table. The risk is that by
conceptualizing the implementation as monomorphization we may unintentionally
introduce cases that cannot be represented as dynamic-dispatch witness tables.

### Instantiation (for example, C++ and Carbon Templates)

Instantiation, like monomorphization, explicitly creates a copy of the template
code and replaces the template components with the concrete type and its
implementation operations. It allows duck typing and lazy binding. Instantiation
implies template code **will** be duplicated.

Unlike monomorphization, this is done **before** type checking completes. Only
when the template is used with a concrete type is the template fully type
checked, and it type checks against the actual concrete type after substituting
it into the template. This means that different instantiations may interpret the
same construct in different ways, and that templates can include constructs that
are not valid for some possible instantiations. However, it also means that some
errors in the template implementation may not produce errors until the
instantiation occurs, and other errors may only happen for **some**
instantiations.

## Specialization

Specialization is essentially overloads for templates/generics. Specialization
is when a template or generic has an overloaded definition for some subset of
concrete arguments used with it.

The key distinction between specialization and normal instantiation is that the
resulting code (and potentially interface!) is customized beyond what can be
done solely through **substitution** -- instead, there is an extra customization
step achieved through **selection**.

With templates, specialization is powerful because it can observe arbitrarily
precise information about the concrete type. In C++, this can be used to bypass
instantiation by creating fully specialized versions that are no longer
dependent in any way and are simply selected when the arguments match.

With generics, this could potentially be used to select between different
implementations and potentially interfaces of the generic code. However, it can
only do so by selecting on different interface constraint sets and/or
properties. In essence, the subsets available are those that can be described
purely in terms of interfaces themselves, not in terms of the concrete types.

While there is nothing fundamentally incompatible about specialization with
generics, even when implemented using witness tables, the result may be
surprising because the selection of the specialized generic happens outside of
the witness table based indirection between the generic code and the concrete
implementation. Provided all selection relies exclusively on interfaces, this
still satisfies the fundamental constraints of generics.

However, type erasure is at least somewhat incompatible with specialization.
Again, because it occurs prior to selecting the type erased generic, at least
some aspects of the type will not have been erased -- specifically those parts
reflected by the interface properties used to select the specialization.

## Conditional conformance

Conditional conformance is when you have a parameterized type that has one API
that it always supports, but satisfies additional interfaces under some
conditions on the type argument.

For example: `Array(T)` might implement `Comparable` if `T` itself implements
`Comparable`, using lexicographical order. This might be supported by way of a
specific "conditionally implements" syntax, or as a special case of a
[templated impls](#templated-impl) facility.

## Interface type parameters versus associated types

Let's say you have an interface defining a container. Different containers will
contain different types of values, and the container API will have to refer to
that "element type" when defining the signature of methods like "insert" or
"find". If that element type is a parameter (input) to the interface type, we
say it is a type parameter; if it is an output, we say it is an associated type.

Type parameter example:

```
interface Stack(Type:$ ElementType) {
  fn Push(Self*: this, ElementType: value);
  fn Pop(Self*: this) -> ElementType;
}
```

Associated type example:

```
interface Stack {
  var Type:$ ElementType;
  fn Push(Self*: this, ElementType: value);
  fn Pop(Self*: this) -> ElementType;
}
```

Associated types are particularly called for when the implementation controls
the type, not the caller. For example, the iterator type for a container is
specific to the container and not something you would expect a user of the
interface to specify.

```
interface Iterator { ... }
interface Container {
  // This does not make sense as an parameter to the container interface,
  // since this type is determined from the container type.
  var Iterator:$ IteratorType;
  ...
  fn Insert(Self*: this, IteratorType: position, ElementType: value);
}
struct ListIterator(Type:$ ElementType) {
  ...
  impl Iterator;
}
struct List(Type:$ ElementType) {
  // Iterator type is determined by the container type.
  var Iterator:$ IteratorType = ListIterator(ElementType);
  fn Insert(Self*: this, IteratorType: position, ElementType: value) {
    ...
  }
  impl Container;
}
```

If [interfaces are concrete types](#interfaces-are-concrete-types), then
commonly there will be a type parameter for each interface that corresponds to
the representation/native type.

Since type parameters are directly under the user's control, it is easier to
express things like "this type parameter is the same for all these interfaces",
and other type constraints.

If you have an interface with type parameters, there is a question of whether a
type can have default impls for different combinations of type parameters, or if
you can only have a single default impl (in which case you can directly infer
the type parameters given just a type implementing the interface). You can
always infer associated types.

## Type constraints

Type constraints restrict which types are legal for template or generic
parameters or associated types. They help define semantics under which they
should be called, and prevent incorrect calls.

We want to be able to say things like:

-   For this container interface we have associated types for iterators and
    elements. The iterator type should also have an element type and it needs to
    match the container's element type.
-   This function accepts two containers. The container types may be different,
    but the element types need to match.
-   An interface may define an associated type that needs to be constrained to
    either implement some (set of) interface(s) or be
    [compatible](terminology.md#compatible-types) with another type.

In general there are a number of different type relationships we would like to
express, and multiple mechanisms we could use to express those constraints:

-   Passing the same name as a type argument to multiple interfaces to ensure
    they agree.
-   Have ways of creating new [type-types](#interfaces-are-type-types) from old
    ones by adding restrictions.
-   Have special "`requires`" clauses with a little language for expressing the
    restrictions we want.
-   others...

## Dependent types (or more generally, values)

A dependent type (or value) is a portion of a generic or template which has
aspects that depend on the particulars of an invocation to the generic or
template. For example, template or generic parameters are clearly dependent
types because they are dependent on the call site.

Indirectly, aspects of dependent types may be used to call other APIs or
similar: this extends the application of dependence. With templates,
instantiation causes a large amount of dependent type cascading across calls.
With generics using interfaces (which are fully type checked), dependence won't
cascade through calls, although it may cascade through interface relationships.

for example, consider this template definition:

```
fn Call[Type:$$ T](T: val) -> Int {
  return val->Call();
}
```

Here, the type of `val` is a dependent type specified by the caller. The type of
`Call` (for example, parameters and return type) is a dependent value because it
depends on the type of `val`.

For contrast, consider this similar generic definition:

```
interface Callable {
  fn Call() -> Int;
}

fn Call[Callable:$ T](T: val) -> Int {
  return val->Call();
}
```

Here, the type of `val` is still a dependent type specified by the caller.
However, the value of `Call` is no longer dependent because its type is defined
by the `Callable` interface.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
