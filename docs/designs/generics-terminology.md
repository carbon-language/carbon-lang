<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon: Generics - Terminology and problem statement

## Terminology

Please see the
[Carbon principle: Generics](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md)
and
[Templates and generics: distinctions (TODO)](#broken-links-footnote)<!-- T:Templates and generics: distinctions -->
docs for a description of generics in Carbon, specifically the goals and
differences from templates. Additionally, the upcoming "Carbon Generic v2" doc
has
"[What are generics?](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md#what-are-generics)"
and
"[Goals: Generics](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md#goals-generics)"
sections.

TODO: Runtime vs. compile-time trade offs, worry about things like size of value
unknown producing much slower code -- needs guardrails, explicit opt-in?

### Parameterized language constructs

Generally speaking, when we talk about either templates or a generics system, we
are talking about generalizing some language construct by adding a parameter to
it. Language constructs here primarily would include functions and types, but we
may want to support parameterizing other language constructs like
[interfaces](#interface-type-parameters-vs-associated-types).

This parameter broadens the scope of the language construct on an axis defined
by that parameter, effectively defining a family of functions (or whatever)
instead of a single one.

### Generic vs. template arguments

When we are distinguishing between generics and templates in Carbon, it is on an
argument by argument basis. A single function can take a mix of regular,
generic, and template arguments.

- **Regular arguments** are designated using "&lt;type>`:` &lt;name>" syntax (or
  "&lt;value>").
- **Generic arguments** are temporarily designated using an additional `$` after
  the `:` (so it is "&lt;type>`:$` &lt;name>"). However, the `$` symbol is not
  easily typed on non-US keyboards, so we will definitely switch to some other
  syntax. Some possibilities that
  [have been suggested (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:# -->
  are: `:!`, `:@`, `:#`, and `::`.
- **Template arguments** are temporarily designated using "&lt;type> `:$$`
  &lt;name>", for similar reasons.

Expected difference between generics and templates:

<table>
  <tr>
   <td><strong>Generics</strong>
   </td>
   <td><strong>Templates</strong>
   </td>
  </tr>
  <tr>
   <td>parametric polymorphism
   </td>
   <td>ad hoc polymorphism
   </td>
  </tr>
  <tr>
   <td>constrained/bounded genericity
   </td>
   <td>unconstrained genericity, though you may still specify constraints
   </td>
  </tr>
  <tr>
   <td>name lookup resolved for definitions in isolation ("early")
   </td>
   <td>some name lookup may require information from calls (name lookup may be "late")
   </td>
  </tr>
  <tr>
   <td>sound to typecheck definitions in isolation ("early")
   </td>
   <td>complete type checking may require information from calls (may be "late")
   </td>
  </tr>
  <tr>
   <td>supports separate type checking; may also support separate compilation, for example when implemented using dynamic witness tables
   </td>
   <td>separate compilation only to the extent that C++ supports it
   </td>
  </tr>
  <tr>
   <td>allowed but not required to be implemented using dynamic dispatch
   </td>
   <td>does not support implementation via dynamic dispatch, just static via instantiation
   </td>
  </tr>
</table>

#### Parametric vs. Ad Hoc polymorphism

From [Wikipedia](https://en.wikipedia.org/wiki/Parametric_polymorphism): Using
parametric polymorphism, a function or a data type can be written generically so
that it can handle values _identically_ without depending on their type.

From [Wikipedia](https://en.wikipedia.org/wiki/Ad_hoc_polymorphism): ad hoc
polymorphism is a kind of polymorphism in which polymorphic functions can be
applied to arguments of different types, because a polymorphic function can
denote a number of distinct and potentially heterogeneous implementations
depending on the type of argument(s) to which it is applied. It is also known as
"function overloading" or "operator overloading".

Contrast with [subtype polymorphism](https://en.wikipedia.org/wiki/Subtyping),
where different descendants of a base class can provide different
implementations of a method, subject to some compatibility restrictions on the
signature.

In Carbon, we expect there to be a compile error if overloading of some name
prevents a generic function from being typechecked from its definition alone.
For example, let's say we have some overloaded function called `F` that has two
overloads (note that this is not real proposed Carbon syntax):

```
fn F[Type:$$ T, requires T != Int](T: x) -> T;
fn F(Int: x) -> Bool;
```

A generic function `G` can call `F` with a type like `T*` that can not possibly
call the `F(Int)` overload for `F`, and so it can consistently determine the
return type of `F`. But `G` can't call `F` with an argument that could match
either overload. (I think it is undecided what to do in the situation where `F`
is overloaded, but the signatures are consistent and so callers could still
typecheck calls to `F`.)

#### Constrained/bounded vs. Unconstrained/unbounded genericity

We will allow some way of specifying constraints as part of a function (or type
or other parameterized language construct). These constraints are a limit on
what callers are allowed to pass in. The distinction between constrained/bounded
and unconstrained/unbounded genericity is whether the body of the function is
limited to just those operations that are guaranteed by the constraints.

With templates using unconstrained/unbounded genericity, you may perform any
operation in the body of the function, and they will be checked against the
specific types used in calls. You can still have constraints, but they will only
be used to resolve overloaded calls to the template and provide clearer error
messages.

With generics using constrained/bounded genericity, the function body can be
checked against the signature at the time of definition. Note that it is still
perfectly permissible to have no constraints on a type; that just means that you
can only perform operations that work for all types (such as manipulate pointers
to values of that type) in the body of the function.

#### Definition checking

Definition checking is the process of semantically checking the definition of
parameterized code for correctness _independently_ of any particular arguments.
It includes type checking and other semantic checks. Typically, all
non-dependent semantics can be checked in the definition, but it is possible to
add constraints (via generics) that increase how much of the definition can be
checked. If anything remains unchecked (normal for templates), instantiating the
implementation it requires instantiation (that may fail) is required in order to
check its correctness once specific arguments can be substituted into the
parameters.

##### Complete definition checking

Complete definition checking is when the definition can be _fully_ semantically
checked, including type checking. It is an especially useful property because it
enables _separate_ semantic checking of the definition, a prerequisite to
separate compilation. It also enables implementation strategies that don’t
instantiate the implementation (for example,
[type erasure](#type-erasure-eg-java) or
[dynamic-dispatch witness tables](#dynamic-dispatch-witness-table)).

##### Early vs. late type checking

Early type checking is where expressions and statements are type checked when
the definition of the function body is compiled, as part of definition checking.
This occurs for regular and generic values.

Late type checking is where expressions and statements may only be fully
typechecked once calling information is known. Late type checking delays
complete definition checking. This occurs for template dependent values.

### Implicit argument

An implicit argument is listed in the optional `[` `]` section right after the
function name in a function signature:

`fn` &lt;name> `[` &lt;implicit arguments> `](` &lt;explicit arguments `) ->`
&lt;return type>

Implicit arguments are determined as a result of pattern matching the explicit
arguments to the values (generally the types of those values). See more
[here](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md#implicit-arguments).

### Interface

An interface is an API constraint used in a function signature to provide
encapsulation. Encapsulation here means that callers of the function only need
to know about the interface requirements to call the function, not anything
about the implementation of the function body, and the compiler can check the
function body without knowing anything more about the caller. Callers of the
function provide a value that has an implementation of the API and the body of
the function may then use that API (and nothing else).

There are a few different possible interface programming models.

#### Semantic vs. structural interfaces

A "structural" interface is one where we say a type satisfies the interface as
long as it has members with a specific list of names, and for each name it must
have some type / signature. A type can satisfy a structural interface without
ever naming that interface, just by virtue of having members with the right
form.

A "semantic" interface is one where we say a type can only satisfy an interface
if there is some explicit statement saying so, for example by defining an
[impl](#impls-implementations-of-interfaces). This allows "satisfies the
interface" to have additional semantic meaning beyond what is directly checkable
by the compiler. For example, knowing whether the "Draw" function means "render
an image to the screen" or "take a card from the top of a deck of cards"; or
that a `+` operator is commutative (and not, say, string concatenation).

#### What kind of values are interfaces?

If you use the name of an interface after it has been defined, what role does it
play grammatically? What kind of value is it?

##### Interfaces are concrete types

In one programming model, an interface is thought of as a forward declaration of
a concrete type. For example, "`interface Comparable(Type:$ T) for T`" means
that there is a type `Comparable(T)` defined for some types `T`, and for those
`T` where it is defined, you may implicitly cast between `T` and `Comparable(T)`
(they are [compatible types](#compatible-types)). Here, `Comparable(T)` is a
[facet type](#facet-types) for `T`, so `Comparable(T)` has the API defined by
the interface. You might define a generic function like so:

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
["Interfaces are concrete facet types"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md#interfaces-are-concrete-types)
in the Generics proposal for more on this programming model.

##### Interfaces are type-types

In other programming models, an interface is a type-type, i.e., an interface is
a type whose values are also types (satisfying some properties). For example, we
might define a generic function like so:

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

###### Facet type-types

Type `T` satisfies `InterfaceName` if it is a
[facet type](#invoking-interface-methods) of `InterfaceName` for some
`NativeType`. That means that `NativeType` provides an implementation of
`InterfaceName` and `T` is the type compatible with `NativeType` that exposes
that specific implementation, so the API of `T` matches `InterfaceName` instead
of `NativeType`. TODO: deep dive describing this programming model; currently
the best we have is
[Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.3kqiirqlj97f -->.

###### Type-types parameterized by reprs

Type `T` satisfies `InterfaceName(R)` if it is a facet type (as above) of
`InterfaceName` that is compatible with (shares a representation with) type `R`.
This is good for representing multiple implementations of a single interface and
implementations of multiple interfaces (in both cases sharing the type `R`).
However it creates problems when you try and infer `R` from the types of the
values passed in by the user, since it is unclear which `R` to pick from the set
of types sharing the same representation.

See
["Type-types parameterized by reprs"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md#type-types-parameterized-by-reprs)
in the Generics proposal for more on this programming model.

##### Interfaces are opaque

Another possibility is that an interface is an opaque key. This has the
advantage that it doesn't privilege any one use of an interface. Any use of an
interface is in the context of a language construct that explicitly takes an
interface as one of its arguments and the use of the construct makes it clear
what role the interface is playing.

An example of a programming model like this is presented in
[Carbon closed function overloading proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#heading=h.qkop7hrup3jx -->.

### Impls: Implementations of interfaces

An _impl_ is an implementation of an interface for a specific type. It is the
place where the function bodies are defined, values for associated types, etc.
are given. A given generics programming model may support default impls, named
impls, or both. Impls are mostly associated with semantic interfaces; structural
interfaces define conformance implicitly instead of by requiring an impl to be
defined.

#### Default impl

A default impl is an implementation associated with a (type, interface) pair
without a separate name.

Generally speaking, in order to make sure that every place looking up the
default impl for a particular interface for a type sees the same result, we
require default impls to be defined either with the interface or the type.

The main benefit of default impls is that it allows
[subsumption](#subsumption-and-casting) to work. That is, by allowing the
compiler to look up the impl for a (type, interface) combination automatically,
you allow passing of values to generic functions transparently, without any
explicit statement of how to satisfy the argument's interface requirements. This
of course means that any given type can have at most one default impl for any
given interface.

#### Named impl

A named impl is an implementation of a (type, interface) pair that is given its
own name. The impl won't be used by default/implicitly; it will only be used as
a result of looking up its name, which means it can be defined anywhere. There
is no restriction that a (type, interface) pair be limited to one named impl.

#### Templated impl

This is where there is a templated definition of an impl, defined along with an
interface, that specifies a common implementation for an interface for a variety
of types. For example, it could be used to say that any type implementing these
other two interfaces automatically gets an implementation for this interface, or
to say that you can satisfy this interface structurally.

### Compatible types

Two types are compatible if they have the same representation, even if they
expose different APIs. The representation of a type includes a bunch of facts
about how the bits of a value of that type are represented in memory. It also
includes things that the compiler can't see directly that involve the
interpretation of those bits, such as the invariants that the type maintains.

We can't just say two types are compatible based on structural reasons. Instead,
we have specific constructs that create compatible types from existing types in
ways that encourage preserving the programmer's intended semantics and
invariants, such as implementing the API of the new type by calling (public)
methods of the original API, instead of accessing any private implementation
details.

### Invoking interface methods

Given a function with a generic argument whose type is known to satisfy some
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

#### Facet types

One approach is to cast values to a facet type. A facet type is a
[compatible type](#compatible-types) to the original type written by the user,
but has an API exactly matching one interface.

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

Facet types may be most useful with semantic interfaces. You could also imagine
facet types in a programming model with structural interfaces and no impls. In
this case, the facet type would be a projection of a type's API onto the subset
defined by the interface. However, in this situation you would have no need to
maintain separate namespaces, and so union types would make more sense.

#### Union Types

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

#### Separate Impls

Separate impls use non-type objects that can be passed around as values. The
type of an impl value is parameterized by both an interface and a compatible
type. E.g., if `x` is the name of an impl value for interface `I` and compatible
type `T` and `F` is the name of function defined in `I`, and `y` is a value of
type `T`, you might have an explicit syntax where you call the `F` from `x` on
`y` using `x.F(y)`, or an implicit syntax like `with (x) { y.F(); }`.

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

### Subsumption and casting

Both subsumption and casting are different names for changing the type of a
value to a compatible type.

Subsumption is an automatic or implicit conversion that happens to argument
values when calling a function or when assigning a value to a variable of a
different type.

Casting is indicated explicitly via some syntax in the source code. You might
use a cast to switch between [type adaptations](#adapting-a-type), or to be
explicit where an implicit cast would otherwise occur. For now, we are saying
"`x as y`" is the syntax in Carbon for casting the value `x` to the type `y`.

Note that subsumption is a bit like coercion, except we want to make it clear
that the representation of the value is not changing, just its type as reflected
in the API available to manipulate the value.

### Adapting a type

A type can be adapted by creating a new type that is
[compatible](#compatible-types) with an existing type, but has a different API.
In particular, the new type might implement different interfaces or provide
different implementations of the same interfaces.

Unlike extending a type (as with C++ class inheritance), you are not allowed to
add new data fields onto the end of the representation -- you may only change
the API. This means that it is safe to [cast](#subsumption-and-casting) a value
between those two types without any dynamic checks or danger of
[object slicing](https://en.wikipedia.org/wiki/Object_slicing).

### Extending/refining an interface

An interface can be extended by defining an interface that includes the full API
of another interface, plus some additional API. Types implementing the extended
interface should automatically be considered to have implemented the narrower
interface.

### Implementation strategies for generics

Witness tables, type erasure, monomorphization and instantiation all describe
methods under which we could implement generics. They are each trying to address
how generics perform operations on the type provided by a caller.

#### Witness tables (e.g., Swift and Carbon Generics)

For witness tables, calls made on the generic's argument are compiled into a
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

##### Dynamic-dispatch witness table

For dynamic-dispatch witness tables, actual function pointers are formed and
used as a dynamic, runtime indirection. As a result, the generic code **will
not** be duplicated for different witness tables.

##### Static-dispatch witness table

For static-dispatch witness tables, the implementation is required to collapse
the table indirections at compile time. As a result, the generic code **will**
be duplicated for different witness tables.

Static-dispatch may be implemented as a performance optimization for
dynamic-dispatch that increases generated code size. The final compiled output
may not retain the witness table.

#### Type erasure (e.g., Java)

Type erasure is similar to dynamic-dispatch witness tables, but it goes further
and pushes the abstraction all the way to runtime. The actual type is completely
unknown at compile time. Type erasure implies generic code **will not** be
duplicated.

A fundamental distinction from a witness table is that the actual type cannot be
recovered. When dynamic-dispatch witness tables are used, they may still model
the actual type through some dynamic system rather than ensuring it is fully
opaque. Type erasure removes that option, which can
[cause problems](https://en.wikipedia.org/wiki/Generics_in_Java#Problems_with_type_erasure).

#### Monomorphization (e.g., Rust)

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

#### Instantiation (e.g., C++ and Carbon Templates)

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

### Specialization

Specialization is essentially overloads for templates/generics. Specialization
is when a template or generic has an overloaded definition for some subset of
concrete parameters used with it.

The key distinction between specialization and normal instantiation is that the
resulting code (and potentially interface!) is customized beyond what can be
done solely through **substitution** -- instead, there is an extra customization
step achieved through **selection**.

With templates, specialization is powerful because it can observe arbitrarily
precise information about the concrete type. In C++, this can be used to bypass
instantiation by creating fully specialized versions that are no longer
dependent in any way and are simply selected when the parameters match.

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

### Conditional conformance

Conditional conformance is when you have a parameterized type that has one API
that it always supports, but satisfies additional interfaces under some
conditions about the type parameter.

For example: `Array(T)` might implement `Comparable` if `T` itself implements
`Comparable`, using lexicographical order. This might be supported via a
specific "conditionally implements" syntax, or as a special case of a
[templated impls](#templated-impl) facility.

### Interface type parameters vs. associated types

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

\
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
  // This does not make sense as an argument to the container interface,
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

### Type constraints

Type constraints restrict type information about template or generic parameters.
They help define semantics under which they should be called, and prevent
incorrect calls.

We want to be able to say things like:

- For this container interface we have associated types for iterators and
  elements. The iterator type should also have an element type and it needs to
  match the container's element type.
- This function accepts two containers. The container types may be different,
  but the element types need to match.
- An interface may define an associated type that needs to be constrained to
  either implement some (set of) interface(s) or be
  [compatible](#compatible-types) with another type.

In general there are a number of different type relationships we would like to
express, and multiple mechanisms we could use to express those constraints:

- Passing the same name as a type parameter to multiple interfaces to ensure
  they agree.
- Have ways of creating new [type-types](#interfaces-are-type-types) from old
  ones by adding restrictions.
- Have special "`requires`" clauses with a little language for expressing the
  restrictions we want.
- others...

### Dependent types (or more generally, values)

A dependent type (or value) is a portion of a generic or template which has
aspects that depend on the particulars of an invocation to the generic or
template. For example, template or generic parameters are clearly dependent
types because they are dependent on the call site.

Indirectly, aspects of dependent types may be used to call other APIs or
similar: this extends the application of dependence. With templates,
instantiation causes a large amount of dependent type cascading across calls.
With generics using interfaces (which are fully type checked), dependence won't
cascade through calls, although it may cascade through interface relationships.

e.g., consider this template definition:

```
fn Call[Type:$$ T](T: val) -> Int {
  return val->Call();
}
```

Here, the type of `val` is a dependent type specified by the caller. The type of
`Call` (e.g., parameters and return type) is a dependent value because it
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

## Problem statement

We want ways of accomplishing the following tasks:

- Define an [interface](#interface).
- Define an interface with
  [type parameters](#interface-type-parameters-vs-associated-types) (maybe)
  and/or [associated types](#interface-type-parameters-vs-associated-types)
  (almost certainly).
- Define an interface with [type constraints](#type-constraints), such as
  associated types or type parameters satisfying some interface. Type
  constraints will also be needed as part of generic function definitions, to
  define relationships between type parameters and associated types.
- Optional, but probably straightforward if we want it: Define an interface that
  [extends/refines](#extendingrefining-an-interface) another interface.
  Similarly we probably want a way to say an interface requires an
  implementation of one or more other interfaces.
- Define how a type [implements](#impls-implementations-of-interfaces) an
  interface ([semantic conformance](#semantic-vs-structural-interfaces)). It
  should address
  [the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
  e.g. by allowing the impl definition to be completely out of line as long as
  it is defined with either the type or the interface.
- Define a parameterized implementation of an interface for a family of types.
  This is both for [structural conformance](#semantic-vs-structural-interfaces)
  via [templated impls](#templated-impl), and
  [conditional conformance](#conditional-conformance). That family of types may
  have generic or regular parameters, so that e.g. you could implement a
  `Printable` interface for arrays of `N` elements of `Printable` type `T`,
  generically for `N` (not separately instantiated for each `N`).
- Control how an interface may be used in order to reserve or abandon rights to
  evolve the interface. See
  [the relevant open question in "Carbon closed function overloading proposal" (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#bookmark=id.hxvlthy3z3g1 -->.
- Specify a generic explicit (non-type or type) argument to a function.
- Specify a generic [implicit argument](#implicit-argument) to a function.
- Specify a generic type argument constrained to conform to an interface. And in
  the function, call methods defined in the the interface on a value of that
  type.
- Specify a generic type argument constrained to conform to multiple interfaces.
  And in the function, call methods defined in each interface on a value of that
  type, and pass the value to functions expecting any subset of those
  interfaces. Ideally this would be convenient enough that we could favor fewer
  narrow interfaces and combine them instead of having a large number of wide
  interfaces.
- Define multiple implementations of an interface for a single type, be able to
  pass those multiple implementations in a single function call, and have the
  function body be able to control which implementation is used when calling
  interface methods. This should work for any interface, without requiring
  cooperation from the interface definition. For example, have a function sort
  songs by artist, then by album, and then by title given those three orderings
  separately.
- In general, ways of specifying new combinations of interface implementations
  for a type. For example, a way to call a generic function with a value of some
  type, even if the interface and type are defined in different libraries
  unknown to each other, by providing an implementation for that interface in
  some way. This problem is described in
  "[The trouble with typeclasses](https://pchiusano.github.io/2018-02-13/typeclasses.html)".
- A value with a type implementing a superset of the interfaces required by a
  generic function may be passed to the function without additional syntax
  beyond passing the same value to a non-generic function expecting the exact
  type of the value ([subsumption](#subsumption-and-casting)). This should be
  true for values with types only known generically, as long as it is
  generically known that the type implements a sufficient set of interfaces.
- Define a parameterized entity (such as a function) such that code for it will
  only be generated once.
- Define a parameterized entity such that code for it will be generated
  separately for each distinct combination of arguments.
- Convert values of arbitrary types implementing an interface into values of a
  single type that implements that same interface, for a sufficiently
  well-behaved interface.

Stretch goals:

- A way to define one or a few functions and get an implementation for an
  interface that has more functions (like defining `&lt;`, `>`, `&lt;=`, `>=`,
  `==`, and `!=` in terms of `&lt;=>`, or `++`, `--`, `+`, `-`, and `-=` from
  `+=`). Possibly the "one or few functions" won't even be part of the
  interface.
- Define an interface implementation algorithmically -- possibly via a function
  returning an impl, or by defining an [adapting type](#adapting-a-type) that
  implements that interface. This could be a solution to the previous bullet.
  Another use case is when there are few standard implementation strategies for
  an interface, and you want to provide those implementations in a way that
  makes it easy for new types to adopt one.
- Support a way to switch between algorithms based on the capabilities of a
  type. For example, we may want to use different algorithms for random-access
  vs. bidirectional iterators. Similarly, a way to have specialization based on
  type information in a generic like you might do in a template function for
  performance but still would allow type checking. Example: In C++,
  `std::vector&lt;T>::resize()` can use a more efficient algorithm if `T` has a
  `noexcept` move constructor. Can this optimization be allowed from generic
  code since it does not affect the signature of `resize()`, and therefore type
  checking? In a non-release build, it would be semantically equivalent but
  slower to ignore the optimized implementation.
- As much as possible, switching a templated function to a generic one should
  involve minimal changes to the function body. It should primarily just consist
  of adding constraints to the signature. When changes are needed, the compiler
  will not accept the code without them. No semantics of any code will change
  merely as the result of switching from template to generics. See
  ["Carbon principle: Generics"](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md).

Very stretch goals (these are more difficult, and possibly optional):

- Define an interface where the relationship between the input and output types
  is a little complicated. For example, widening multiplication from an integer
  type to one with more bits, or `Abs: Complex(SomeIntType) -> SomeFloatType`.
  One possible strategy is to have the return type be represented by an
  [associated type](#interface-type-parameters-vs-associated-types).
- Define an interface that has multiple related types, like Graph/Nodes/Edges.
  TODO: A concrete combination of `Graph`, `Edge`, and `Node` types that we
  would like to define an interface for. Is the problem when you `Edge` and
  `Node` refer to each other, so you need a forward declaration to break the
  cycle?
- Impls where the impl itself has state. (from richardsmith@) Use case:
  implementing interfaces for a flyweight in a Flyweight pattern where the Impl
  needs a reference to a key -> info map.
- "Higher-ranked types": A solution to the problem posed
  [here (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.qvhzlz54obmt -->,
  where we need a representation for a way to go from a type to an
  implementation of an interface parameterized by that type. Examples of things
  we might want to express: _
  `struct PriorityQueue( \ Type:$ T, fn (Type:$ U)->QueueInterface(U):$ QueueLike) { \ ... \ }`
  _ `fn Map[Type:$ T, fn (Type:$ U)->StackInterface(U):$ StackLike,`

          ```
                 Type:$ V]
      (StackLike(T)*: x, fn (T)->V: f) -> StackLike(V) { ... }

          ```

These mechanisms need to have an underlying programming model that allows users
to predict how to do these things, how to compose these things, and what
expressions are legal.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
