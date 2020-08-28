<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon: Generics - use cases and problem statement

## Basic use cases

We want ways of accomplishing the following tasks:

-   Define an
    [interface](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface).
-   Define an interface with
    [type parameters](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-vs-associated-types)
    (maybe) and/or
    [associated types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-vs-associated-types)
    (almost certainly).
-   Define an interface with
    [type constraints](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#type-constraints),
    such as associated types or type parameters satisfying some interface. Type
    constraints will also be needed as part of generic function definitions, to
    define relationships between type parameters and associated types.
-   Optional, but probably straightforward if we want it: Define an interface
    that
    [extends/refines](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#extendingrefining-an-interface)
    another interface. Similarly we probably want a way to say an interface
    requires an implementation of one or more other interfaces.
-   Define how a type
    [implements](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#impls-implementations-of-interfaces)
    an interface
    ([semantic conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#semantic-vs-structural-interfaces)).
    It should address
    [the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
    e.g. by allowing the impl definition to be completely out of line as long as
    it is defined with either the type or the interface.
-   Define a parameterized implementation of an interface for a family of types.
    This is both for
    [structural conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#semantic-vs-structural-interfaces)
    via
    [templated impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#templated-impl),
    and
    [conditional conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#conditional-conformance).
    That family of types may have generic or regular parameters, so that e.g.
    you could implement a `Printable` interface for arrays of `N` elements of
    `Printable` type `T`, generically for `N` (not separately instantiated for
    each `N`).
-   Control how an interface may be used in order to reserve or abandon rights
    to evolve the interface. See
    [the relevant open question in "Carbon closed function overloading proposal" (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#bookmark=id.hxvlthy3z3g1 -->.
-   Specify a generic explicit (non-type or type) parameter to a function.
-   Specify a generic
    [implicit parameter](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#implicit-parameter)
    to a function.
-   Specify a generic type parameter constrained to conform to an interface. And
    in the function, call methods defined in the the interface on a value of
    that type.
-   Specify a generic type parameter constrained to conform to multiple
    interfaces. And in the function, call methods defined in each interface on a
    value of that type, and pass the value to functions expecting any subset of
    those interfaces. Ideally this would be convenient enough that we could
    favor fewer narrow interfaces and combine them instead of having a large
    number of wide interfaces.
-   Define multiple implementations of an interface for a single type, be able
    to pass those multiple implementations in a single function call, and have
    the function body be able to control which implementation is used when
    calling interface methods. This should work for any interface, without
    requiring cooperation from the interface definition. For example, have a
    function sort songs by artist, then by album, and then by title given those
    three orderings separately.
-   In general, ways of specifying new combinations of interface implementations
    for a type. For example, a way to call a generic function with a value of
    some type, even if the interface and type are defined in different libraries
    unknown to each other, by providing an implementation for that interface in
    some way. This problem is described in
    "[The trouble with typeclasses](https://pchiusano.github.io/2018-02-13/typeclasses.html)".
-   A value with a type implementing a superset of the interfaces required by a
    generic function may be passed to the function without additional syntax
    beyond passing the same value to a non-generic function expecting the exact
    type of the value
    ([subsumption](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#subsumption-and-casting)).
    This should be true for values with types only known generically, as long as
    it is generically known that the type implements a sufficient set of
    interfaces.
-   Define a parameterized entity (such as a function) such that code for it
    will only be generated once.
-   Define a parameterized entity such that code for it will be generated
    separately for each distinct combination of arguments.
-   Convert values of arbitrary types implementing an interface into values of a
    single type that implements that same interface, for a sufficiently
    well-behaved interface.

## Stretch goals

-   A way to define one or a few functions and get an implementation for an
    interface that has more functions (like defining `<`, `>`, `<=`, `>=`, `==`,
    and `!=` in terms of `<=>`, or `++`, `--`, `+`, `-`, and `-=` from `+=`).
    Possibly the "one or few functions" won't even be part of the interface.
-   Define an interface implementation algorithmically -- possibly via a
    function returning an impl, or by defining an
    [adapting type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#adapting-a-type)
    that implements that interface. This could be a solution to the previous
    bullet. Another use case is when there are few standard implementation
    strategies for an interface, and you want to provide those implementations
    in a way that makes it easy for new types to adopt one.
-   Support a way to switch between algorithms based on the capabilities of a
    type. For example, we may want to use different algorithms for random-access
    vs. bidirectional iterators. Similarly, a way to have specialization based
    on type information in a generic like you might do in a template function
    for performance but still would allow type checking. Example: In C++,
    `std::vector<T>::resize()` can use a more efficient algorithm if `T` has a
    `noexcept` move constructor. Can this optimization be allowed from generic
    code since it does not affect the signature of `resize()`, and therefore
    type checking? In a non-release build, it would be semantically equivalent
    but slower to ignore the optimized implementation.
-   As much as possible, switching a templated function to a generic one should
    involve minimal changes to the function body. It should primarily just
    consist of adding constraints to the signature. When changes are needed, the
    compiler will not accept the code without them. No semantics of any code
    will change merely as the result of switching from template to generics. See
    ["Carbon principle: Generics"](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md).

## Very stretch goals

These are more difficult, and possibly optional:

-   Define an interface where the relationship between the input and output
    types is a little complicated. For example, widening multiplication from an
    integer type to one with more bits, or
    `Abs: Complex(SomeIntType) -> SomeFloatType`. One possible strategy is to
    have the return type be represented by an
    [associated type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-vs-associated-types).
-   Define an interface that has multiple related types, like Graph/Nodes/Edges.
    TODO: A concrete combination of `Graph`, `Edge`, and `Node` types that we
    would like to define an interface for. Is the problem when you `Edge` and
    `Node` refer to each other, so you need a forward declaration to break the
    cycle?
-   Impls where the impl itself has state. (from richardsmith@) Use case:
    implementing interfaces for a flyweight in a Flyweight pattern where the
    Impl needs a reference to a key -> info map.
-   "Higher-ranked types": A solution to the problem posed
    [here (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.qvhzlz54obmt -->,
    where we need a representation for a way to go from a type to an
    implementation of an interface parameterized by that type. Examples of
    things we might want to express:

    -   This priority queue's second argument (`QueueLike`) is a function that
        takes a type `U` and returns a type that implements `QueueInterface(U)`:

```
struct PriorityQueue(
    Type:$ T, fn (Type:$ U)->QueueInterface(U):$ QueueLike) {
  ...
}
```

    -  Map takes a container of type `T` and function from `T` to `V` into
       a container of type `V`:

```
fn Map[Type:$ T,
       fn (Type:$ U)->StackInterface(U):$ StackLike,
       Type:$ V]
    (StackLike(T)*: x, fn (T)->V: f) -> StackLike(V) { ... }
```

## Last word: programming model

These mechanisms need to have an underlying programming model that allows users
to predict how to do these things, how to compose these things, and what
expressions are legal.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
