<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon principle: Generics

## Definition

C++ supports
[parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism)
and [generic programming](https://en.wikipedia.org/wiki/Generic_programming)
through templates, which use
[structural typing](https://en.wikipedia.org/wiki/Structural_type_system) to
determine which arguments are valid. Carbon will support (possibly in addition
to a template system) _generics_, which are still a form of parametric
polymorphism for generic programming, but instead of structural typing use
[bounded parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism#Bounded_parametric_polymorphism).
This means the legal arguments and the legal uses of a parameter are both
goverened by explicit bounds on the parameter in a generic function's signature.

### Semantics

A generic function (or type) will take some "generic parameters", which will
frequently be types, and in some cases will be implicit / inferred from the
types of the values of explicit parameters to the function. If a generic
parameter is a type, the generic function's signature can specify constraints
that the caller's type must satisfy (this should also be legal for template
parameters, but less needed and less common). For example, a resizable array
type (like C++'s `std::vector`) might have a generic type parameter with the
constraint that the type must be movable and have a static size. A sort function
might apply to any array whose elements are comparable and movable.

## Goal

Our goal for generics support in Carbon is to get most of the expressive
benefits of C++ templates and open overloading with fewer downsides. We in
particular want to support
[generic programming](https://en.wikipedia.org/wiki/Generic_programming),
including generic algorithms and generic data types. C++ templates are also used
for some things that are out of scope for Carbon generics:
[template metaprogramming](https://en.wikipedia.org/wiki/Template_metaprogramming),
[expression templates](https://en.wikipedia.org/wiki/Expression_templates), and
[variadics](https://en.wikipedia.org/wiki/Variadic_function). We expect to
address those use cases with Carbon metaprogramming or Carbon templates.

## Principles

### Performance

Generics shall provide better code generation (both code size and speed) in all
cases over C++ templates.
[Performance is the top priority for Carbon](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#performance-critical-software),
and we expect to use generics pervasively, and so they can't compromise that
goal in release builds.

### Better compiler experience

Compared to C++ templates:

-   Reduce build times, particularly when developing.
-   Clearer and earlier error reporting.

### Encapsulation

With a template, the implementation is part of the interface and types are only
checked when the function is called and the template is instantiated.

A generic function is type checked when it is defined, and type checking can't
use any information that is only known when the function is instantiated (such
as the exact type). Furthermore, calls to a generic function may be type checked
using only its declaration, not its body. You should be able to call a generic
function using only a forward declaration.

### Dispatch control

Enable simple user control of whether to use dynamic or static dispatch.

**Implementation strategy:** There are two strategies for generating code for
generic functions:

-   Static specialization strategy: Like template parameters, the values for
    generic parameters must be statically known at the callsite, or known to be
    a generic parameter to the calling function. This can generate separate,
    specialized versions of each combination of generic & template arguments, in
    order to optimize for those types / values.
-   Dynamic strategy: Unlike template parameters, we require that it be possible
    to generate a single version of the function that uses runtime dispatch to
    get something semantically equivalent to separate instantiation, but likely
    with different size, build time, and performance characteristics.

The strategy used by the compiler should not be semantically visible to the
user. Users should be able to write a generic version of a function once, and
then be able to control the mix of strategies used to generate the code for that
function. For example, this can be to trade off binary size vs speed (maybe some
specific specializations are needed for performance, but others would just be
code bloat), or support dynamic dispatch when types are not known at compile
time.

### Upgrade path from templates

We want there to be a natural upgrade path from templated code to generic code,
which gives us these additional principles:

-   Converting from a template parameter to a generic parameter should be safe.
    It should either fail to compile or work, never silently change semantics.
-   We should minimize the effort to convert from template code to generic code.
    Ideally it should just require specifying the type constraints, affecting
    just the signature of the function, not its body.
-   It should be legal to call templated code from generic code when it would
    have the same semantics as if called from non-generic code, and an error
    otherwise. This is to allow more templated functions to be converted to
    generics, instead of requiring them to be converted specifically in
    bottom-up order.

### Coherence

Also, we want the generics system to have the _coherence_ property. This means
that the behavior of any type is consistent independent of context such as the
libraries imported into a given file or being inside a generic function.

### No novel name lookup

No novel (generic specific) rules for name lookup:

-   An example of these would be the name lookup rules inside of Rust's traits.
-   Instead, structure generics in a way that reuses existing name lookup
    facilities of the language.

### Closed overloading

One name lookup problem we would like to avoid is caused by open overloading.
Overloading is where you provide multiple implementations of a function with the
same name, and the implementation used in a specific context is determined by
the argument types. Open overloading is overloading where the overload set is
not restricted to a single file or library.

This is commonly used to provide a type-specific implementation of some
operation, but doesn't provide any enforcement of consistency across the
different overloads. It makes the meaning of code dependent on which overloads
are imported, and is at odds with being able to type check a function
generically.

Our goal is to address this use case, known as
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
with a mechanism that does enforce consistency so that type checking is possible
without seeing all implementations.

## Caveats

-   Don't need to provide full flexibility of templates from generics.
    -   We still intend to provide templates for those exceptional cases that
        don't fit inside generics.
    -   If you want [duck](https://en.wikipedia.org/wiki/Duck_typing) /
        [structural](https://en.wikipedia.org/wiki/Structural_type_system)
        typing, that is available by way of templates.
    -   Notably, there is no need to allow a specialization of some generic
        interface for some particular type to actually expose a _different_
        interface (different set of methods or types within the interface for
        example).
-   Features of generics that for example allow you to put constraints on the
    types passed in are presented as part of generics since that is where they
    are most useful, but are still intended to be available for template
    functions.

## Applications of these principles

We write a type constraint in Carbon by saying we restrict to types that
implement specific _Interfaces_. Interfaces serve several purposes:

-   They specify a set of functions (names and signatures) that must be
    available for any type being passed to a generic function, and therefore the
    only functions that may be called in the body of that function.
-   **Maybe:** They will allow other type constraints such as on size, prefix of
    the data layout, specified method implementations, tests that must pass,
    etc.
-   They allow a set of constraints to be given a name so they can be reused.
-   **Maybe:** They implicitly specify the intended semantics and invariants of
    and between those functions. Unlike the function signatures, this contract
    is between the implementers and the consumers of interfaces and is not
    enforced by Carbon itself. For example, a _Draw_ method would mean different
    things when it is part of a _GameResult_ interface versus a _2DImage_
    interface, even if those methods happen to have the same signature.
-   **Maybe:** Have mechanisms to support evolution, such as allowing new
    additions to an interface to have default implementations and/or be marked
    "upcoming" to allow for a period of transition.

There are some desirable capabilities which are in tension with the coherence
property:

-   They should be some way of selecting between multiple implementations of an
    interface for a given type. For example, a _Song_ might support multiple
    orderings (by title, by artist, etc.), which would be represented by having
    multiple implementations of a _Comparable_ interface.
-   In order to allow libraries to be composed, there must be some way of saying
    a type implements an interface that is in another package that the authors
    of the type were unaware of. This is especially important since the library
    a type is defined in may not be able to see the interface definition without
    creating a dependency cycle or layering violation.

This means either that the interface implementations are external to types and
are passed in to generic functions separately, or there is some way to create
multiple types that are compatible with a given value that you can switch
between using casts to select different interface implementations.

### Out of scope

What are we **not** doing with generics, particularly things that some other
language does?

-   Cannot add features to generics that inherently require monomorphization (or
    creating differently specialized code) or templating, that would prevent the
    dynamic compilation strategy.
-   Cannot defer type checking of generic definitions (an implementation
    strategy used by some C++ compilers where the tokens are stashed and
    replayed on use).
-   We won't support runtime specialization as an implementation strategy. That
    is, some language runtimes JIT a specialization when it is first needed, but
    that is out of scope for Carbon.
-   We won't support unbounded type families
    ([such as this example from Swift](https://forums.swift.org/t/ergonomics-generic-types-conforming-in-more-than-one-way/34589/71)).
    This is an obstacle to supporting static dispatch.

Another example of unbounded type families:

```
fn Sort[Comparable T](List(T) list) -> List(T) {
  if (list.size() == 1) return list;
  var List(List(T)) chunks = FormChunks(list, sqrt(list.size()));
  chunks = chunks.ApplyToEach(Sort);
  chunks = Sort(chunks);
  return MergeSortedListOfSortedLists(chunks);
}
```

This, given an implementation of `Comparable` for any list with elements that
are themselves `Comparable`, would recursively call itself to produce a set of
types without bound. That is, calling `Sort` on a `List(Int)` would internally
call `Sort` on a `List(List(Int))` and so on recursively without any static
limit.
