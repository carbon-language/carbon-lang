# Carbon: Generics goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
    -   [Definition of generics](#definition-of-generics)
    -   [Generic parameters](#generic-parameters)
    -   [Interfaces](#interfaces)
    -   [Relationship to templates](#relationship-to-templates)
-   [Goals](#goals)
    -   [Use cases](#use-cases)
        -   [Generic programming](#generic-programming)
        -   [Upgrade path from C++ abstract interfaces](#upgrade-path-from-c-abstract-interfaces)
        -   [Dependency injection](#dependency-injection)
        -   [Generics instead of open overloading](#generics-instead-of-open-overloading)
        -   [Use cases that are out of scope](#use-cases-that-are-out-of-scope)
    -   [Performance](#performance)
    -   [Better compiler experience](#better-compiler-experience)
    -   [Encapsulation](#encapsulation)
    -   [Predictability](#predictability)
    -   [Dispatch control](#dispatch-control)
    -   [Upgrade path from templates](#upgrade-path-from-templates)
    -   [Coherence](#coherence)
    -   [No novel name lookup](#no-novel-name-lookup)
    -   [Learn from others](#learn-from-others)
    -   [Interfaces are nominal](#interfaces-are-nominal)
    -   [Interop and evolution](#interop-and-evolution)
-   [Non-goals, caveats, limitations, and out-of-scope issues](#non-goals-caveats-limitations-and-out-of-scope-issues)

<!-- tocstop -->

## Background

### Definition of generics

C++ supports
[parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism)
and [generic programming](https://en.wikipedia.org/wiki/Generic_programming)
through templates, which use
[compile-time duck typing](https://en.wikipedia.org/wiki/Duck_typing#Templates_or_generic_types)
(which might alternatively be described as usage-based
[structural typing](https://en.wikipedia.org/wiki/Structural_type_system)) to
determine which arguments are valid. Carbon will support (possibly in addition
to a template system) _generics_, which are still a form of parametric
polymorphism for generic programming, but supports definition checking. This
means that the body of a function can be type checked when it is defined without
any information from the call site, such as the actual argument values of
generic parameters. This is accomplished by using
[bounded parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism#Bounded_parametric_polymorphism)
instead of compile-time duck typing. This means the legal arguments and the
legal uses of a parameter are both goverened by explicit bounds on the parameter
in a generic function's signature.

### Generic parameters

A generic function (or type) will take some "generic parameters", which will
frequently be types, and in some cases will be implicit / inferred from the
types of the values of explicit parameters to the function. If a generic
parameter is a type, the generic function's signature can specify constraints
that the caller's type must satisfy. For example, a resizable array type (like
C++'s `std::vector`) might have a generic type parameter with the constraint
that the type must be movable and have a static size. A sort function might
apply to any array whose elements are comparable and movable. A constraint might
involve multiple generic parameters. For example, a merge function might apply
to two arbitrary containers so long as their elements have the same type.

### Interfaces

We need some way to express the bounds on a generic type parameter. In Carbon we
express these "type constraints" by saying we restrict to types that implement
specific _interfaces_. Interfaces describe an API a type could implement. For
example it might specify a set of functions, including names and signatures. A
type implementing an interface may be passed as a generic type argument to a
function that has that interface as a requirement of its generic type parameter.
And then the functions defined in the interface may be called in the body of the
function. Further, interfaces have names that allow them to be reused.

This is much like these compile-time and run-time constructs from other
programming languages:

-   [Rust's traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
-   [Swift's protocols](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html)
-   [Java interfaces](<https://en.wikipedia.org/wiki/Interface_(Java)>)
-   [C++ concepts](<https://en.wikipedia.org/wiki/Concepts_(C%2B%2B)>)
    (compile-time only)
-   [Abstract base classes](<https://en.wikipedia.org/wiki/Class_(computer_programming)#Abstract_and_concrete>)
    in C++, etc. (run-time only)
-   [Go interfaces](https://gobyexample.com/interfaces) (run-time only)

In addition to specifying the methods available on a type, we may in the future
expand the role of interfaces to allow other type constraints such as on size,
prefix of the data layout, specified method implementations, tests that must
pass, etc. This might be part of making interfaces as expressive as classes, as
part of a strategy to migrate to a future version of Carbon that uses interfaces
instead of rather than in addition to standard inheritance-and-classes
object-oriented language support. For the moment, though, this is out of scope.

### Relationship to templates

The question of whether Carbon has direct support for templates is out of scope
for this document. The generics design is not completely separate from
templates, so it is written as if Carbon will have its own templating system. It
is assumed to be similar to C++ templates with some specific changes:

-   It may have some limitations to be more compatible with generics, much like
    how we [restrict overloading](#generics-instead-of-open-overloading) below.
-   We likely will have a different method of selecting between different
    template instantiations, since SFINAE makes it difficult to deliver high
    quality compiler diagnostics.

We assume Carbon will have templates for a few different reasons:

-   Carbon generics will definitely have to interact with _C++_ templates, and
    many of the issues will be similar.
-   We want to leave room in the design for templates, since it seems like it
    would be easier to remove templates if they are not pulling their weight
    than figure out how to add them in if they turn out to be needed.
-   We may want to have templates in Carbon as a temporary measure, to make it
    easier for users to transition off of C++ templates.

## Goals

Our goal for generics support in Carbon is to get most of the expressive
benefits of C++ templates and open overloading with fewer downsides.

### Use cases

To clarify the expressive range we are aming for, here are some specific use
cases we expect Carbon generics to cover.

#### Generic programming

We in particular want to support
[generic programming](https://en.wikipedia.org/wiki/Generic_programming),
including:

-   Containers (arrays, maps, lists), and more complicated data structures like
    trees and graphs.
-   Algorithms (sort, search)
-   Wrappers (optional, variant, expected/result, smart pointers)
-   Parameterized numeric types (`std::complex<T>`)
-   Configurable / parametric APIs such as the storage-customized `std::chrono`
    APIs
-   [Policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design)

These would generally involve static, compile-time type arguments, and so would
generally be used with [static dispatch](#dispatch-control).

#### Upgrade path from C++ abstract interfaces

Interfaces in C++ are often represented by abstract base classes. Generics
should offer an alternative that does not rely on inheritance. This means looser
coupling and none of the problems of multiple inheritance. In fact,
[Sean Parent](https://sean-parent.stlab.cc/papers-and-presentations/#better-code-runtime-polymorphism)
(and others) advocate for runtime polymorphism patterns in C++ that avoid
inheritance, since it can cause runtime performance, correctness, and code
maintenance problems in some situations. Carbon generics provide an alternative
for those situations inheritance doesn't handle as well. In particular, we would
like Carbon generics to supplant the need to support multiple inheritance in
Carbon,

will be able to represent this form of polymorphism without all the boilerplate
and complexity required in C++, to

This is a case that would use [dynamic dispatch](#dispatch-control).

#### Dependency injection

Types which only support subclassing for test stubs and mocks, as in
["dependency injection"](https://en.wikipedia.org/wiki/Dependency_injection),
should be able to easily migrate to generics. This extends outside the realm of
testing, allowing general configuration of how dependencies can be satisfied.
For example, generics might be used to configure how a library writes logs.

This would allow you to avoid the runtime overhead of virtual functions, using
[static dispatch](#dispatch-control) without the
[poor build experience of templates](#better-compiler-experience).

#### Generics instead of open overloading

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
with a mechanism within generics that does enforce consistency so that type
checking is possible without seeing all implementations. This will be Carbon's
replacement for open overloading. As a result, Carbon generics will need to be
able to support operator overloading.

#### Use cases that are out of scope

C++ templates are also used for some things that are out of scope for Carbon
generics such as
[template metaprogramming](https://en.wikipedia.org/wiki/Template_metaprogramming).
We expect to address those use cases with metaprogramming or templates in
Carbon. We will also not require Carbon generics to support
[expression templates](https://en.wikipedia.org/wiki/Expression_templates) or
[variadics](https://en.wikipedia.org/wiki/Variadic_function), those are both
non-goals. It would be fine for our generics system to support these features,
but they won't drive any accommodation in the generics design, at least until we
have some resolution about templates in Carbon.

### Performance

Generics shall provide at least as good code generation (both code size and
execution speed) in all cases over C++ templates.
[Performance is the top priority for Carbon](../../project/goals.md#performance-critical-software),
and we expect to use generics pervasively, and so they can't compromise that
goal in release builds.

**Nice to have:** There are cases where we should aim to do better than C++
templates. For example, the additional structure of generics should make it
easier to reduce generated code duplication, reducing code size and cache
misses.

### Better compiler experience

Compared to C++ templates, we expect to reduce build times, particularly when
developing. We also expect the compiler to be able to report clearer errors, and
report them earlier in the build process.

One source of improvement is that the bodies of generic functions and types can
be type checked once when they are defined, instead of every time they are used.
This is both a reduction in the total work done, and how errors can be reported
earlier. On use, the errors can be a lot clearer since they will be of the form
"argument did not satisfy function's contract as stated in its signature"
instead of "substitution failed at this line of the function's implementation."

**Nice to have:** In development builds, we will have the option of using
[dynamic dispatch](#dispatch-control) to reduce build times. We may also be able
to reduce the amount of redundant compilation work even with the
[static strategy](#dispatch-control) by identifying instantiations with the same
arguments and only generating code for them once.

### Encapsulation

With a template, the implementation is part of the interface and types are only
checked when the function is called and the template is instantiated.

A generic function is type checked when it is defined, and type checking can't
use any information that is only known when the function is instantiated (such
as the exact type). Furthermore, calls to a generic function may be type checked
using only its declaration, not its body. You should be able to call a generic
function using only a forward declaration.

### Predictability

A general property of generics is they are more predictable than templates. They
make clear when a type satisfies the requirements of a function; they have a
documented contract. Further, that contract is enforced by the compiler, not
sensitive to implementation details in the function body. This eases evolution
by reducing the impact of [Hyrum's law](https://www.hyrumslaw.com/).

**Nice to have:** We also want well-defined boundaries between what is legal and
not. This is "will my code be accepted by the compiler" predictability. We would
prefer to avoid algorithms in the compiler with the form "run for up to N steps
and report an error if it isn't resolved by then." For example, C++ compilers
will typically have a template recursion limit. With generics, these problems
arise due to trying to reason whether something is legal in all possible
instantiations, rather than with specific, concrete types. Some of this is
likely unavoidable (or too costly to avoid), as most existing generics systems
[have undecidable aspects to their type system](https://3fx.ch/typing-is-hard.html)
(including [Rust](https://sdleffler.github.io/RustTypeSystemTuringComplete/) and
[Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024)).

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

By default, we expect the implementation strategy to be controlled by the
compiler, and not semantically visible to the user. For example, the compiler
might use the static strategy for release builds and the dynamic strategy for
development. Or it might choose between them on a more granular level (maybe
some specific specializations are needed for performance, but others would just
be code bloat) based on code analysis or profiling.

In addition, the user may opt in to using the dynamic strategy in specific
cases. This could be just to control binary size in cases the user knows are not
performance sensitive, or it could be to get the additional capability of
operating on values with dynamic types. We may need to restrict this in various
ways to maintain efficiency, like Rust does with object-safe traits.

We also anticipate that the user may want to force the compiler to use the
static strategy in specific cases. This might be to keep runtime performance
acceptable even when running a development or debug build.

### Upgrade path from templates

The entire idea of statically typed languages is that coding against specific
types and interfaces is a better model and experience. Unfortunately, templates
don't provide many of those benefits to programmers until it's too late (users
are consuming the API) and with high overhead (template error messages).
Generally, code should move towards more rigorously type checked constructs.
However, existing C++ code is full of unrestricted usage of compile-time
duck-typed templates. They are incredibly convenient to write and so likely will
continue to exist for a long time.

We want there to be a natural, incremental upgrade path from templated code to
generic code. This gives us these additional principles:

-   Users should be able to convert a single template parameter to be generic at
    a time.
-   Converting from a template parameter to a generic parameter should be safe.
    It should either work or fail to compile, never silently change semantics.
-   We should minimize the effort to convert functions and types from templated
    to generic. Ideally it should just require specifying the type constraints,
    affecting just the signature of the function, not its body.
-   **Nice to have:** It should be legal to call templated code from generic
    code when it would have the same semantics as if called from non-generic
    code, and an error otherwise. This is to allow more templated functions to
    be converted to generics, instead of requiring them to be converted
    specifically in bottom-up order.
-   **Nice to have:** When defining a new generic interface to replace a
    template, support providing using the old templated implementation
    temporarily as a default until types transition.

### Coherence

Also, we want the generics system to have the _coherence_ property. This means
that there is a single answer to the question:

> What is the implementation of this interface for this type, if any?

independent of context, such as the libraries imported into a given file. Since
a generic function only depends on interface implementations, they will always
behave consistently on a given type, independent of context.

There are some capabilities we would like for interfaces which are in tension
with the coherence property:

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

### No novel name lookup

We want to avoid adding rules for name lookup that are specific to generics.
This is in contrast to Rust which has different lookup rules inside its traits.
Instead, we should structure generics in a way that reuses existing name lookup
facilities of the language.

**Nice to have:** For example, if `x` has type `T`, then if you write `x.y` you
should be able to look up `y` in the definition of `T`.

### Learn from others

Many languages have implemented generics systems, and we should learn from those
experiences. We should copy what works and makes sense in the context of Carbon,
and change decisions that led to undesirable compromises. We are taking the
strongest guidance from Rust and Swift, which have the most similar goals.

### Interfaces are nominal

Interfaces can either be structural, as in Go, or nominal, as in Rust and Swift.
Structural interfaces match any type that has the required methods, whereas
nominal interfaces only match if there is an explicit declaration of that fact
for that specific type. Carbon will support nominal interfaces, consistent with
the philosophy of being explicit.

This means that interfaces implicitly specify the intended semantics and
invariants of and between those functions. Unlike the function signatures, this
contract is between the implementers and the consumers of interfaces and is not
enforced by Carbon itself. For example, a _Draw_ method would mean different
things when it is part of a _GameResult_ interface versus a _Image2D_ interface,
even if those methods happen to have the same signature.

### Interop and evolution

[Evolution is a high priority for Carbon](../../project/goals.md#software-and-language-evolution),
and so will need mechanisms to support evolution when using generics, such as
allowing new additions to an interface to have default implementations and/or be
marked "upcoming" to allow for a period of transition. Evolution in particular
means that the set of names in an interface can change, and so two interfaces
that don't start with name conflicts can develop them.

To handle name conflicts, interfaces should be separate, isolated namespaces. We
should provide mechanisms to allow one type to implement two interfaces that
accidentally use the same name for different things, and for functions to use
interfaces with name conflicts together on a single type. Contrast this with
Swift's protocols, where interfaces with associated types that have the same
name are aliased.

## Non-goals, caveats, limitations, and out-of-scope issues

What are we **not** doing with generics, particularly things that some other
languages do?

-   Don't need to provide full flexibility of templates from generics.
    -   [Templates](#relationship-to-templates) can still cover those
        exceptional cases that don't fit inside generics.
    -   If you want compile-time duck typing, that is available by way of
        templates.
    -   Notably, there is no need to allow a specialization of some generic
        interface for some particular type to actually expose a _different_
        interface (different set of methods or types within the interface for
        example).
-   Some features are presented as being part of generics because that is where
    we expect them to be most useful or most used. These features may be allowed
    and have application outside of generics. For example, we expect to allow
    type constraints on template parameters in addition to generic parameters.
-   Cannot add features to generics that inherently require monomorphization (or
    creating differently specialized code) or templating, that would prevent the
    dynamic compilation strategy.
-   Cannot defer type checking of generic definitions (an implementation
    strategy used by some C++ compilers where the tokens are stashed and
    replayed on use).
-   We won't consider runtime specialization as an implementation strategy. That
    is, some language runtimes JIT a specialization when it is first needed, but
    it is not a goal for Carbon to support such an implementation strategy.
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
