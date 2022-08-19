# Generics: Goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Purpose of this document](#purpose-of-this-document)
-   [Background](#background)
    -   [Generic parameters](#generic-parameters)
    -   [Interfaces](#interfaces)
    -   [Relationship to templates](#relationship-to-templates)
-   [Goals](#goals)
    -   [Use cases](#use-cases)
        -   [Generic programming](#generic-programming)
        -   [Upgrade path from C++ abstract interfaces](#upgrade-path-from-c-abstract-interfaces)
        -   [Dependency injection](#dependency-injection)
        -   [Generics instead of open overloading and ADL](#generics-instead-of-open-overloading-and-adl)
    -   [Performance](#performance)
    -   [Better compiler experience](#better-compiler-experience)
    -   [Encapsulation](#encapsulation)
    -   [Predictability](#predictability)
    -   [Dispatch control](#dispatch-control)
    -   [Upgrade path from templates](#upgrade-path-from-templates)
    -   [Path from regular functions](#path-from-regular-functions)
    -   [Coherence](#coherence)
    -   [No novel name lookup](#no-novel-name-lookup)
    -   [Learn from others](#learn-from-others)
    -   [Interfaces are nominal](#interfaces-are-nominal)
    -   [Interop and evolution](#interop-and-evolution)
    -   [Bridge for C++ customization points](#bridge-for-c-customization-points)
-   [What we are not doing](#what-we-are-not-doing)
    -   [Not the full flexibility of templates](#not-the-full-flexibility-of-templates)
    -   [Template use cases that are out of scope](#template-use-cases-that-are-out-of-scope)
    -   [Generics will be checked when defined](#generics-will-be-checked-when-defined)
    -   [Specialization strategy](#specialization-strategy)
-   [References](#references)

<!-- tocstop -->

## Purpose of this document

This document attempts to clarify our goals for the design of the generics
feature for Carbon. While these are not strict requirements, they represent the
yardstick by which we evaluate design decisions. We do expect to achieve most of
these goals, though some of these goals are somewhat aspirational or
forward-looking.

## Background

Carbon will support
[generics](terminology.md#generic-versus-template-parameters) to support generic
programming by way of
[parameterization of language constructs](terminology.md#parameterized-language-constructs)
with [early type checking](terminology.md#early-versus-late-type-checking) and
[complete definition checking](terminology.md#complete-definition-checking).

This is in contrast with the
[compile-time duck typing](https://en.wikipedia.org/wiki/Duck_typing#Templates_or_generic_types)
approach of C++ templates, and _in addition_ to
[template support in Carbon](#relationship-to-templates), if we decide to
support templates in Carbon beyond interoperability with C++ templates.

### Generic parameters

Generic functions and generic types will all take some "generic parameters",
which will frequently be types, and in some cases will be
[deduced](terminology.md#deduced-parameter) from the types of the values of
explicit parameters.

If a generic parameter is a type, the generic function's signature can specify
constraints that the caller's type must satisfy. For example, a resizable array
type (like C++'s `std::vector`) might have a generic type parameter with the
constraint that the type must be movable and have a static size. A sort function
might apply to any array whose elements are comparable and movable.

A constraint might involve multiple generic parameters. For example, a merge
function might apply to two arbitrary containers so long as their elements have
the same type.

### Interfaces

We need some way to express the constraints on a generic type parameter. In
Carbon we express these "type constraints" by saying we restrict to types that
implement specific [_interfaces_](terminology.md#interface). Interfaces describe
an API a type could implement; for example, it might specify a set of functions,
including names and signatures. A type implementing an interface may be passed
as a generic type argument to a function that has that interface as a
requirement of its generic type parameter. Then, the functions defined in the
interface may be called in the body of the function. Further, interfaces have
names that allow them to be reused.

Similar compile-time and run-time constructs may be found in other programming
languages:

-   [Rust's traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
-   [Swift's protocols](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html)
-   [Java interfaces](<https://en.wikipedia.org/wiki/Interface_(Java)>)
-   [C++ concepts](<https://en.wikipedia.org/wiki/Concepts_(C%2B%2B)>)
    (compile-time only)
-   [Abstract base classes](<https://en.wikipedia.org/wiki/Class_(computer_programming)#Abstract_and_concrete>)
    in C++, etc. (run-time only)
-   [Go interfaces](https://gobyexample.com/interfaces) (run-time only)

In addition to specifying the methods available on a type, we may in the future
expand the role of interfaces to allow other type constraints, such as on size,
prefix of the data layout, specified method implementations, tests that must
pass, etc. This might be part of making interfaces as expressive as classes, as
part of a strategy to migrate to a future version of Carbon that uses interfaces
instead of, rather than in addition to, standard inheritance-and-classes
object-oriented language support. For the moment, everything beyond specifying
the _methods_ available is out of scope.

### Relationship to templates

The entire idea of statically typed languages is that coding against specific
types and interfaces is a better model and experience. Unfortunately, templates
don't provide many of those benefits to programmers until it's too late, when
users are consuming the API. Templates also come with high overhead, such as
[template error messages](#better-compiler-experience).

We want Carbon code to move towards more rigorously type checked constructs.
However, existing C++ code is full of unrestricted usage of compile-time
duck-typed templates. They are incredibly convenient to write and so likely will
continue to exist for a long time.

The question of whether Carbon has direct support for templates is out of scope
for this document. The generics design is not completely separate from
templates, so it is written as if Carbon will have its own templating system. It
is assumed to be similar to C++ templates with some specific changes:

-   It may have some limitations to be more compatible with generics, much like
    how we
    [restrict overloading](#generics-instead-of-open-overloading-and-adl).
-   We likely will have a different method of selecting between different
    template instantiations, since
    [SFINAE](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
    makes it difficult to deliver high quality compiler diagnostics.

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
Additionally, we want to support some dynamic dispatch use cases; for example,
in cases that inheritance struggles with.

### Use cases

To clarify the expressive range we are aiming for, here are some specific use
cases we expect Carbon generics to cover.

#### Generic programming

We in particular want to support
[generic programming](https://en.wikipedia.org/wiki/Generic_programming),
including:

-   Containers: arrays, maps, lists, and more complicated data structures like
    trees and graphs
-   Algorithms: sort, search
-   Wrappers: optional, variant, expected/result, smart pointers
-   Parameterized numeric types: `std::complex<T>`
-   Configurable and parametric APIs: the storage-customized `std::chrono` APIs
-   [Policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design)

These would generally involve static, compile-time type arguments, and so would
generally be used with [static dispatch](#dispatch-control).

#### Upgrade path from C++ abstract interfaces

Interfaces in C++ are often represented by abstract base classes. Generics
should offer an alternative that does not rely on inheritance. This means looser
coupling and none of the problems of multiple inheritance. Some people, such as
[Sean Parent](https://sean-parent.stlab.cc/papers-and-presentations/#better-code-runtime-polymorphism),
advocate for runtime polymorphism patterns in C++ that avoid inheritance because
it can cause runtime performance, correctness, and code maintenance problems in
some situations. Those patterns require a lot of boilerplate and complexity in
C++. It would be nice if those patterns were simpler to express with Carbon
generics. More generally, Carbon generics will provide an alternative for those
situations inheritance doesn't handle as well. As a specific example, we would
like Carbon generics to supplant the need to support multiple inheritance in
Carbon.

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

#### Generics instead of open overloading and ADL

One name lookup problem we would like to avoid is caused by open overloading.
Overloading is where you provide multiple implementations of a function with the
same name, and the implementation used in a specific context is determined by
the argument types. Open overloading is overloading where the overload set is
not restricted to a single file or library. This works with
[Argument-dependent lookup](https://en.wikipedia.org/wiki/Argument-dependent_name_lookup),
or [ADL](https://en.cppreference.com/w/cpp/language/adl), a mechanism for
enabling open overloading without having to reopen the namespace where the
function was originally defined. Together these enable
[C++ customization points](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html).

This is commonly used to provide a type-specific implementation of some
operation, but doesn't provide any enforcement of consistency across the
different overloads. It makes the meaning of code dependent on which overloads
are imported, and is at odds with being able to type check a function
generically.

Our goal is to address this use case, known more generally as
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
with a generics mechanism that does enforce consistency so that type checking is
possible without seeing all implementations. This will be Carbon's replacement
for open overloading. As a consequence, Carbon generics will need to be able to
support operator overloading.

A specific example is the absolute value function `Abs`. We would like to write
`Abs(x)` for a variety of types. For some types `T`, such as `Int32` or
`Float64`, the return type will be the same `T`. For other types, such as
`Complex64` or `Quaternion`, the return type will be different. The generic
functions that call `Abs` will need a way to specify whether they only operate
on `T` such that `Abs` has signature `T -> T`.

This does create an issue when interoperating with C++ code using open
overloading, which will
[need to be addressed](#bridge-for-c-customization-points).

### Performance

For any real-world C++ template, there shall be an idiomatic reformulation in
Carbon generics that has equal or better performance.
[Performance is the top priority for Carbon](/docs/project/goals.md#performance-critical-software),
and we expect to use generics pervasively, and so they can't compromise that
goal in release builds.

**Nice to have:** There are cases where we should aim to do better than C++
templates. For example, the additional structure of generics should make it
easier to reduce generated code duplication, reducing code size and cache
misses.

### Better compiler experience

Compared to C++ templates, we expect to reduce build times, particularly in
development builds. We also expect the compiler to be able to report clearer
errors, and report them earlier in the build process.

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
arguments or identical implementations and only generating code for them once.

### Encapsulation

With a template, the implementation is part of the interface and types are only
checked when the function is called and the template is instantiated.

A generic function is type checked when it is defined, and type checking can't
use any information that is only known when the function is instantiated such as
the exact argument types. Furthermore, calls to a generic function may be type
checked using only its declaration, not its body. You should be able to call a
generic function using only a forward declaration.

### Predictability

A general property of generics is they are more predictable than templates. They
make clear when a type satisfies the requirements of a function; they have a
documented contract. Further, that contract is enforced by the compiler, not
sensitive to implementation details in the function body. This eases evolution
by reducing (but not eliminating) the impact of
[Hyrum's law](https://www.hyrumslaw.com/).

**Nice to have:** We also want well-defined boundaries between what is legal and
not. This is "will my code be accepted by the compiler" predictability. We would
prefer to avoid algorithms in the compiler with the form "run for up to N steps
and report an error if it isn't resolved by then." For example, C++ compilers
will typically have a template recursion limit. With generics, these problems
arise due to trying to reason whether something is legal in all possible
instantiations, rather than with specific, concrete types.

Some of this is likely unavoidable or too costly to avoid, as most existing
generics systems
[have undecidable aspects to their type system](https://3fx.ch/typing-is-hard.html),
including [Rust](https://sdleffler.github.io/RustTypeSystemTuringComplete/) and
[Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024). We
fully expect there to be metaprogramming facilities in Carbon that will be able
to execute arbitrary Turing machines, with infinite loops and undecidable
stopping criteria. We don't see this as a problem though, just like we don't
worry about trying to make the compiler reliably prevent you from writing
programs that don't terminate.

We _would_ like to distinguish "the executed steps are present in the program's
source" from "the compiler has to search for a proof that the code is legal." In
the former case, the compiler can surface a problem to the user by pointing to
lines of code in a trace of execution. The user could employ traditional
debugging techniques to refine their understanding until they can determine a
fix. What we want to avoid is the latter case, since it has bad properties:

-   Error messages end up in the form: "this was too complicated to figure out,
    I eventually gave up."
-   Little in the way of actionable feedback on how to fix problems.
-   Not much the user can do to debug problems.
-   If the compiler is currently right at a limit for figuring something out, it
    is easy to imagine a change to a distant dependency can cause it to suddenly
    stop compiling.

If we can't find acceptable restrictions to make problems efficiently decidable,
the next best solution is to require the proof to be in the source instead of
derived by the compiler. If authoring the proof is too painful for the user, the
we should invest in putting the proof search into IDEs or other tooling.

### Dispatch control

Enable simple user control of whether to use dynamic or static dispatch.

**Implementation strategy:** There are two strategies for generating code for
generic functions:

-   Static specialization strategy: Like template parameters, the values for
    generic parameters must be statically known at the callsite, or known to be
    a generic parameter to the calling function. This can generate separate,
    specialized versions of each combination of generic and template arguments,
    in order to optimize for those types or values.
-   Dynamic strategy: This is when the compiler generates a single version of
    the function that uses runtime dispatch to get something semantically
    equivalent to separate instantiation, but likely with different size, build
    time, and performance characteristics.

By default, we expect the implementation strategy to be controlled by the
compiler, and not semantically visible to the user. For example, the compiler
might use the static strategy for release builds and the dynamic strategy for
development. Or it might choose between them on a more granular level based on
code analysis, specific features used in the code, or profiling -- maybe some
specific specializations are needed for performance, but others would just be
code bloat.

We require that all generic functions can be compiled using the static
specialization strategy. For example, the values for generic parameters must be
statically known at the callsite. Other limitations are
[listed below](#specialization-strategy).

**Nice to have:** It is desirable that the majority of functions with generic
parameters also support the dynamic strategy. Specific features may prevent the
compiler from using the dynamic strategy, but they should ideally be relatively
rare, and easy to identify. Language features should avoid making it observable
whether function code generated once or many times. For example, you should not
be able to take the address of a function with generic parameters, or determine
if a function was instantiated more than once using function-local static
variables.

There are a few obstacles to supporting dynamic dispatch efficiently, which may
limit the extent it is used automatically by implementations. For example, the
following features would benefit substantially from guaranteed monomorphization:

-   Field packing in class layout. For example, packing a `bool` into the lower
    bits of a pointer, or packing bit-fields with generic widths.
-   Allocating local variables in stack storage. Without monomorphization, we
    would need to perform dynamic memory allocation -- whether on the stack or
    the heap -- for local variables whose sizes depend on generic parameters.
-   Passing parameters to functions. We cannot pass values of generic types in
    registers.

While it is possible to address these with dynamic dispatch, handling some of
them might have far-reaching and surprising performance implications. We don't
want to compromise our goal for predictable performance.

We will allow the user to explicitly opt-in to using the dynamic strategy in
specific cases. This could be just to control binary size in cases the user
knows are not performance sensitive, or it could be to get the additional
capability of operating on values with dynamic types. We may need to restrict
this in various ways to maintain efficiency, like Rust does with object-safe
traits.

We also anticipate that the user may want to force the compiler to use the
static strategy in specific cases. This might be to keep runtime performance
acceptable even when running a development or debug build.

### Upgrade path from templates

We want there to be a natural, incremental upgrade path from templated code to
generic code.
[Assuming Carbon will support templates directly](#relationship-to-templates),
the first step of migrating C++ template code would be to first convert it to a
Carbon template. The problem is then how to convert templates to generics within
Carbon. This gives us these sub-goals:

-   Users should be able to convert a single template parameter to be generic at
    a time. A hybrid function with both template and generic parameters has all
    the limitations of a template function: it can't be completely definition
    checked, it can't use the dynamic strategy, etc. Even so, there are still
    benefits from enforcing the function's declared contract for those
    parameters that have been converted.
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
-   **Nice to have:** Provide a way to migrate from a template to a generic
    without immediately updating all of the types used with the template. For
    example, if the generic code requires types to implement a new interface,
    one possible solution would use the original template code to provide an
    implementation for that interface for any type that structurally has the
    methods used by the original template.

If Carbon does not end up having direct support for templates, the transition
will necessarily be less incremental.

### Path from regular functions

Replacing a regular, non-parameterized function with a generic function should
not affect existing callers of the function. There may be some differences, such
as when taking the address of the function, but ordinary calls should not see
any difference. In particular, the return type of a generic function should
match, without any type erasure or additional named members.

### Coherence

We want the generics system to have the
[_coherence_ property](terminology.md#coherence), so that the implementation of
an interface for a type is well defined. Since a generic function only depends
on interface implementations, they will always behave consistently on a given
type, independent of context. For more on this, see
[this description of what coherence is and why Rust enforces it](https://github.com/Ixrec/rust-orphan-rules#what-is-coherence).

Coherence greatly simplifies the language design, since it reduces the need for
complicated rules to picking an implementation when there are many candidates.
It also has a number of benefits for users:

-   It removes a way packages can conflict with each other.
-   It makes the behavior of code more consistent and predictable.
-   It means there is no need to provide a disambiguation mechanism.
    Disambiguation is particularly problematic since the ambiguous call is often
    in generic code rather than code you control.
-   A consistent definition of a type is useful for instantiating a C++ or
    Carbon template on that type.

The main downside of coherence is that there are some capabilities we would like
for interfaces that are in tension with having an orphan rule limiting where
implementations may be defined. For example, we would like to address
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions#another-clojure-solution-using-protocols).
We can get some of the way there by allowing the implementation of an interface
for a type to be defined with either the interface or the type. But some use
cases remain:

-   They should be some way of selecting between multiple implementations of an
    interface for a given type. For example, a _Song_ might support multiple
    orderings, such as by title or by artist. These would be represented by
    having multiple implementations of a _Comparable_ interface.
-   In order to allow libraries to be composed, there must be some way of saying
    a type implements an interface that is in another package that the authors
    of the type were unaware of. This is especially important since the library
    a type is defined in may not be able to see the interface definition without
    creating a dependency cycle or layering violation.

We should have some mechanism for addressing these use cases. There are multiple
approaches that could work:

-   Interface implementations could be external to types and are passed in to
    generic functions separately.
-   There could be some way to create multiple types that are compatible with a
    given value that you can switch between using casts to select different
    interface implementations. This is the approach used by Rust
    ([1](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-the-newtype-pattern-to-implement-external-traits-on-external-types),
    [2](https://github.com/Ixrec/rust-orphan-rules#user-content-why-are-the-orphan-rules-controversial)).

Alternatives to coherence are discussed in [an appendix](appendix-coherence.md).

### No novel name lookup

We want to avoid adding rules for name lookup that are specific to generics.
This is in contrast to Rust which has different lookup rules inside its traits.
Instead, we should structure generics in a way that reuses existing name lookup
facilities of the language.

**Nice to have:** One application of this that would be nice to have is if the
names of a type's members were all determined by a type's definition. So if `x`
has type `T`, then if you write `x.y` you should be able to look up `y` in the
definition of `T`. This might need to be somewhat indirect in some cases. For
example, if `T` inherits from `U`, the name `y` might come from `U` and not be
mentioned in the definition of `T` directly. We may have similar mechanisms
where `T` gets methods that have default implementations in interfaces it
implements, as long as the names of those interfaces are explicitly mentioned in
the definition of `T`.

### Learn from others

Many languages have implemented generics systems, and we should learn from those
experiences. We should copy what works and makes sense in the context of Carbon,
and change decisions that led to undesirable compromises. We are taking the
strongest guidance from Rust and Swift, which have similar goals and significant
experience with the implementation and usability of generics. They both use
nominal interfaces, were designed with generics from the start, and produce
native code. Contrast with Go which uses structural interfaces, or Java which
targets a virtual machine that predated its generics feature.

For example, Rust has found that supporting defaults for interface methods is a
valuable feature. It is useful for [evolution](#interop-and-evolution),
implementation reuse, and for bridging the gap between the minimal functionality
a type wants to implement and the rich API that users want to consume
([example](https://doc.rust-lang.org/std/iter/trait.Iterator.html)).

We still have the flexibility to make simplifications that Rust cannot because
they need to maintain compatibility. We could remove the concept of
`fundamental` and explicit control over which methods may be specialized. These
are complicated and
[impose coherence restrictions](http://aturon.github.io/tech/2017/02/06/specialization-and-coherence/).

### Interfaces are nominal

Interfaces can either be [structural](terminology.md#structural-interfaces), as
in Go, or [nominal](terminology.md#nominal-interfaces), as in Rust and Swift.
Structural interfaces match any type that has the required methods, whereas
nominal interfaces only match if there is an explicit declaration stating that
the interface is implemented for that specific type. Carbon will support nominal
interfaces, allowing them to designate _semantics_ beyond the basic structure of
the methods.

This means that interfaces implicitly specify the intended semantics and
invariants of and between those functions. Unlike the function signatures, this
contract is between the implementers and the consumers of interfaces and is not
enforced by Carbon itself. For example, a `Draw` method would mean different
things when it is part of a `GameResult` interface versus an `Image2D`
interface, even if those methods happen to have the same signature.

### Interop and evolution

[Evolution is a high priority for Carbon](/docs/project/goals.md#software-and-language-evolution),
and so will need mechanisms to support evolution when using generics. New
additions to an interface might:

-   need default implementations
-   be marked "upcoming" to allow for a period of transition
-   replace other APIs that need to be marked "deprecated"

Experience with C++ concepts has shown that interfaces are
[hard to evolve](https://www.youtube.com/watch?v=v_yzLe-wnfk) without these
kinds of supporting language mechanisms. Otherwise changes to interfaces need to
made simultaneously with updates to types that implement the interface or
functions that consume it.

Another way of supporting evolution is to allow one interface to be
substitutable for another. For example, a feature that lets you use an
implementation of `Interface1` for a type to automatically get an implementation
of `Interface2`, as well as the other way around, would help transitioning
between those two interfaces.

Evolution in particular means that the set of names in an interface can change,
and so two interfaces that don't start with name conflicts can develop them.

To handle name conflicts, interfaces should be separate, isolated namespaces. We
should provide mechanisms to allow one type to implement two interfaces that
accidentally use the same name for different things, and for functions to use
interfaces with name conflicts together on a single type. Contrast this with
Swift, where a type can only supply one associated type of a given name even
when implementing multiple protocols. Similarly a function in Swift with a given
name and signature can only have a single implementation for a type.

Note this is possible since [interfaces are nominal](#interfaces-are-nominal).
The place where types specify that they implement an interface is also the
vehicle for unambiguously designating which function implementation goes with
what interface.

### Bridge for C++ customization points

There will need to be some bridge for C++ extension points that currently rely
on open overloading or
[ADL](https://en.wikipedia.org/wiki/Argument-dependent_name_lookup). For
example, we need some way for C++
[customization points](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html)
like `swap` to work on Carbon types. We might define `CPlusPlus.ADL.swap` as a
Carbon interface to be that bridge. Carbon types could implement that interface
to work from C++, and Carbon functions could use that interface to invoke `swap`
on C++ types.

Similarly, we will want some way to implement Carbon interfaces for C++ types.
For example, we might have a template implementation of an `Addable` interface
for any C++ type that implements `operator+`.

## What we are not doing

What are we **not** doing with generics, particularly things that some other
languages do?

### Not the full flexibility of templates

Generics don't need to provide full flexibility of C++ templates:

-   The current assumption is that
    [Carbon templates](#relationship-to-templates) will cover those cases that
    don't fit inside generics, such as code that relies on compile-time duck
    typing.
-   We won't allow a specialization of some generic interface for some
    particular type to actually expose a _different_ interface, with different
    methods or different types in method signatures. This would break modular
    type checking.
-   [Template metaprogramming](https://en.wikipedia.org/wiki/Template_metaprogramming)
    will not be supported by Carbon generics. We expect to address those use
    cases with metaprogramming or templates in Carbon.

### Template use cases that are out of scope

We will also not require Carbon generics to support
[expression templates](https://en.wikipedia.org/wiki/Expression_templates),
[variadics](https://en.wikipedia.org/wiki/Variadic_function), or
[variadic templates](https://en.wikipedia.org/wiki/Variadic_template). Those are
all out of scope. It would be fine for our generics system to support these
features, but they won't drive any accommodation in the generics design, at
least until we have some resolution about templates in Carbon.

### Generics will be checked when defined

C++ compilers must defer full type checking of templates until they are
instantiated by the user. Carbon will not defer type checking of generic
definitions.

### Specialization strategy

We want all generic Carbon code to support [static dispatch](#dispatch-control).
This means we won't support unbounded type families. Unbounded type families are
when recursion creates an infinite collection of types, such as in
[this example from Swift](https://forums.swift.org/t/ergonomics-generic-types-conforming-in-more-than-one-way/34589/71)
or:

```carbon
fn Sort[T:! Comparable](list: List(T)) -> List(T) {
  if (list.size() == 1) return list;
  var chunks: List(List(T)) = FormChunks(list, sqrt(list.size()));
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

We won't require all generic Carbon code to support dynamic dispatch, but we
would like it to be an implementation option for the compiler in the majority of
cases.

Lastly, runtime specialization is out of scope as an implementation strategy.
That is, some language runtimes JIT a specialization when it is first needed,
but it is not a goal for Carbon to support such an implementation strategy.

## References

-   [#24: Generics goals](https://github.com/carbon-language/carbon-lang/pull/24)
-   [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
