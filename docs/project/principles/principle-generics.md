<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM Exceptions.
See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon principle: Generics

## Principle

**Goal:** Our goal for generics support in Carbon is to get most of the
expressive benefits of C++ templates -- an abstraction mechanism that optimizes
for performance,
[the top priority for Carbon](https://docs.google.com/document/d/1MJvVIDXQrhIj6hZ7NwMDbDch9XLO2VaYrGq29E57meU/edit#heading=h.hntrglpoczk)
-- with fewer downsides. In particular:

- Reduce build times, particularly when developing.
- Clearer and earlier error reporting.
- Encapsulation of logic (with a template, the implementation is part of the
  interface).
- Enable simple user control of whether to use dynamic dispatch or
  specialization.
- Provide generally better code generation (both code size and speed) in all
  cases.

**Semantics:** A generic function will take some "generic arguments", which will
frequently be types, and in some cases will be implicit / inferred from the
types of the values of explicit arguments to the function. If a generic argument
is a type, the generic function's signature can specify constraints that the
caller's type must satisfy (this should also be legal for template arguments,
but less needed and less common). For example, a resizable array type (like
C++'s `std::vector`) might have a generic type parameter with the constraint
that the type must be movable. A sort function might apply to any array whose
elements are comparable and movable.

The key difference between templates and generics is that a generic function is
type checked when it is defined, and type checking can't use any information
that is only known when the function is instantiated (such as the exact type).

**Implementation strategy:** There are two strategies for generating code for
generic functions:

- Static specialization strategy: Like template arguments, the values for
  generic arguments must be statically known at the callsite, or known to be a
  generic argument to the calling function. This can generate separate,
  specialized versions of each combination of generic & template arguments, in
  order to optimize for those types / values.
- Dynamic strategy: Unlike template arguments, we require that it be possible to
  generate a single version of the function that uses runtime dispatch to get
  something semantically equivalent to separate instantiation, but likely with
  different size, build time, and performance characteristics.

The strategy used by the compiler should not be semantically visible to the
user. Users should be able to write a generic version of a function once, and
then be able to control the mix of strategies used to generate the code for that
function. For example, this can be to trade off binary size vs speed (maybe some
specific specializations are needed for performance, but others would just be
code bloat), or support dynamic dispatch when types are not known at compile
time.

**Desiderata:** We want there to be a natural upgrade path from templated code
to generic code, which gives us these additional principles:

- Converting from a template to a generic argument should be safe -- it should
  either fail to compile or work, never silently change semantics.
- We should minimize the effort to convert from template code to generic code.
  Ideally it should just require specifying the type constraints, affecting just
  the signature of the function, not its body.
- It should be legal to call templated code from generic code when it would have
  the same semantics as if called from non-generic code, and an error otherwise.
  This is to allow more templated functions to be converted to generics, instead
  of requiring them to be converted specifically in bottom-up order.

### Caveats

- Don't need to provide full flexibility of templates from generics.
  - We still intend to provide templates for those exceptional cases that don't
    fit inside generics.
  - If you want [duck](https://en.wikipedia.org/wiki/Duck_typing) /
    [structural](https://en.wikipedia.org/wiki/Structural_type_system) typing,
    that is available via templates.
  - Notably, there is no need to allow a specialization of some generic
    interface for some particular type to actually expose a _different_
    interface (different set of methods or types within the interface for
    example).
- No novel (generic specific) rules for name lookup.
  - An example of these would be the name lookup rules inside of Rust's traits.
  - Instead, structure generics in a way that reuses existing name lookup
    facilities of the language.
- Features of generics that e.g. allow you to put constraints on the types
  passed in are presented as part of generics since that is where they are most
  useful, but are still intended to be available for template functions.

## Applications of these principles

We write a type constraint in Carbon by saying we restrict to types that
implement specific _Interfaces_. Interfaces serve several purposes:

- They specify a set of functions (names and signatures) that must be available
  for any type being passed to a generic function, and therefore the only
  functions that may be called in the body of that function.
- **MAYBE:** They will allow other type constraints such as on size, prefix of
  the data layout, specified method implementations, tests that must pass, etc.
- They allow a set of constraints to be given a name so they can be reused.
- **MAYBE:** They implicitly specify the intended semantics and invariants of
  and between those functions. Unlike the function signatures, this contract is
  between the implementers and the consumers of interfaces and is not enforced
  by Carbon itself. For example, a _Draw_ method would mean different things
  when it is part of a _GameResult_ interface vs. a _2DImage_ interface, even if
  those methods happen to have the same signature.
- **MAYBE:** They allow multiple implementations of an interface for a given
  type. For example, a _Song_ might support multiple orderings (by title, by
  artist, etc.), which would be represented by having multiple implementations
  of a _Comparable_ interface.
- **MAYBE:** They allow you to define a set of constraints by composing multiple
  interfaces.
- **MAYBE:** Have mechanisms to support evolution, such as allowing new
  additions to an interface to have default implementations or be marked
  "upcoming" to allow for a period of transition.
- In order to allow libraries to be composed, there must be some way of saying a
  type implements an interface that is in another package that the authors of
  the type were unaware of. This is especially important since the library a
  type is defined in may not be able to see the interface definition without
  creating a dependency cycle or layering violation.

What are we NOT doing with generics, particularly things that some other
language does?

- Cannot add features to generics that inherently require monomorphization (or
  creating differently specialized code)
  - MAYBE can add such features to a restricted subset, but would create a
    significantly restricted subset.
- Cannot defer type checking of generic definitions (an implementation strategy
  used by some C++ compilers where the tokens are stashed and replayed on use).
- ...

## Proposals relevant to these principles

From newest to oldest (and most likely to be out of date):

- [Carbon meeting Nov 27, 2019 on Generics & Interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces -->
- [Carbon generic -> template function calls (TODO)](#broken-links-footnote)<!-- T:Carbon generic -> template function calls -->
- [Carbon closed function overloading proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal -->
- [Carbon: facet types and interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon: facet types and interfaces --><!-- A:#heading=h.cg5jp928f02n -->
- [Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types -->
- [Carbon pervasive generics (TODO)](#broken-links-footnote)<!-- T:Carbon pervasive generics -->
- [Carbon templates and generics (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics -->

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
