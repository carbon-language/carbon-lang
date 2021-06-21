# Carbon Generics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [What are generics?](#what-are-generics)
-   [Goals](#goals)
-   [Terminology](#terminology)
-   [Non-type generics](#non-type-generics)
    -   [Basic generics](#basic-generics)
    -   [No address of a function with generic parameters](#no-address-of-a-function-with-generic-parameters)
    -   [Generic syntax is a placeholder](#generic-syntax-is-a-placeholder)
    -   [Implicit parameters](#implicit-parameters)
    -   [Mixing](#mixing)
    -   [FIXME: Local constants](#fixme-local-constants)
-   [Proposed programming model](#proposed-programming-model)
    -   [Syntax examples of common use cases](#syntax-examples-of-common-use-cases)

<!-- tocstop -->

## What are generics?

Generics are a mechanism for writing more abstract code that applies more
generally instead of making near duplicates for very similar situations, much
like templates. For example, instead of having one function per
type-you-can-sort:

```
fn SortInt32Vector(Ptr(Vector(Int32)): a) { ... }
fn SortStringVector(Ptr(Vector(String)): a) { ... }
...
```

you might have one function that could sort any array with comparable elements:

```
fn SortVector[Comparable:$ T](Ptr(Vector(T)): a) { ... }
```

Where the `SortVector` function applied to a `Ptr(Vector(Int32))` input is
semantically identical to `SortInt32Vector`, and similarly for
`Ptr(Vector(String))` input and `SortStringVector`.

Here `Comparable` is the name of an _interface_, which describes the
requirements for the type `T`. These requirements form the contract that allows
us to have an API boundary encapsulating the implementation of the function,
unlike templates. That is, given that we know `T` satisfies the requirements, we
can typecheck the body of the `SortVector` function; similarly, we can typecheck
that a call to `SortVector` is valid by checking that the type of the member
elements of the passed-in array satisfy the same requirements, without having to
look at the body of the `SortVector` function. These are in fact the main
differences between generics and templates:

-   We can completely typecheck a generic definition without information from
    the callsite.
-   We can completely typecheck a call to a generic with just information from
    the function's signature, not its body.

Contrast with a template function, where you may be able to do some checking
given a function definition, but more checking of the definition is required
after seeing the call sites (and you know which
[instantiations](terminology.md#instantiation) are needed).

The [generics terminology document](terminology.md) goes into more detail about
the
[difference between generics and templates](terminology.md#generic-versus-template-parameters).

## Goals

In general we aim to make Carbon Generics into an alternative to templates for
writing generic code, with improved software engineering properties at the
expense of some restrictions. See [the detailed discussion of goals](goals.md).

## Terminology

Terminology is described in the [generics terminology document](terminology.md)

## Non-type generics

Imagine we had a regular function that printed some number of 'X' characters:

```
fn PrintXs_Regular(Int: n) {
  var Int: i = 0;
  while (i < n) {
    Print("X");
    i += 1;
  }
}

PrintXs_Regular(1); // Prints: X
PrintXs_Regular(2); // Prints: XX
var Int: n = 3;
PrintXs_Regular(n); // Prints: XXX
```

### Basic generics

What would it mean to change the parameter to be a generic parameter?

```
fn PrintXs_Generic(Int:$ N) {
  var Int: i = 0;
  while (i < N) {
    Print("X");
    i += 1;
  }
}

PrintXs_Generic(1);  // Prints: X
PrintXs_Generic(2);  // Prints: XX
var Int: m = 3;
PrintXs_Generic(m);  // Compile error: value for generic parameter `n`
                     // unknown at compile time.
```

For the definition of the function there is only one difference: we added a `$`
to indicate that the parameter named `n` is generic. The body of the function
type checks using the same logic as `PrintXs_Regular`. However, callers must be
able to know the value of the argument at compile time. This allows the compiler
to adopt a code generation strategy that creates a separate copy of the
`PrintXs_Generic` function for each combination of values of the generic (and
template) arguments, called [static specialization](goals.md#dispatch-control).
In this case, this means that the compiler can generate different binary code
for the calls passing `n==1` and `n==2`. Knowing the value of `n` at code
generation time allows the optimizer to unroll the loop, so that the call
`PrintXs_Generic(2)` could be transformed into:

```
Print("X");
Print("X");
```

Since we know the generic parameter is restricted to values known at compile
time, we can use the generic parameter in places we would expect a compile-time
constant value, such as in types.

```
fn CreateArray(UInt:$ N, Int: value) -> FixedArray(Int, N) {
  var FixedArray(Int, N): ret;
  var Int: i = 0;
  while (i < N) {
    ret[i] = value;
    i += 1;
  }
  return ret;
}
```

**Comparison with other languages:** This feature is part of
[const generics in Rust](https://blog.rust-lang.org/2021/02/26/const-generics-mvp-beta.html).

### No address of a function with generic parameters

Since a function with a generic parameter can have many different addresses, we
have this rule:

**Rule:** It is illegal to take the address of any function with generic
parameters (similarly template parameters).

This rule also makes the difference between the compiler generating separate
static specializations or using a single generated function with runtime dynamic
dispatch harder to observe, enabling the compiler to switch between those
strategies without danger of accidentally changing the semantics of the program.

Generally speaking, we should hide the differences between the
[static and dynamic dispatch strategies](#dispatch-control). This includes how
many instances of a
[static local function variable](https://en.wikipedia.org/wiki/Local_variable#Static_local_variables)
are created, if we support them.

### Generic syntax is a placeholder

**NOTE:** The `$` syntax is a placeholder. In addition to `:$`, we are
considering: `:!`, `:@`, `:#`, and `::`. We might use the same character here as
we decide for FIXME: metaprogramming
[Carbon metaprogramming](https://github.com/josh11b/carbon-lang/blob/metaprogramming/docs/design/metaprogramming.md)
constructs.

### Implicit parameters

An [implicit parameter](terminology.md#implicit-parameter) is a value that is
determined by the type of the value passed to another parameter, and not passed
explicitly to the function. Implicit parameters are declared using square
brackets before the usual parameter list, as in:

```
fn PrintArraySize[Int: n](Ptr(FixedArray(String, n)): array) {
  Print(n);
}

var FixedArray(String, 3): a = ...;
PrintArraySize(&a);  // Prints: 3
```

**Open question:** Are regular dynamic parameters like `n` here allowed to be
used in type expressions like `FixedArray(String, n)`?

What happens here is the type for the `array` parameter is determined from the
value passed in, and the pattern-matching process used to see if the types match
finds that it does match if `n` is set to `3`.

Normally you would pass in an implicit parameter as a generic or template, not
as a regular parameter. This avoids overhead from having to support types (like
the type of `array` inside the `PrintArraySize` function body) that are only
fully known with dynamic information. For example:

```
fn PrintStringArray[Int:$ n](Ptr(FixedArray(String, n)): array) {
  for (var Int: i = 0; i < n; ++i) {
    Print(array->get(i));
  }
}
```

Implicit arguments are always determined from the explicit arguments. It is
illegal not to mention implicit parameters in the explicit parameter list and
there is no syntax for specifying implicit arguments directly at the call site.

```
// ERROR: can't determine `n` from explicit parameters
fn Illegal[Int:$ n](Int: i) -> Bool { return i < n; }
```

### Mixing

-   A function can have a mix of generic, template, and regular parameters.
-   Can pass a template or generic value to a generic or regular parameter.
-   FIXME: There are restrictions passing a generic value to a template
    parameter, discussed in a [dedicated document](generic-to-template.md).

### FIXME: Local constants

You may also have local generic constants as members of types. Just like generic
parameters, they have compile-time, not runtime, storage. You may also have
template constant members, with the difference that template constant members
can use the actual value of the member in type checking. In both cases, these
can be initialized with values computed from generic/template parameters, or
other things that are effectively constant and/or available at compile time.

We also support local generic constants in functions:

```
fn PrintOddNumbers(Int:$ N) {
  // last_odd is computed and stored at compile time.
  var Int:$ LastOdd = 2 * N - 1;
  var Int: i = 1;
  while (i <= LastOdd) {
    Print(i);
    i += 2;
  }
}
```

FIXME Local template constants may be used in type checking:

```
fn PrimesLessThan(Int:$$ N) {
  var Int:$$ MaxNumPrimes = N / 2;
  // Value of MaxNumPrimes is available at type checking time.
  var FixedArray(Int, MaxNumPrimes): primes;
  var Int: num_primes_found = 0;
  // ...
}
```

Interfaces may include requirements that a type's implementation of that
interface have local constants with a particular type and name. These are called
FIXME: [associated constants](details.md#associated-constants).

## Proposed programming model

FIXME: This is described in detail in a separate document,
["Carbon deep dive: combined interfaces"](details.md). In summary:

-   Interfaces have a name and describe methods, functions, and other items for
    types to implement.
-   Types may implement interfaces at most once.
-   Implementations may be part of the type's definition, in which case you can
    directly call the interface's methods on those types. Or they may be
    external, in which case the implementation is allowed to be defined in the
    library defining the interface. The goal is that methods for interfaces
    implemented with the type are generally available without qualification, and
    a function with a generic type parameter can have the same function body as
    an unparameterized one.
-   Interfaces are usable as type-types, meaning you can define a generic
    function that have a type parameter satisfying the interface by saying the
    type of the type parameter is the interface. Inside such a generic function,
    the API of the type is [erased](terminology.md#type-erasure), except for the
    names defined in the interface.
-   You may also declare type-types, called
    ["structural interfaces"](terminology.md#structural-interfaces), that
    require multiple interfaces to be implemented, and give you control over how
    name conflicts are handled.
-   Alternatively, you may use a qualified syntax to directly call a function
    from a specific interface.
-   The `+` operation on type-types allows you conveniently combine interfaces.
    It gives you all the names that don't conflict. Names with conflicts can be
    accessed using the qualified syntax.
-   Interfaces can require other interfaces be implemented, or
    [extend/refine](terminology.md#extendingrefining-an-interface) them.

Future work:

-   A "newtype" mechanism called "adapting types" is provided to create new
    types that compatible with existing types but with different interface
    implementations. This can be used to add or replace implementations, or
    define implementations for reuse.
-   Associated types and interface parameters are two features provided to allow
    function signatures to vary with the implementing type. The biggest
    difference between these is that associated types ("output types") may be
    deduced from a type, and types can implement the same interface multiple
    times with different interface parameters ("input types").
-   Constraints have not been finalized.
-   Implementations can be parameterized, to apply to multiple types. These
    implementations can be restricted to various conditions are true for the
    parameters. When there are two implementations that can apply, there is a
    specialization rule that picks the more specific one.
-   Support types that vary at runtime.
-   Defaults and other ways to reuse code across implementations.
-   Types can define overloads for operators by implementing standard
    interfaces.
-   Ability to mark items as `upcoming` or `deprecated` to support evolution.
-   Generic associated and higher-ranked/kinded types.

### Syntax examples of common use cases

Here are some examples of writing an interface definition:

```
interface Printable {
  // `Print` is an associated method.
  method (Self: this) Print();
  // Method syntax here is a placeholder, should match
  // whatever syntax is used to define methods in a struct.
}

interface Media {
  // `Play` is an associated method that can mutate `this`.
  method (Ptr(Self): this) Play();
}
```

The `interface` keyword is used to define a
[_nominal interface_](terminology.md#nominal-interfaces). That means that types
need to explicitly implement them, using an `impl` block:

```
struct Song {
  // ...

  // Implementing `Printable` for `Song` inside the definition of `Song`
  // means all names of `Printable`, such as `F`, are included as a part
  // of the `Song` API.
  impl Printable {
    // Could use `Self` in place of `Song` here.
    method (Song: this) Print() { ... }
  }
}
// Implement `Media` for `Song` without changing the API of `Song`
// using an `extend` declaration. This may be defined in either
// the library defining `Song` or `Media`.
extend Song {
  impl Media {
    // Could use either `Self` or `Song` here.
    method (Ptr(Self): this) Play() { ... }
  }
}

var Song: song;
// `song.Print()` is allowed, unlike `song.Play()`.
song.Print();
// To call `Play` on `song`, use the qualified syntax:
song.(Media.Play)();
// Can also call `Print` using the qualified syntax:
song.(Printable.Print)();
```

Here are some functions taking a value with type conforming to an interface:

```
// These definitions are completely equivalent.
fn PrintIt1(Ptr(Printable:$ T): y) {
  y->Print();
}
fn PrintIt2[Printable:$ T](Ptr(T): y) {
  y->Print();
}
PrintIt1(&song);
PrintIt2(&song);
```

The `+` operator is the common way of combining interfaces, used here to express
a function taking a value with type conforming to two different interfaces:

```
fn PrintAndPlay[Printable + Media:$ T](Ptr(T): p) {
  // `T` has all the names of `Printable` and `Media`
  // that don't conflict.
  p->Print();
  // Can call `Play` here, even though `song.Play()`
  // isn't allowed since `Media` is external.
  p->Play();
  // Qualified syntax works even if there is a name
  // conflict between Printable and Media.
  p->(Media.Play)();
}
PrintAndPlay(&song);
```

FIXME: Interface with associated types

```
interface Container {
  // Element can be any type.
  var Type:$ Element;
  // Iter is an associated type that
  // must conform to the `Iter` interface.
  var Iterable:$ Iter;

  method (Ptr(Self): this) Begin() -> Iter;
  method (Ptr(Self): this) Insert(Iter: pos, Element: value);
}

struct SomeStrings {
  // ...
  impl Container {
    // FIXME: Is this needed?
    // FIXME: Or maybe use `alias` here?
    var Type:$ Element = String;
    var Iterable:$ Iter = Int;
    method (Ptr(Self): this) Begin() -> Int { ... }
    method (Ptr(Self): this) Insert(Int: pos String: value) { ... }
  }
}
```

FIXME: Function taking a value with a list of implementations for a single
interface, all compatible with a single representation type

```
// Compares `*a` with `*b`, returning the result of the first
// comparison in `compare` that considers them unequal.
// This could be used to implement lexicographical comparison.
fn IsGreater[Type:$ T](Ptr(T): a, Ptr(T): b,
                       List(CompatibleWith(Comparable, T)):$ compare)
    -> Bool { ... }
```

FIXME: Function taking a list of values with different types that all implement
a single interface

```
// `NTuple(N, ToString)` is `(ToString, ..., ToString)`, the tuple
// with `N` copies of `ToString`. `Ts` is a tuple of `N` types
// `(Ts[0], ..., Ts[N-1])`, where each type `Ts[i]` has type `ToString`.
// `input` is a tuple of `N` values, where `input[i]` has type `Ts[i]`.
fn StringConcat[Int:$ N, NTuple(N, ToString):$ Ts](Ts...: input)
    -> String { ... }
```

The `impl` keyword is also used to express that an interface requires another
interface to be implemented:

```
interface Equatable {
  method (Self: this) IsEqual(Self: that) -> Bool;
}

// `Iterable` requires that `Equatable` is implemented.
interface Iterable {
  impl Equatable;
  method (Ptr(Self): this) Advance();
}

struct SomeStringsIterator {
  // ...
  impl Iterable {
    method (Ptr(Self): this) Advance() { ... }
  }
  impl Equatable {
    method (Self: this) IsEqual(Self: that) -> Bool { ... }
  }
  // If the definition of `Equatable` was deleted, you would get
  // Error: Missing implementation of interface `Equatable`
  //        required by `Iterable`

}
var SomeStringsIterator: i = ...;
i.Advance();
i.IsEqual(i);
```

The `extends` keyword is used to
[extend/refine](terminology.md#extendingrefining-an-interface) another
interface. This means the refined interface is both required and all its methods
are included in the refining interface.

```
// `Hashable` refines `Equatable`.
interface Hashable {
  extends Equatable;
  method (Self: this) Hash() -> UInt64;
}
// `Hashable` is equivalent to:
interface Hashable {
  impl Equatable;
  alias IsEqual = Equatable.IsEqual;
  method (Self: this) Hash() -> UInt64;
}

struct Key {
  // ...
  impl Hashable {
    method (Key: this) IsEqual(Key: that) -> Bool { ... }
    method (Key: this) Hash() -> UInt64 { ... }
  }
  // No need to separately implement `Equatable`.
}
var Key: k = ...;
k.Hash();
k.IsEqual(k);
```

TODO: Also include covariant refinement of individual associated types of the
refined interface.

A [`structural interface`](terminology.md#structural-interfaces) allows you to
express a combination of nominal interfaces without introducing a new nominal
interface. The structural interface is implemented for exactly those types that
implement the nominal interface requirements.

```
// `PrintableMedia` has all names from `Printable` and `Media`,
// which must not conflict. As long as there are no name
// conflicts, this definition is equivalent to
// `Printable + Media`.
structural interface PrintableMedia {
  extends Printable;
  extends Media;
}

// `PrintableMedia2` is exactly equivalent to `PrintableMedia`.
// `extends` means require the interface be implemented,
// just like `impl`, and `alias` all of the names.
structural interface PrintableMedia2 {
  impl Printable;
  alias Print = Printable.Print;
  impl Media;
  alias Play = Media.Play.
}

// `PrintAndPlay2` is equivalent to `PrintAndPlay` above.
fn PrintAndPlay2[PrintableMedia:$ T](Ptr(T): p) {
  p->Print();
  p->Play();
  // Qualified syntax also works.
  p->(Media.Play)();
}

// Song implements `PrintableMedia` without an explicit
// declaration. Anything that implements both `Printable`
// and `Media` implements `PrintableMedia`.
PrintAndPlay2(&song);
```

You may implement a structural interface as long as it has aliases for every
name we need to implement for its required interfaces. Implementing the
structural interface is equivalent to implementing all the interfaces it
requires.

```
// Can implement `PrintableMedia` since it has aliases for all
// the names in `Printable` and `Media`.
struct Playlist {
  // ...
  impl PrintableMedia {
    method (Self: this) Print() { ... }
    method (Ptr(Self): this) Play() { ... }
  }
}

// The above is equivalent to:
struct Playlist2 {
  // ...
  impl Printable {
    method (Self: this) Print() { ... }
  }
  impl Media {
    method (Ptr(Self): this) Play() { ... }
  }
}
```

Structural interfaces can be used to combine two interfaces even when they have
name conflicts.

```
interface Renderable {
  method (Self: this) Center() -> (Int, Int);
  method (Self: this) Draw();
}
interface EndOfGame {
  method (Self: this) Draw();
  method (Self: this) Winner(Int: player);
}

// `Combined1` has all names from `Renderable` and `EndOfGame`
// that do not conflict. Can use qualification, like
// `x.(Renderable.Draw)()`, to get any names from `Renderable`
// or `EndOfGame` even if there is a conflict.
structural interface Combined1 {
  extends Renderable + EndOfGame;
  alias Draw_Renderable = Renderable.Draw;
  alias Draw_EndOfGame = EndOfGame.Draw;
}
// `Combined2` uses `impl` and so only has names that are
// mentioned explicitly in its definition.
// Can use qualification (`x.(Renderable.Center)()`) to access
// any names from `Renderable` or `EndOfGame` even if they are
// not mentioned in `Combined2`.
structural interface Combined2 {
  impl Renderable;
  impl EndOfGame;
  alias Draw_Renderable = Renderable.Draw;
  alias Draw_EndOfGame = EndOfGame.Draw;
  alias Winner = EndOfGame.Winner;
}

// All of these functions accept the same values, namely anything
// with a type implementing both `Renderable` and `EndOfGame`.
fn F1[Combined1:$ T](T: x) { ... }
fn F2[Combined2:$ T](T: x) { ... }
fn FPlus[Renderable + EndOfGame:$ T](T: x) { ... }
```
