# Carbon Generics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [What are generics?](#what-are-generics)
-   [Goals: Generics](#goals-generics)
-   [Glossary / Terminology](#glossary--terminology)
-   [Non-type generics](#non-type-generics)
    -   [Basic generics](#basic-generics)
    -   [Basic templates](#basic-templates)
        -   [Difference between templates and generics](#difference-between-templates-and-generics)
        -   [Substitution failure is an error](#substitution-failure-is-an-error)
    -   [Implicit parameters](#implicit-parameters)
    -   [Mixing](#mixing)
    -   [Local constants](#local-constants)
    -   [Generic type parameters versus templated type parameters](#generic-type-parameters-versus-templated-type-parameters)
-   [Proposed programming model](#proposed-programming-model)
    -   [Syntax examples of common use cases](#syntax-examples-of-common-use-cases)
    -   [Calling templated code](#calling-templated-code)

<!-- tocstop -->

## What are generics?

Generics are a mechanism for writing more abstract code that applies more
generally instead of making near duplicates for very similar situations, much
like templates. For example, instead of having one function per
type-you-can-sort:

```
fn SortInt32Array(Ptr(Array(Int32)): a) { ... }
fn SortStringArray(Ptr(Array(String)): a) { ... }
...
```

you might have one function that could sort any array with comparable elements:

```
fn SortArray[Comparable:$ T](Ptr(Array(T)): a) { ... }
```

Where the `SortArray` function applied to an `Array(Int32)*` input is
semantically identical to `SortInt32Array`, and similarly for `Array(String)*`
input and `SortStringArray`.

Here `Comparable` is the name of an _interface_, which describes the
requirements for the type `T`. These requirements form the contract that allows
us to have an API boundary encapsulating the implementation of the function,
unlike templates. That is, given that we know `T` satisfies the requirements, we
can typecheck the body of the `SortArray` function; similarly, we can typecheck
that a call to `SortArray` is valid by checking that the type of the member
elements of the passed-in array satisfy the same requirements, without having to
look at the body of the `SortArray` function. These are in fact the main
differences between generics and templates:

-   We can completely typecheck a generic definition without information from
    the callsite.
-   We can completely typecheck a call to a generic with just information from
    the function's signature, not its body.

Contrast with a template function, where you may be able to do some checking
given a function definition, but more checking of the definition is required
after seeing the call sites (and you know which specializations are needed).

Read more here:
[Carbon Generics: Terminology: "Generic versus template parameters" section](terminology.md#generic-versus-template-parameters).

## Goals: Generics

In general we aim to make Carbon Generics into an alternative to templates for
writing generic code, with improved software engineering properties at the
expense of some restrictions. See [the detailed discussion of goals](goals.md).

In this proposal we try and define a generics system that has these properties
to allow migration from templates:

-   Templated code (perhaps migrated from C++) can be converted to generics
    incrementally, one function or parameter at a time. Typically this would
    involve determining the interfaces that generic types need to implement to
    call the function, proving the API the function expects.
-   It should be legal to call templated code from generic code when it would
    have the same semantics as if called from non-generic code, and an error
    otherwise. This is to allow more templated functions to be converted to
    generics, instead of requiring them to be converted specifically in
    bottom-up order.
-   Converting from a template to a generic parameter should be safe -- it
    should either fail to compile or work, never silently change semantics.
-   We should minimize the effort to convert from template code to generic code.
    Ideally it should just require specifying the type constraints, affecting
    just the signature of the function, not its body.

Also we would like to support:

-   Selecting between a dynamic and a static strategy for the generated code, to
    give the user control over performance, binary sizes, build speed, etc.
-   Use of generic functions with types defined in different libraries from
    those defining the interface requirements for those types.

MAYBE: migration to generics from inheritance, to disentangle subtyping from
implementation inheritance.

## Glossary / Terminology

See [Carbon Generics: Terminology](terminology.md)

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
fn PrintXs_Generic(Int:$ n) {
  var Int: i = 0;
  while (i < n) {
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
to adopt a code generation strategy that creates a _specialization_ of the
function `PrintXs_Generic` for each combination of values of the generic (and
template) arguments. In this case, this means that the compiler can generate
different binary code for the calls passing `n==1` and `n==2`. Knowing the value
of `n` at code generation time allows the optimizer to unroll the loop, so that
the call `PrintXs_Generic(2)` could be transformed into:

```
Print("X");
Print("X");
```

Since a function with a generic parameter can have many different addresses, we
have this rule:

**Rule:** It is illegal to take the address of any function with generic
parameters (similarly template parameters).

This rule also makes the difference between the compiler generating separate
specializations or using a single generated function with runtime dynamic
dispatch harder to observe, enabling the compiler to switch between those
strategies without danger of accidentally changing the semantics of the program.

**NOTE:** The `$` syntax is temporary and won't be the final syntax we use,
since it is not easy to type `$` from non-US keyboards. Instead of `:$`, we are
considering: `:!`, `:@`, `:#`, and `::`. We might use the same character here as
we decide for
[Carbon metaprogramming](https://github.com/josh11b/carbon-lang/blob/metaprogramming/docs/design/metaprogramming.md)
constructs.

**Comparison with other languages:** This feature is part of
[const generics in Rust](https://blog.rust-lang.org/2021/02/26/const-generics-mvp-beta.html).

### Basic templates

For this function, we could change the parameter to be a template parameter by
replacing "`Int:$ n`" with "`Int:$$ n`", but there would not be a difference
that you would observe. However, with template parameters we would have more
capabilities inside the function, so we could write this:

```
fn NumXs(1) -> Char {
  return 'X';
}
fn NumXs(2) -> String {
  return "XX";
}
fn PrintXs_Template(Int:$$ n) {
  Print(NumXs(n));
}

PrintXs_Template(1);  // Prints: X (using Print(Char))
PrintXs_Template(2);  // Prints: XX (using Print(String))
var Int: m = 3;
PrintXs_Template(m);  // Compile error: value for template parameter `n`
                      // unknown at compile time.
PrintXs_Template(3);  // Compile error: NumXs(3) undefined.
```

Since type checking is delayed until `n` is known, we don't need the return type
of `NumXs` to be consistent across different values of `n`.

**Comparison with other languages:** These are called
[non-type template parameters in C++](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter).

#### Difference between templates and generics

For generics, the body of the function is fully checked when it is defined; it
is an error to perform an operation the compiler can't verify. For templates,
name lookup and type checking may only be able to be resolved using information
from the call site.

#### Substitution failure is an error

Note: This is a difference from C++, and the rules may be different when calling
C++ code from Carbon.

In Carbon, when you call a function, the corresponding implementation (function
body) is resolved using name lookup and overload resolution rules, which use
information in the function signature but not the function body. The function
signature can include arbitrary code to determine if a function is applicable,
but once it is selected it won't ever switch to another function body. This
means that if substituting in templated arguments into a function triggers an
error, that error will be reported to the user instead of trying another
function body (say for a different overload of the same name that matches but
isn't preferred, perhaps because it is less specific).

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
-   There are restrictions passing a generic value to a template parameter,
    discussed in a (dedicated
    document)[https://github.com/josh11b/carbon-lang/blob/generic-to-template/docs/design/generics/generic-to-template.md].

### Local constants

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

Local template constants may be used in type checking:

```
fn PrimesLessThan(Int:$$ N) {
  var Int:$$ MaxNumPrimes = N / 2;
  // Value of MaxNumPrimes is available at type checking time.
  var FixedArray(Int, MaxNumPrimes): primes;
  var Int: num_primes_found = 0;
  // ...
}
```

Interfaces may include requirements that a type have local constants with a
particular type and name.

### Generic type parameters versus templated type parameters

Recall, from
[the "Difference between templates and generics" section above](#difference-between-templates-and-generics),
that we fully check functions with generic parameters at the time they are
defined, while functions with template parameters can use information from the
caller.

If you have a value of a generic type, you need to provide constraints on that
type that define what you can do with values of that type. However when using a
templated type, you can perform any operation on values of that type, and what
happens will be resolved once that type is known. This may be an error if that
type doesn't support that operation, but that will be reported at the call site
not the function body; other call sites that call the same function with
different types may be fine.

So while you can define constraints for template type parameters, they are
needed for generic type parameters. In fact type constraints are the main thing
we need to add to support generic type parameters, beyond what is described in
[the "non-type generics" section above](#non-type-generics).

## Proposed programming model

This is described in detail in a separate document,
["Carbon deep dive: combined interfaces"](combined-interfaces.md). In summary:

-   Interfaces have a name and describe functions and other items for types to
    implement.
-   Types may implement interfaces at most once.
-   Implementations may be part of the type's definition, in which case you can
    directly call functions on those types. Or they may be external, in which
    case the implementation is allowed to be defined in the library defining the
    interface. The goal is that methods for interfaces implemented with the type
    are generally available without qualification, and a function with a generic
    type parameter can have the same function body as an unparameterized one.
-   Interfaces are usable as type-types, meaning you can define a generic
    function that have a type parameter satisfying the interface by saying the
    type of the type parameter is the interface. Inside such a generic function,
    the API of the type is erased, except for the names of the interface.
-   You may also declare type-types, called "structural interfaces", that
    require multiple interfaces to be implemented, and give you control over how
    name conflicts are handled.
-   Alternatively, you may use a qualified syntax to directly call a function
    from a specific interface.
-   The `+` operation on type-types allows you conveniently combine interfaces.
    It gives you all the names that don't conflict. Names with conflicts can be
    accessed using the qualified syntax.
-   Interfaces can require other interfaces be implemented, or extend/refine
    them.

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

Interface definition

```
interface Foo {
  // `F` is an associated method.
  method (Ptr(Self): this) F();
}
```

Interface with associated type conforming to an interface

```
interface Bar {
  var Foo:$ U;

  method (Ptr(Self): this) G() -> U;
}
```

Type that implements an interface

```
struct Baz {
  // ...

  // Implementing `Foo` for `Baz` inside the definition of `Baz`
  // means all names of `Foo`, such as `F`, are included as a part
  // of the `Baz` API.
  impl Foo {
    // Could use `Self` in place of `Baz` here.
    method (Ptr(Baz): this) F() { ... }
  }
}
// Implement `Bar` for `Baz` without changing the API of `Baz`
// using an `extend` declaration. This may be defined in either
// the library defining `Baz` or `Bar`.
extend Baz {
  impl Bar {
    var Foo:$ U = ...;
    method (Ptr(Self): this) G() -> U { ... }
  }
}

var Baz: x;
// `x.F()` is allowed, unlike `x.G()`.
x.F();
// To call `G` on `x`, use the qualified syntax:
x.(Bar.G)();
// Can also call `F` using the qualified syntax:
x.(Foo.F)();
```

Function taking a value with type conforming to an interface

```
fn H1(Ptr(Foo:$ T): y) {
  y->F();
}
fn H2[Foo:$ T](Ptr(T): y) {
  y->F();
}
H1(&x); H2(&x);
```

Function taking a value with type conforming to two different interfaces

```
fn Ha[Foo + Bar:$ T](Ptr(T): y) {
  // `T` has all the names of `Foo` and `Bar` that don't conflict.
  y->F();
  // Can call `G` here, even though `x.G()` isn't allowed since
  // `Bar` is external.
  y->G();
  // Qualified syntax works even if there is a name conflict between
  // Foo and Bar.
  y->(Bar.G)();
}
Ha(&x);
```

Function taking a value with a list of implementations for a single interface,
all compatible with a single representation type

```
// Compares `*a` with `*b`, returning the result of the first
// comparison in `compare` that considers them unequal.
// This could be used to implement lexicographical comparison.
fn IsGreater[Type:$ T](Ptr(T): a, Ptr(T): b,
                       List(CompatibleWith(Comparable, T)):$ compare)
    -> Bool { ... }
```

Function taking a list of values with different types that all implement a
single interface

```
// `NTuple(N, ToString)` is `(ToString, ..., ToString)`, the tuple
// with `N` copies of `ToString`. `Ts` is a tuple of `N` types
// `(Ts[0], ..., Ts[N-1])`, where each type `Ts[i]` has type `ToString`.
// `input` is a tuple of `N` values, where `input[i]` has type `Ts[i]`.
fn StringConcat[Int:$ N, NTuple(N, ToString):$ Ts](Ts...: input)
    -> String { ... }
```

Interface semantically containing other interfaces

```
interface Inner {
  method (Self: this) K();
}

// `Outer1` requires that `Inner` is implemented.
interface Outer1 {
  impl Inner;
}
struct S {
  // ...
  impl Inner {
    method (S: this) K() { ... }
  }
  impl Outer1 { }
}
var S: y = ...;
y.K();

// `Outer2` refines `Inner`.
interface Outer2 {
  extends Inner;
}
// `Outer2` is equivalent to:
interface Outer2 {
  impl Inner;
  alias K = Inner.K;
}

struct T {
  // ...
  impl Outer2 {
    method (T: this) K() { ... }
  }
  // No need to separately implement `Inner`.
}
var S: z = ...;
z.K();
```

Interface structurally consisting of other interfaces

```
interface A { ... }
interface B { ... }

// Combined1 has all names from A and B, must not conflict.
// Can use qualification (`x.(A.F)()`), but not required.
structural interface Combined1 {
  extends A;
  extends B;
}
// Combined2 has all names from A and B that do not conflict.
// Can use qualification (`x.(A.F)()`) to get any names from
// `A` or `B` even if there is a conflict.
structural interface Combined2 {
  extends A + B;
}
// Combined3 only has names mentioned explicitly.
// Can use qualification (`x.(A.F)()`) to get any names from
// `A` or `B` even if they are not mentioned in `Combined3`.
structural interface Combined3 {
  impl A;
  impl B;
  alias F = A.F;
  alias G_A = A.G;
  alias G_B = B.G;
}

// All of these functions accept the same values, namely anything
// with a type implementing both `A` and `B`.
fn F1[Combined1:$ T](T: x) { ... }
fn F2[Combined2:$ T](T: x) { ... }
fn F3[Combined3:$ T](T: x) { ... }
fn FPlus[A + B:$ T](T: x) { ... }
```

### Calling templated code

["Passing generic arguments to template parameter"](https://github.com/josh11b/carbon-lang/blob/generic-to-template/docs/design/generics/generic-to-template.md)
