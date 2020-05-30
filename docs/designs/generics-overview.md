<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon Generics

## Context

This is version 2 of a proposal for generics, and replaces the description of
generics in
[Carbon Templates and Generics (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics -->.
That document also describes how templates work in detail, which we only touch
on briefly here.

**NOTE:** We fully expect that we will not get generics right at first, they are
even more subject to change than the other parts of Carbon.

## What are generics?

Generics are a mechanism for writing more abstract code that applies more
generally instead of making near duplicates for very similar situations, much
like templates. For example, instead of having one function per
type-you-can-sort:

```
fn SortInt32Array(Array(Int32)*: a) { ... }
fn SortStringArray(Array(String)*: a) { ... }
...
```

you might have one function that could sort any array with comparable elements:

```
fn SortArray[Comparable:$ T](Array(T)*: a) { ... }
```

Where the `SortArray` function applied to an `Array(Int32)*` input is
semantically identical to `SortInt32Array`, and similarly for `Array(String)*`
input and `SortStringArray`.

Here `Comparable` is the name of an _interface_, which describes the
requirements for the type `T`. These requirements form the contract that allows
us to have an API boundary encapsulating the implementation of the function,
unlike templates. I.e., given that we know `T` satisfies the requirements, we
can typecheck the body of the `SortArray` function; similarly, we can typecheck
that a call to `SortArray` is valid by checking that the type of the member
elements of the passed-in array satisfy the same requirements, without having to
look at the body of the `SortArray` function. These are in fact the main
differences between generics and templates:

- We can completely typecheck a generic definition without information from the
  callsite.
- We can completely typecheck a call to a generic with just information from the
  function's signature, not its body.

Contrast with a template function, where you may be able to do some checking
given a function definition, but more checking of the definition is required
after seeing the call sites (and you know which specializations are needed).

Read more here:
[Carbon Generics: Terminology and Problem Statement: "Generic vs. template arguments" section](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#generic-vs-template-arguments).

## Goals: Generics

In general we aim to make Carbon Generics into an alternative to templates for
writing generic code, with improved software engineering properties at the
expense of some restrictions. See
[Carbon principle: Generics](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md)
for a detailed discussion of goals. Also see [motivational use cases](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-motivation.md).

In this proposal we try and define a generics system that has these properties
to allow migration from templates:

- Templated code (perhaps migrated from C++) can be converted to generics
  incrementally, one function or argument at a time. Typically this would
  involve determining the interfaces that generic types need to implement to
  call the function, proving the API the function expects.
- It should be legal to call templated code from generic code when it would have
  the same semantics as if called from non-generic code, and an error otherwise.
  This is to allow more templated functions to be converted to generics, instead
  of requiring them to be converted specifically in bottom-up order.
- Converting from a template to a generic argument should be safe -- it should
  either fail to compile or work, never silently change semantics.
- We should minimize the effort to convert from template code to generic code.
  Ideally it should just require specifying the type constraints, affecting just
  the signature of the function, not its body.

Also we would like to support:

- Selecting between a dynamic and a static strategy for the generated code, to
  give the user control over performance, binary sizes, build speed, etc.
- Use of generic functions with types defined in different libraries from those
  defining the interface requirements for those types.

MAYBE: migration to generics from inheritance, to disentangle subtyping from
implementation inheritance.

## Glossary / Terminology

See
[Carbon Generics: Terminology and Problem statement](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#terminology)

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

What would it mean to change the argument to be a generic argument?

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
var Int: n = 3;
PrintXs_Generic(n);  // Compile error: value for generic argument `n`
                     // unknown at compile time.
```

For the definition of the function there is only one difference: we added a `$`
to indicate that the argument named `n` is generic. The body of the function
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

Since a function with a generic argument can have many different addresses, we
have this rule:

**Rule:** It is illegal to take the address of any function with generic
arguments (similarly template arguments).

This rule also makes the difference between the compiler generating separate
specializations or using a single generated function with runtime dynamic
dispatch harder to observe, enabling the compiler to switch between those
strategies without danger of accidentally changing the semantics of the program.

**NOTE:** The `$` syntax is temporary and won't be the final syntax we use,
since it is not easy to type `$` from non-US keyboards. Instead of `:$`,
[we are considering (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:# -->:
`:!`, `:@`, `:#`, and `::`. We might use the same character here as we decide
for
[Carbon metaprogramming (TODO)](#broken-links-footnote)<!-- T:Carbon metaprogramming -->
constructs.

### Basic templates

For this function, we could change the argument to be a template argument by
replacing "`Int:$ n`" with "`Int:$$ n`", but there would not be a difference
that you would observe. However, with template arguments we would have more
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
var Int: n = 3;
PrintXs_Template(n);  // Compile error: value for template argument `n`
                      // unknown at compile time.
PrintXs_Template(3);  // Compile error: NumXs(3) undefined.
```

Since type checking is delayed until `n` is known, we don't need the return type
of `NumXs` to be consistent across different values of `n`.

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

### Implicit arguments

An implicit argument is a value that is determined by the type of the value
passed to another argument, and not passed explicitly to the function. Implicit
arguments are passed using square brackets before the usual parameter list, as
in:

```
fn PrintArraySize[Int: n](FixedArray(String, n)*: array) {
  Print(n);
}

var FixedArray(String, 3): a = ...;
PrintArraySize(&a);  // Prints: 3
```

What happens here is the type for the `array` argument is determined from the
value passed in, and the pattern-matching process used to see if the types match
finds that it does match if `n` is set to `3`.

Normally you would pass in an implicit argument as a generic or template, not as
a regular argument. This avoids overhead from having to support types (like the
type of `array` inside the `PrintArraySize` function body) that are only fully
known with dynamic information. For example:

```
fn PrintStringArray[Int:$ n](FixedArray(String, n)*: array) {
  for (var Int: i = 0; i < n; ++i) {
    Print(array->get(i));
  }
}
```

\
Implicit arguments are always determined from the explicit arguments. It is illegal
not to mention implicit arguments in the explicit argument list and there is no syntax
for specifying implicit arguments directly at the call site.

```
// ERROR: can't determine `n` from explicit arguments
fn Illegal[Int:$ n](Int: i) -> Bool { return i < n; }
```

### Mixing

- A function can have a mix of generic, template, and regular arguments.
- Can pass a template or generic value to a generic or regular parameter.

### Generic members

You may also have generic members of functions/types. Just like generic
parameters, they have compile-time, not runtime, storage. You may also have
template members, with the difference that template members can use the actual
value of the member. In both cases, these can be initialized with values
computed from generic/template arguments, or other things that are effectively
constant and/or available at compile time.

A generic function member allows you define a local constant:

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

Template function members may be used in type checking:

```
fn PrimesLessThan(Int:$$ N) {
  var Int:$$ MaxNumPrimes = N / 2;
  // Value of MaxNumPrimes is available at type checking time.
  var FixedArray(Int, MaxNumPrimes): primes;
  var Int: num_primes_found = 0;
  // ...
}
```

Similarly, you may also use generic and template members of types, which allows
you to define constant members of the type.

## Type generics design space

If we want to be generic across types, in order to do type checking we may need
to put some constraints on the type so that a
[parameterized language construct](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#parameterized-language-constructs)
like a function can (for example) call methods defined for values of that type.
A type constraint is called an
[interface](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#interface).

Interfaces can match structurally or you can make there be a separate step to
explicitly say that there is additional semantic information to say that a type
conforms to an interface; see
[Carbon Generics: Terminology and Problem statement: "Semantic vs. structural interfaces"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#semantic-vs-structural-interfaces).
If interfaces are semantic, then interface implementations can be part of a type
("facets") or separate, named entities ("witnesses").

Types can contain interface implementations or implement them. The difference is
whether the API described by the interface is included in the type's interface,
or if you have to take an explicit step to say "use the one from that
interface".

### Problem statement

[See "Problem statement" section in "Carbon Generics: Terminology and Problem statement"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#problem-statement).

Generally speaking, we want generics and interfaces to be sufficiently
expressive and convenient that we can skip open overloading. Open overloading
causes a number of problems, and closed function overloading is significantly
simpler. See the more in-depth discussion in
[Carbon closed function overloading proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal -->.
This means both sufficient expressivity, but also being able to address the same
use cases as open overloading without excessive ceremony.

Older context:
[Carbon closed function overloading proposal: "Open problems" section (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#heading=h.gjf459pitr6v -->

### Generic type parameters vs. templated type parameters

Recall, from
[the "Difference between templates and generics" section above](#difference-between-templates-and-generics),
that we fully check functions with generic arguments at the time they are
defined, while functions with template arguments can use information from the
caller.

If you have a value of a generic type, you need to provide constraints on that
type that define what you can do with values of that type. However when using a
templated type, you can perform any operation on values of that type, and what
happens will be resolved once that type is known. This may be an error if that
type doesn't support that operation, but that will be reported at the call site
not the function body; other call sites that call the same function with
different types may be fine.

So while you can define constraints for template type arguments, they are needed
for generic type arguments. In fact type constraints are the main thing we need
to add to support generic type arguments, beyond what is described in
[the "non-type generics" section above](#non-type-generics).

### Programming model proposals

(Quoting heavily from the linked docs.)

#### Interfaces are concrete types

##### [Carbon templates and generics (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.atq778faeibo -->

```
interface Printable(Type: T) {
  fn Print(T: thing);
}
template fn PrintSomething[Type: T](T: something) where Printable(T) {
  Printable.Print(something);
}
interface Summable(Type: T) {
  fn Sum(T: left, T: right) -> T;
}
template fn PrintSum[Type: T](T: left, T: right) where Printable(T), Summable(T) {
  Printable.Print(Summable.Sum(left, right));
}
interface MemberPrintable(Type: T) for T {
  // Using whatever syntax we end up with to describe the member object...
  fn Print(T: self);
}
template fn PrintWithMember[Type: T](T: something) where MemberPrintable(T) {
  something.Print();
}
```

##### [Carbon: facet types and interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon: facet types and interfaces --><!-- A:#heading=h.6ox7fklhkrei -->

"A facet type provides an alternative interface for another type. Concretely, a
facet type has some associated storage type, and shares the storage behavior
with that type, but has distinct interface behavior (typically implementing some
interface for the type)."

```
interface Cmp(Type: T) where Self = T {  // shorthand: interface Cmp(Type: Self) {
  public fn compare(Self*: this, Self*: that) -> Ordering;
}
impl Cmp(A) {
  fn compare(A*: this, A*: that) -> Ordering { /*...*/ }
}
```

This defines a facet type `Cmp(A)` whose storage behavior is the same as its
self-type (`A`) but that has different interface behavior. The interface of a
facet type is always exactly the interface that the facet type implements.
Because the name of the facet type is `Cmp(A)`, this effectively extends the
behavior of the `Cmp` function to be able to produce a comparison facet for `A`.

In this model, the name of an interface is a partially-defined function from
type->type that looks up a storage type and returns a compatible type (same
storage representation, can cast values without copying the data) with the API
defined by the interface.

A parameterized implementation might be written as something like:

```
// Given a type T, with a constraint that there is a Cmp(T), implement Cmp(Vec(T)).
impl[Type: T, Cmp(T)] Cmp(Vec(T)) {
   …   // Need to cast from T -> Cmp(T) to compare
}
// or more idiomatic:
impl[Type: T] Cmp(Vec(Cmp(T))) {
   ...  // No need to cast, values are already of type Cmp(T).
}
```

[From the doc (TODO)](#broken-links-footnote)<!-- T:Carbon: facet types and interfaces --><!-- A:#heading=h.fjx0bardg0ov -->:
When we define

```
interface Cmp(Type: T) where Self = T {
```

what are we actually defining? What is`Cmp`, exactly?

Proposal: `Cmp` is a function that returns a facet type whose storage type is
the storage type of its self-type and that implements the specified interface.
An `impl` declaration defines a facet type and extends the set of input ->
output pairs of the interface function with that facet type.

##### [Follow-up Jan 23, 2020 (TODO)](#broken-links-footnote)<!-- T:Follow-up Jan 23, 2020 -->

Observations:

- Rust models interfaces as type-types.
- Rust has a special `<Type as Interface>` syntax parallel to its
  `(Value as Type)` syntax.
- Rust has a special type constraint language for defining traits.
- With facets, it is hard to say "this interface I1 has an associated type T
  that satisfies interface I2", because in general in this model it is hard to
  have a declaration saying that "X names a type that satisfies a particular
  interface".
- Saying an interface is "`for T`" means the same thing as "`require Self ~ T`"
  where `U ~` means "satisfies the `HasSameRepr(U, T)` interface". (Except, we
  might want to define a slightly different relationship with an impl for the
  two forms.)
- Generally speaking, you would make interfaces of the form "`Foo(T) for T`".
  Instead of additional arguments, you probably want associated types. Instead
  of something different after "`for`", generally would instead just have that
  as part of a specific impl (though Josh made an argument that there is at
  least some utility for putting restrictions into the interface.)
- There might be a use case for interface / impl separation.

##### [Deep dive: interfaces as concrete types (TODO)](#broken-links-footnote)<!-- T:Deep dive: interfaces as concrete types --><!-- A:# -->

A fairly complete and up-to-date description of this programming model.

**Pro:** Single framework for classes/methods and generics

**Concerns:**

- Verbosity for simple cases:
  - Type parameter or associated type must satisfy interface
  - TODO
- Arguments to interfaces serve two different functions, one is generally for
  the representation type, others are for parameterization
- Can't say things like "list of implementations of this interface compatible
  with a specific type"

#### Interfaces are facet type-types

##### [Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.3kqiirqlj97f -->

In this model, an interface is a type whose values are "types that conform to
that interface".

This proposal uses
[named impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#named-impl),
but we'd probably prefer to just have
[default impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-terminology.md#default-impl)
like the next iteration described next.

```
interface I {
  fn F(Ptr(self): this);
  fn G(Ptr(self): this);
}
// The meaning of self is a little tricky:
// I is a type-type, which means that some types T will have type I.
// Which types T meet the requirements to have type I?
// Those with member functions F and G with types fn(Ptr(T): this).

fn CompileTime[I:$ T](Ptr(T): a) {
  a->F();
}
impl FooI(Foo, .implements = I) {
  .F = Foo.F;  // May only name public members of `Foo`.
  fn G(Ptr(Foo): this) {
    Print("Via interface! ");
    this->G();  // Naturally restricted to just the public API of Foo.
  }
}
// FooI is of type impl(Foo, .implements = I) (which extends I).
CompileTime(FooI.CastToGeneric(&a));  // Explicit
CompileTime(&a);  // Implicit
interface I2 extends I {  ... }  // Supports interface inheritance
// Supports interfaces with type arguments
interface Tree(Type: T) { ... }
interface PushPopContainer(Type: T) {
  fn Push(Ptr(Self): this, T: x);
  fn Pop(Ptr(Self): this) -> T;
}
fn TreeTraversal[Type:$ T, Tree(T):$ TreeType](
    Ptr(TreeType): to_traverse,
    PushPopContainer(T):$ stack_or_queue_type,
    fn(T): visitor_function) { ... }
TreeTraversal(&my_int_tree, Stack(Int), fn(Int: x) { Print(x); });
```

##### [Carbon deep dive: interfaces as facet type-types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md)

This is JoshL@'s favorite proposal at this time, since it can express things
like "list of implementations for an interface X for representation T", is
reasonably concise for simple tasks, and has no major objections.

```
// Define an interface
interface Vector {
    // Here "Self" means "the type implementing this interface"
  fn Add(Self: a, Self: b) -> Self;
  fn Scale(Self: a, Double: v) -> Self;
}

// Type that implements an interface
struct Point {
  var Double: x;
  var Double: y;
  impl Vector {  // may also impl out of line
    // Here "Self" is an alias for "Point"
    fn Add(Self: a, Self: b) -> Self {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    }
    fn Scale(Self: a, Double: v) -> Self {
      return Point(.x = a.x * v, .y = a.y * v);
    }
  }
}

// Function that takes values of any type that implements the interface
fn AddAndScale[Vector:$ T](T: a, T: b, Double: s) -> T {
  return a.Add(b).Scale(s);
}
var Point: a = (.x = 1.0, .y = 2.0);
var Point: b = (.x = 3.0, .y = 4.0);
var Point: v = AddAndScale(a, b, 2.5);

// Interfaces can be parameterized
interface Stack(Type:$ ElementType) {
  fn Push(Self*: this, ElementType: value);
  fn Pop(Self*: this) -> ElementType;
  fn IsEmpty(Self*: this) -> Bool;
}

// Type parameters for interfaces can be constrained
fn SumIntStack[Stack(Int):$ T](T*: s) -> Int { ... }
// More complicated example: two containers constrained to have the same element
// type, which itself is constrained to implement the HasEquality interface.
fn EqualContainers[HasEquality:$ ElementType,
                   Container(ElementType):$ T1,
                   Container(ElementType):$ T2](T1*: c1, T2*: c2) -> Bool { ... }
```

\
**Concern:** Does not represent binary relationships between two types like "A is
comparable to B (and therefore B is comparable to A)".

#### Impls are values passed as arguments with defaults

For details, see:
[Carbon: Impls are values passed as arguments with defaults (TODO)](#broken-links-footnote)<!-- T:Carbon: Impls are values passed as arguments with defaults -->.

Main idea: default implementation of interface for a type, available as
`Type.DefaultImpl(Interface)`. Can be used as default value in function
signature:

```
fn Sort[Type: T](Array(T)*: to_sort,
                 Impl(Comparable, T): compare = T.DefaultImpl(Comparable));
```

(We would need a more concise syntax for this.) If `T` doesn't have an impl for
`Comparable`, then the compiler emits an error (or considers this overload
non-viable) unless an explicit value is provided for `compare` at the caller.

Also see
[Carbon meeting Nov 27, 2019 on Generics & Interfaces: Approach 2 (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.b9ww6a2g3tdi -->

**Concern:** Does not handle conditional conformance as gracefully.

#### Type-types parameterized by reprs

[Carbon meeting Nov 27, 2019 on Generics & Interfaces: Approach 1 (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.n2k3narvy12b -->

**Proposal:** Let us define an interface as follows:

    **Interface:** A interface such as `Foo(T)` is a type whose values are themselves types that have the same representation as `T` and provides the functions defined by `Foo` operating on [`T`s | that type].

So we might define the interface `Comparable` like so:

```
struct Comparable(Type: T) {
  var fn(T, T) -> Bool : Compare;
}
```

So if `U` is of type `Comparable(T)` then `U` and `T` have the same
representation and you may cast freely between them. Furthermore, you may call
`Compare` on two values of type `U`.

This is like
[the "type-types" model above](#carbon-types-as-function-tables-interfaces-as-type-types),
but where the interfaces include a parameter that defines the representation
type.

Two main concerns (from
[Carbon Generics meeting Jan 22, 2020 (TODO)](#broken-links-footnote)<!-- T:Carbon Generics meeting Jan 22, 2020 --><!-- A:#heading=h.g0n6klm6qsdz -->):

- In our system, multiple types will have the same representation, so there is
  ambiguity about what representation type should be inferred. This has led to a
  lot of confusion when working out edge cases.
- It ends up verbose even for simple cases.

### Comparisons

#### "Type-types parameterized by reprs" vs "facet types"

[Carbon Generics meeting Jan 22, 2020 (TODO)](#broken-links-footnote)<!-- T:Carbon Generics meeting Jan 22, 2020 -->

#### "Interfaces are concrete types" vs. "Facet type-types"

**Interfaces are concrete types**

Interface definition

```
interface Foo(Type:$ T) for T {
  fn F(Self*: this);
}
```

\
Interface with associated type conforming to an interface

```
interface Bar(Type:$ T) for T {
  var Type:$ U;
  require Foo(U);
  fn G(Self*: this) -> Foo(U);
}
```

\
Type that implements an interface

```
struct Baz {
  impl Foo(Baz) {
    fn F(Baz*: this) { ... }
  }
}
impl Bar(Baz) { ... }

var Baz: x;
```

\
Function taking a value with type conforming to an interface

```
fn H1(Foo(Type:$ T)*: y) { y->F(); }
fn H2[Type:$ T](Foo(T)*: y) {
  y->F();
}
fn H3[Type:$ T, Foo(T)]
    (T*: y) {
  (y as Foo(T)*)->F();
}
H1(&x); H2(&x); H3(&x);
```

\
Function taking a value with type conforming to two different interfaces

```
fn Ha[Type:$ T, Foo(T), Bar(T)]
    (T*: y) {
  (y as Foo(T)*)->F();
  var Bar(T)*: z = y;
  z->G();
}
Ha(&x);
```

\
Function taking a value with a list of implementations for a single interface, all
compatible with a single representation type

```
???


```

\
**Facet type-types**

Interface definition

```
interface Foo {
  fn F(Self*: this);
}
```

\
Interface with associated type conforming to an interface

```
interface Bar {
  var Foo:$ U;

  fn G(Self*: this) -> U;
}
```

\
Type that implements an interface

```
struct Baz {
  impl Foo {
    fn F(Baz*: this) { ... }
  }
}
impl Bar for Baz { ... }

var Baz: x;
```

\
Function taking a value with type conforming to an interface

```
fn H1((Foo:$ T)*: y) { y->F(); }
fn H2[Foo:$ T](T*: y) {
  y->F();
}
fn H3[TypeImplements(Foo):$ T]
    (T*: y) {
  (y as (T as Foo)*)->F();
}
H1(&x); H2(&x); H3(&x);
```

\
Function taking a value with type conforming to two different interfaces

```
fn Ha[TypeImplements(Foo, Bar):$ T]
    (T*: y) {
  (y as (T as Foo)*)->F();
  var (T as Bar)*: z = y;
  z->G();
}
Ha(&x);
```

\
Function taking a value with a list of implementations for a single interface, all
compatible with a single representation type

```
fn IsGreater[Type:$ T](T*: a, T*: b,
 List(CompatibleWith(Comparable, T)):$
 compare) -> Bool { ... }
```

Interface semantically containing other interfaces

```
interface Inner {
  fn K(Self: this);
}
interface Outer {
  impl Inner;
}
struct S {
  impl Outer {
    impl Inner {
      fn K(S: this) { ... }
    }
  }
}
var S: y = ...;
(y as ((S as Outer) as Inner)).K();
```

\
TODO: Interface structurally consisting of other interfaces

TODO: Subsumption

TODO: Intersection of constraints: two interfaces, interface and representation
constraint

TODO

## Proposed programming model

[Carbon deep dive: interfaces as facet type-types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md)

### Calling templated code

["How Carbon generics can use templates" (TODO)](#broken-links-footnote)<!-- T:How Carbon generics can use templates -->

(this was the follow up to the conversation started in
[Carbon discussion re: Generics & Templates Jan 17, 2020 (TODO)](#broken-links-footnote)<!-- T:Carbon discussion re: Generics & Templates Jan 17, 2020 --><!-- A:#heading=h.svuyvyvy6lde -->)

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
