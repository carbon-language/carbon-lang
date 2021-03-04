<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Comparison of Carbon Generics alternatives

## Table of contents

<!-- toc -->

-   [Type generics design space](#type-generics-design-space)
    -   [Problem statement](#problem-statement)
-   [Programming model proposals](#programming-model-proposals)
    -   [Interfaces are concrete types](#interfaces-are-concrete-types)
        -   [Carbon templates and generics (TODO)](#carbon-templates-and-generics-todo)
        -   [Carbon: facet types and interfaces (TODO)](#carbon-facet-types-and-interfaces-todo)
        -   [Follow-up Jan 23, 2020 (TODO)](#follow-up-jan-23-2020-todo)
        -   [Deep dive: interfaces as concrete types (TODO)](#deep-dive-interfaces-as-concrete-types-todo)
    -   [Interfaces are facet type-types](#interfaces-are-facet-type-types)
        -   [Carbon: types as function tables, interfaces as type-types (TODO)](#carbon-types-as-function-tables-interfaces-as-type-types-todo)
        -   [Carbon deep dive: interfaces as facet type-types](#carbon-deep-dive-interfaces-as-facet-type-types)
        -   [Carbon deep dive: combined interfaces](#carbon-deep-dive-combined-interfaces)
    -   [Impls are values passed as arguments with defaults](#impls-are-values-passed-as-arguments-with-defaults)
    -   [Type-types parameterized by reprs](#type-types-parameterized-by-reprs)
-   [Comparisons](#comparisons)
    -   ["Type-types parameterized by reprs" vs "facet types"](#type-types-parameterized-by-reprs-vs-facet-types)
    -   [Interfaces are concrete types](#interfaces-are-concrete-types-1)
    -   [Facet type-types](#facet-type-types)
    -   [Combined interfaces](#combined-interfaces)
-   [Broken links footnote](#broken-links-footnote)

<!-- tocstop -->

## Type generics design space

If we want to be generic across types, in order to do type checking we may need
to put some constraints on the type so that a
[parameterized language construct](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#parameterized-language-constructs)
like a function can (for example) call methods defined for values of that type.
A type constraint is called an
[interface](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface).

Interfaces can match structurally or you can make there be a separate step to
explicitly say that there is additional semantic information to say that a type
conforms to an interface; see
[Carbon Generics: Terminology and Problem statement: "Semantic vs. structural interfaces"](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#semantic-vs-structural-interfaces).
If interfaces are semantic, then interface implementations can be part of a type
("facets") or separate, named entities ("witnesses").

Types can contain interface implementations or implement them. The difference is
whether the API described by the interface is included in the type's interface,
or if you have to take an explicit step to say "use the one from that
interface".

### Problem statement

[See the "use cases and problem statement" document about Carbon Generics](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/use-cases.md).

Generally speaking, we want generics and interfaces to be sufficiently
expressive and convenient that we can skip open overloading. Open overloading
causes a number of problems, and closed function overloading is significantly
simpler. See the more in-depth discussion in
[Carbon closed function overloading proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal -->.
This means both sufficient expressivity, but also being able to address the same
use cases as open overloading without excessive ceremony.

Older context:
[Carbon closed function overloading proposal: "Open problems" section (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#heading=h.gjf459pitr6v -->

## Programming model proposals

(Quoting heavily from the linked docs.)

### Interfaces are concrete types

#### [Carbon templates and generics (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.atq778faeibo -->

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

#### [Carbon: facet types and interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon: facet types and interfaces --><!-- A:#heading=h.6ox7fklhkrei -->

"A facet type provides an alternative interface for another type. Concretely, a
facet type has some associated storage type, and shares the storage behavior
with that type, but has distinct interface behavior (typically implementing some
interface for the type)."

```
// shorthand: interface Cmp(Type: Self) {
interface Cmp(Type: T) where Self = T {
  public fn compare(Ptr(Self): this, Ptr(Self): that) -> Ordering;
}
impl Cmp(A) {
  fn compare(Ptr(A): this, Ptr(A): that) -> Ordering { /*...*/ }
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

#### [Follow-up Jan 23, 2020 (TODO)](#broken-links-footnote)<!-- T:Follow-up Jan 23, 2020 -->

Observations:

-   Rust models interfaces as type-types.
-   Rust has a special `<Type as Interface>` syntax parallel to its
    `(Value as Type)` syntax.
-   Rust has a special type constraint language for defining traits.
-   With facets, it is hard to say "this interface I1 has an associated type T
    that satisfies interface I2", because in general in this model it is hard to
    have a declaration saying that "X names a type that satisfies a particular
    interface".
-   Saying an interface is "`for T`" means the same thing as
    "`require Self ~ T`" where `U ~` means "satisfies the `HasSameRepr(U, T)`
    interface". (Except, we might want to define a slightly different
    relationship with an impl for the two forms.)
-   Generally speaking, you would make interfaces of the form "`Foo(T) for T`".
    Instead of additional arguments, you probably want associated types. Instead
    of something different after "`for`", generally would instead just have that
    as part of a specific impl (though Josh made an argument that there is at
    least some utility for putting restrictions into the interface.)
-   There might be a use case for interface / impl separation.

#### [Deep dive: interfaces as concrete types (TODO)](#broken-links-footnote)<!-- T:Deep dive: interfaces as concrete types --><!-- A:# -->

A fairly complete and up-to-date description of this programming model.

**Pro:** Single framework for classes/methods and generics

**Concerns:**

-   Verbosity for simple cases:
    -   Type parameter or associated type must satisfy interface
    -   TODO
-   Arguments to interfaces serve two different functions, one is generally for
    the representation type, others are for parameterization
-   Can't say things like "list of implementations of this interface compatible
    with a specific type"

### Interfaces are facet type-types

#### [Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.3kqiirqlj97f -->

In this model, an interface is a type whose values are "types that conform to
that interface".

This proposal uses
[named impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#named-impl),
but we'd probably prefer to just have
[default impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#default-impl)
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

fn GenericCompileTimeTypeParam[I:$ T](Ptr(T): a) {
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
GenericCompileTimeTypeParam(FooI.CastToGeneric(&a));  // Explicit
GenericCompileTimeTypeParam(&a);  // Implicit
interface I2 extends I {  ... }  // Supports interface inheritance
// Supports interfaces with type arguments
interface Tree(Type: T) { ... }
interface PushPopContainer(Type: T) {
  method (Ptr(Self): this) Push(T: x);
  method (Ptr(Self): this) Pop() -> T;
}
fn TreeTraversal[Type:$ T, Tree(T):$ TreeType](
    Ptr(TreeType): to_traverse,
    PushPopContainer(T):$ stack_or_queue_type,
    fn(T): visitor_function) { ... }
TreeTraversal(&my_int_tree, Stack(Int), fn(Int: x) { Print(x); });
```

#### [Carbon deep dive: interfaces as facet type-types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/facet-type-types.md)

Josh11b@ prefers this proposal over
[interfaces as concrete types](#deep-dive-interfaces-as-concrete-types-todo),
since it can express things like "list of implementations for an interface X for
representation T" and "list of types that all implement interface X", is
reasonably concise for simple tasks, and has no major objections. However, it is
cumbersome to access methods defined as part of the implementation of an
interface outside of a generic, or to operate on types implementing multiple
interfaces inside a generic.

```
// Define an interface
interface Vector {
    // Here "Self" means "the type implementing this interface"
  method (Self: a) Add(Self: b) -> Self;
  method (Self: a) Scale(Double: s) -> Self;
}

// Type that implements an interface
struct Point {
  var Double: x;
  var Double: y;
  impl Vector {  // may also impl out of line
    // Here "Self" is an alias for "Point"
    method (Self: a) Add(Self: b) -> Self {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    }
    method (Self: a) Scale(Double: s) -> Self {
      return Point(.x = a.x * s, .y = a.y * s);
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
  method (Ptr(Self): this) Push(ElementType: value);
  method (Ptr(Self): this) Pop() -> ElementType;
  method (Ptr(Self): this) IsEmpty() -> Bool;
}

// Type parameters for interfaces can be constrained
fn SumIntStack[Stack(Int):$ T](Ptr(T): s) -> Int { ... }
// More complicated example: two containers constrained to have the same element
// type, which itself is constrained to implement the HasEquality interface.
fn EqualContainers[HasEquality:$ ElementType,
                   Container(ElementType):$ T1,
                   Container(ElementType):$ T2]
                  (Ptr(T1): c1, Ptr(T2): c2) -> Bool { ... }
```

**Concern:** Does not represent binary relationships between two types like "A
is comparable to B (and therefore B is comparable to A)".

#### [Carbon deep dive: combined interfaces](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/combined-interfaces.md)

This is a modification of
[interfaces as facet type-types](#carbon-deep-dive-interfaces-as-facet-type-types)
that attempts to improve usability:

-   Methods for interfaces implemented with the type are generally available
    without qualification.
-   A function can be parameterized by a generic type implementing multiple
    interfaces, and methods from all interfaces may be called without
    qualification as long as there are no conflicts.
-   We provide a more concise qualification syntax for when that is needed.

TODO

### Impls are values passed as arguments with defaults

For details, see:
[Carbon: Impls are values passed as arguments with defaults (TODO)](#broken-links-footnote)<!-- T:Carbon: Impls are values passed as arguments with defaults -->.

Main idea: default implementation of interface for a type, available as
`Type.DefaultImpl(Interface)`. Can be used as default value in function
signature:

```
fn Sort[Type: T](Ptr(Array(T)): to_sort,
                 Impl(Comparable, T): compare = T.DefaultImpl(Comparable));
```

(We would need a more concise syntax for this.) If `T` doesn't have an impl for
`Comparable`, then the compiler emits an error (or considers this overload
non-viable) unless an explicit value is provided for `compare` at the caller.

Also see
[Carbon meeting Nov 27, 2019 on Generics & Interfaces: Approach 2 (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.b9ww6a2g3tdi -->

**Concern:** Does not handle conditional conformance as gracefully.

A brief sketch of this idea (without defaults) was suggested in
[Another Generic Dilemma](https://matklad.github.io//2021/02/24/another-generic-dilemma.html).

### Type-types parameterized by reprs

[Carbon meeting Nov 27, 2019 on Generics & Interfaces: Approach 1 (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.n2k3narvy12b -->

**Proposal:** Let us define an interface as follows:

> **Interface:** A interface such as `Foo(T)` is a type whose values are
> themselves types that have the same representation as `T` and provides the
> functions defined by `Foo` operating on [`T`s | that type].

So we might define the interface `Comparable` like so:

```
struct Comparable(Type: T) {
  var fntype(T, T) -> Bool : Compare;
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

-   In our system, multiple types will have the same representation, so there is
    ambiguity about what representation type should be inferred. This has led to
    a lot of confusion when working out edge cases.
-   It ends up verbose even for simple cases.

## Comparisons

### "Type-types parameterized by reprs" vs "facet types"

[Carbon Generics meeting Jan 22, 2020 (TODO)](#broken-links-footnote)<!-- T:Carbon Generics meeting Jan 22, 2020 -->

### Interfaces are concrete types

Interface definition

```
interface Foo(Type:$ T) for T {
  method (Ptr(Self): this) F();
}
```

Interface with associated type conforming to an interface

```
interface Bar(Type:$ T) for T {
  var Type:$ U;
  require Foo(U);
  method (Ptr(Self): this) G() -> Foo(U);
}
```

Type that implements an interface

```
struct Baz {
  impl Foo(Baz) {
    method (Ptr(Baz): this) F() { ... }
  }
}
impl Bar(Baz) { ... }

var Baz: x;
```

Function taking a value with type conforming to an interface

```
fn H1(Ptr(Foo(Type:$ T)): y) {
  y->F();
}
fn H2[Type:$ T](Ptr(Foo(T)): y) {
  y->F();
}
fn H3[Type:$ T, Foo(T)](Ptr(T): y) {
  (y as Ptr(Foo(T)))->F();
}
H1(&x); H2(&x); H3(&x);
```

Function taking a value with type conforming to two different interfaces

```
fn Ha[Type:$ T, Foo(T), Bar(T)](Ptr(T): y) {
  (y as Ptr(Foo(T)))->F();
  var Ptr(Bar(T)): z = y;
  z->G();
}
Ha(&x);
```

Function taking a value with a list of implementations for a single interface,
all compatible with a single representation type

```
???
```

### Facet type-types

Interface definition

```
interface Foo {
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
  impl Foo {
    method (Ptr(Baz): this) F() { ... }
  }
}
impl Bar for Baz { ... }

var Baz: x;
```

Function taking a value with type conforming to an interface

```
fn H1(Ptr(Foo:$ T): y) {
  y->F();
}
fn H2[Foo:$ T](Ptr(T): y) {
  y->F();
}
fn H3[TypeImplements(Foo):$ T](Ptr(T): y) {
  (y as Ptr(T as Foo))->F();
}
H1(&x); H2(&x); H3(&x);
```

Function taking a value with type conforming to two different interfaces

```
fn Ha[TypeImplements(Foo, Bar):$ T](Ptr(T): y) {
  (y as Ptr(T as Foo))->F();
  var Ptr(T as Bar): z = y;
  z->G();
}
Ha(&x);
```

Function taking a value with a list of implementations for a single interface,
all compatible with a single representation type

```
fn IsGreater[Type:$ T](Ptr(T): a, Ptr(T): b,
                       List(CompatibleWith(Comparable, T)):$ compare)
    -> Bool { ... }
```

Interface semantically containing other interfaces

```
interface Inner {
  method (Self: this) K();
}
interface Outer {
  impl Inner;
}
struct S {
  impl Outer {
    impl Inner {
      method (S: this) K() { ... }
    }
  }
}
var S: y = ...;
(y as ((S as Outer) as Inner)).K();
```

TODO: Interface structurally consisting of other interfaces

### Combined interfaces

Interface definition

```
interface Foo {
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
  impl Foo {
    method (Ptr(Baz): this) F() { ... }
  }
}
extend Baz {
  impl Bar { ... }
}

var Baz: x;
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
  y->F();
  // Qualified syntax works even if there is a name conflict between
  // Foo and Bar.
  y->(Bar.G)();
}
Ha(&x);
```

Function taking a value with a list of implementations for a single interface,
all compatible with a single representation type

```
fn IsGreater[Type:$ T](Ptr(T): a, Ptr(T): b,
                       List(CompatibleWith(Comparable, T)):$ compare)
    -> Bool { ... }
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
  impl Outer1 {}
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

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
