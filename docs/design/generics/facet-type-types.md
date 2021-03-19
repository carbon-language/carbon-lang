# Carbon deep dive: interfaces as facet type-types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [What is this?](#what-is-this)
-   [Overview](#overview)
-   [Interfaces](#interfaces)
-   [Implementing interfaces](#implementing-interfaces)
    -   [Out-of-line impl arguments for parameterized types](#out-of-line-impl-arguments-for-parameterized-types)
    -   [Impl lookup](#impl-lookup)
-   [Generics](#generics)
-   [Model](#model)
-   [Adapting types](#adapting-types)
    -   [Example: Defining an impl for use by other types](#example-defining-an-impl-for-use-by-other-types)
-   [Associated types](#associated-types)
    -   [Constraints on associated types in interfaces](#constraints-on-associated-types-in-interfaces)
        -   [Model](#model-1)
        -   [External constraints by way of optional parameters](#external-constraints-by-way-of-optional-parameters)
    -   [Constraints that are hard to express](#constraints-that-are-hard-to-express)
-   [Parameterized interfaces [optional feature]](#parameterized-interfaces-optional-feature)
-   [Conditional conformance](#conditional-conformance)
-   [Templated impls for generic interfaces](#templated-impls-for-generic-interfaces)
    -   [Structural conformance](#structural-conformance)
    -   [Bridge for C++ templates](#bridge-for-c-templates)
        -   [Calling C++ template code from Carbon](#calling-c-template-code-from-carbon)
        -   [Moving a C++ template to Carbon](#moving-a-c-template-to-carbon)
    -   [Subtlety around templated interfaces](#subtlety-around-templated-interfaces)
    -   [Lookup resolution](#lookup-resolution)
-   [Composition of type-types](#composition-of-type-types)
    -   [Interface extension [optional feature]](#interface-extension-optional-feature)
        -   [Use case: overload resolution](#use-case-overload-resolution)
        -   [Covariant return type constraints](#covariant-return-type-constraints)
        -   [Model](#model-2)
    -   [Type implementing multiple interfaces](#type-implementing-multiple-interfaces)
        -   [Subsumption](#subsumption)
        -   [Model](#model-3)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other impls](#example-creating-an-impl-out-of-other-impls)
    -   [Other type constraints](#other-type-constraints)
        -   [Recommendation: interface adapter](#recommendation-interface-adapter)
        -   [Rejected alternative: `ForSome(F)`](#rejected-alternative-forsomef)
    -   [Sized types and type-types](#sized-types-and-type-types)
        -   [Model](#model-4)
-   [Dynamic types](#dynamic-types)
    -   [Runtime type parameters](#runtime-type-parameters)
    -   [Runtime type fields](#runtime-type-fields)
        -   [Dynamic pointer type](#dynamic-pointer-type)
            -   [Model](#model-5)
        -   [Deref](#deref)
        -   [Boxed](#boxed)
        -   [DynBoxed](#dynboxed)
        -   [MaybeBoxed](#maybeboxed)
-   [Implicit interface arguments [rejected optional feature]](#implicit-interface-arguments-rejected-optional-feature)
-   [Interface nesting/containment [optional feature]](#interface-nestingcontainment-optional-feature)
-   [Index of examples](#index-of-examples)
-   [Notes](#notes)
-   [Broken links footnote](#broken-links-footnote)

<!-- tocstop -->

## What is this?

This document is a **deep dive**: a detailed exploration of one possible model
for part of the Carbon language, and its consequences. It is not intended to
present any questions or choices, but instead to flesh out one set of possible
design details for a design option so as to make it easier to conceptualize and
analyze.

This document presents a hypothetical world: details discussed here, while
presented as facts, are all only elements of that world and do not constitute
decisions about Carbon.

## Overview

Imagine we want to write a function parameterized by a type argument. Maybe our
function is `PrintToStdout` and let's say we want to operate on values that have
a type for which we have an implementation of the `ConvertibleToString`
interface. The `ConvertibleToString` interface has a `ToString` method returning
a string. To do this, we give the `PrintToStdout` function two parameters: one
is the value to print, let's call that `val`, the other is the type of that
value, let's call that `T`. The type of `val` is `T`, what is the type of `T`?
Well, since we want to let `T` be any type implementing the
`ConvertibleToString` interface, we express that in the "interfaces are
type-types" model by saying the type of `T` is `ConvertibleToString`.

Since we can figure out `T` from the type of `val`, we don't need the caller to
pass in `T` explicitly, it can be an
[implicit argument](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#implicit-argument)
(also see
[implicit argument](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/overview.md#implicit-arguments)
in the Generics V2 doc). Basically, the user passes in a value for `val`, and
the type of `val` determines `T`. `T` still gets passed into the function
though, and it plays an important role -- it defines the implementation of the
interface. We can think of the interface as defining a struct type whose members
are function pointers, and an implementation of an interface as a value of that
struct with actual function pointer values. So an implementation is a table of
function pointers (one per function defined in the interface) that gets passed
into a function as the type argument. For more on this, see
[the model section](#model) below.

In addition to function pointer members, interfaces can include any constants
that belong to a type. For example, the
[type's size](#sized-types-and-type-types) (represented by an integer constant
member of the type) is an optional member of an interface and its
implementation. There are a few cases why we would include another interface
implementation as a member:

-   [associated types](#associated-types)
-   [type parameters](#parameterized-interfaces-optional-feature)
-   [interface nesting/containment](#interface-nestingcontainment-optional-feature)
    (unlike the others, these define types with the same data representation as
    the type implementing the interface)

The function can decide whether that type argument is passed in
[statically](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#static-dispatch-witness-table)
(basically generating a separate function body for every different type passed
in) by using the "generic argument" syntax (`:$`, see
[the generics section](#generics) below) or
[dynamically](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#dynamic-dispatch-witness-table)
using the regular argument syntax (just a colon, `:`, see
[the runtime type parameters section](#runtime-type-parameters) below). Either
way, the interface contains enough information to
[type/definition check](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#complete-definition-checking)
the function body -- you can only call functions defined in the interface in the
function body. Contrast this with making the type a template argument, where you
could just use `Type` instead of an interface and it will work as long as the
function is only called with types that allow the definition of the function to
compile. You are still allowed to declare templated type arguments as having an
interface type, and this will add a requirement that the type satisfy the
interface independent of whether that is needed to compile the function body,
but it is strictly optional (you might do this to get clearer error messages,
document expectations, or express that a type has certain semantics beyond what
is captured in its member function names and signatures).

The last piece of the puzzle is how the caller of the function can produce a
value with the right type. Let's say the user has a value of type `Widget`, and
of course widgets have all sorts of functionality. If we want a `Widget` to be
printed using the `PrintToStdout` function, it needs to implement the
`ConvertibleToString` interface. Note that we _don't_ say that `Widget` is of
type `ConvertibleToString` but instead that it has a "facet type". This means
there is another type, called `Widget as ConvertibleToString`, with the
following properties:

-   `Widget as ConvertibleToString` has the same _data representation_ as
    `Widget`.
-   `Widget as ConvertibleToString` is an implementation of the interface
    `ConvertibleToString`. The functions of `Widget as ConvertibleToString` are
    just implementations of the names and signatures defined in the
    `ConvertibleToString` interface, like `ToString`, and not the functions
    defined on `Widget` values.
-   Carbon will implicitly cast values from type `Widget` to type
    `Widget as ConvertibleToString` when calling a function that can only accept
    types of type `ConvertibleToString`.

We define these facet types (alternatively interface implementations) either
with the type, with the interface, or somewhere else where Carbon can be
guaranteed to see when needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

If `Widget` doesn't implement an interface or we would like to use a different
implementation of that interface, we can define another type that also has the
same data representation as `Widget` that has whatever different interface
implementations we want. However, Carbon won't implicitly cast to that other
type, the user will have to explicitly cast to that type in order to select
those alternate implementations. For more on this, see
[the adapting type section](#adapting-types) below.

## Interfaces

An interface defines an API that a given type can implement. For example, an
interface capturing a vector API might have two methods:

```
interface Vector {
  // Here "Self" means "the type implementing this interface"
  method (Self: a) Add(Self: b) -> Self;
  method (Self: a) Scale(Double: v) -> Self;
}
```

An interface defines a type-type, that is a type whose values are types. The
values of an interface are specifically
[facet types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#invoking-interface-methods),
by which we mean types that are declared as specifically implementing
**exactly** this interface, and which provide definitions for all the functions
(and other members) declared in the interface.

## Implementing interfaces

Given a type, it can define an "impl" that defines how that interface is
implemented for that type.

```
struct Point {
  var Double: x;
  var Double: y;
  impl Vector {
    // In this scope, "Self" is an alias for "Point".
    method (Self: a) Add(Self: b) -> Self {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    }
    method (Self: a) Scale(Double: v) -> Self {
      return Point(.x = a.x * v, .y = a.y * v);
    }
  }
}
```

Impls may also be defined out of line:

```
struct Point {
  var Double: x;
  var Double: y;
}
impl Vector for Point {
  // Again, "Self" is an alias for "Point".
  method (Self: a) Add(Self: b) -> Self {
    return Point(.x = a.x + b.x, .y = a.y + b.y);
  }
  method (Self: a) Scale(Double: v) -> Self {
    return Point(.x = a.x * v, .y = a.y * v);
  }
}
```

To address concerns re:
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
we should allow out-of-line impl definitions either in the library that defines
the interface (`Vector`) or in the library that defines the type (`Point`).

In either case, the impl definition defines a
[facet type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#invoking-interface-methods):
`Point as Vector`. While the API of `Point` includes the two fields `x` and `y`,
the API of `Point as Vector` _only_ has the `Add` and `Scale` methods of the
`Vector` interface. The facet type `Point as Vector` is
[compatible](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#compatible-types)
with `Point`, so we allow you to cast between the two implicitly:

```
var Point: a = (.x = 1.0, .y = 2.0);
// Cast from Point implicitly
var Point as Vector: b = a;
// Cast from (Point as Vector)
var Point: c = b.Scale(1.5);
// Will also implicitly cast when calling functions:
fn F(Point as Vector: d, Point: e) { ... }
F(a, b);
```

or explicitly:

```
var Point as Vector: z = (a as (Point as Vector)).Scale(3.0);
var Point: w = z as Point;
```

**Note:** In general the above is written assuming that casts are written
"`a as T`" where `a` is a value and `T` is the type to cast to. When we write
`Point as Vector`, the value `Point` is a type, and `Vector` is a type of a
type, or a "type-type".

**Note:** If `Point` defines a method required by the interface, say `Scale`,
with the correct signature, then defining `Scale` in the impl definition becomes
optional, defaulting to the definition in `Point`. `Scale` is allowed to have
its own definition (which can have completely different semantics) in the
`Vector` impl definition since `Point` and `Point as Vector` are different
types.

**Note:** A type may implement any number of different interfaces, but may
provide at most one implementation of any single interface. This makes the act
of selecting an implementation of an interface for a type unambiguous throughout
the whole program, so for example `Point as Vector` is clearly defined.

### Out-of-line impl arguments for parameterized types

What if our `Point` type was parameterized? If the `impl` is defined inline, the
impl definition is already inside the scope with the parameterization, so there
is no difficulty in using the type variables, as in:

```
struct PointT(Type:$$ T) {
  var T: x;
  var T: y;
  impl Vector {
    method (Self: a) Add(Self: b) -> Self { /* Can use T here. */  }
    method (Self: a) Scale(Double: v) -> Self { ... }
  }
}
```

When defining the impl out-of-line, we need some way of providing those
parameters. For example,

```
struct PointT(Type:$$ T) {
  var T: x;
  var T: y;
}
impl[Type:$$ T] Vector for PointT(T) {
  method (Self: a) Add(Self: b) -> Self { ... }
  method (Self: a) Scale(Double: v) -> Self { ... }
}
```

We put these arguments syntactically early so those names could also be used
with [parameterized interfaces](#parameterized-interfaces-optional-feature).
This syntax is also convenient for
[conditional conformance](#conditional-conformance) and
[templated impls](#templated-impls-for-generic-interfaces), below. We use square
brackets here as is done for implicit arguments to functions, since there is no
explicit caller providing these arguments.

### Impl lookup

Let's say you have some interface `I(T, U(V))` being implemented for some type
`A(B(C(D), E))`. That impl must be defined in the same library that defines the
interface or one of the names needed by the type. That is, the impl must be
defined with (exactly) one of `I`, `A`, `B`, `C`, `D`, or `E`. Note that you
can't define the impl with `T`, `U`, `V`, or any other library unless it also
defines one of `I`, `A`, ..., or `E`. We further require anything looking up
this impl to import the _definitions_ of all of those names. Seeing a forward
declaration of these names is insufficient, since you can presumably see forward
declarations without seeing an impl with the definition. This accomplishes a few
goals:

-   The compiler can check that there is only one definition of any impl that is
    actually used, avoiding
    [One Definition Rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule)
    problems.
-   Every attempt to use an impl will see the exact same impl, making the
    interpretation and semantics of code consistent no matter its context, in
    accordance with the
    [Refactoring principle](https://github.com/josh11b/carbon-lang/blob/principle-refactoring/docs/project/principles/principle-refactoring.md).
-   Allowing the impl to be defined with either the interface or the type
    addresses the
    [expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

Note that interface parameters are treated differently because they can be
inferred as part of calling a function call, as described in
[this appendix](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/appendix-interface-param-impl.md).
We could allow implementations to be defined with arguments that can't be
inferred, if we were willing to use a more complicated rule.

## Generics

Now let us write a function that can accept values of any type that has
implemented the `Vector` interface:

```
fn AddAndScale[Vector:$ T](T: a, T: b, Double: s) -> T {
  return a.Add(b).Scale(s);
}
var Point: v = AddAndScale(a, w, 2.5);
```

Here `T` is a type whose type is `Vector`. The `:$` syntax means that `T` is a
_[generic argument](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#generic-versus-template-parameters)_,
that is it must be known to the caller but we will only use the information
present in the signature of the function to typecheck the body of
`AddAndScale`'s definition. In this case, we know that any value of type `T` has
an `Add` and a `Scale` method. When we call `AddAndScale`, it is passed values
with type `Point`. `Point` is not an implementation of `Vector`, but `Point`
values may be implicitly converted to `Point as Vector` values which do
implement `Vector`, so `T` is set to `Point as Vector`. This is consistent with
calling `Add` and `Scale` on values `a` and `b` in the body of the function,
since `Point as Vector` does have those methods. That is if we make the function
non-generic, it would still type check using `Point as Vector`:

```
fn AddAndScaleForPointAsVector(Point as Vector: a, Point as Vector: b, Double: s)
      -> Point as Vector {
  return a.Add(b).Scale(s);
}
// May still be called with Point arguments, due to implicit casts.
// Similarly the return value can be implicitly cast to a Point.
var Point: v2 = AddAndScaleForPointAsVector(a, w, 2.5);
```

even though it would not with `Point`:

```
fn AddAndScaleForPoint(Point: a, Point: b, Double: s) -> Point {
  // Error: Point does not have an "Add" method.
  return a.Add(b).Scale(s);
}
```

If we had another type unrelated to `Point` implementing the `Vector` interface,
it too could be passed to `AddAndScale`.

## Model

The underlying model here is
[interfaces are type-types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interfaces-are-type-types),
in particular
[facet type-types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#facet-type-types):

-   [Interfaces](#interfaces) are types of
    [witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#witness-tables-for-example-swift-and-carbon-generics)s
-   Facet types (defined by [Impls](#implementing-interfaces)) are
    [witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#witness-tables-for-example-swift-and-carbon-generics)
    values
-   The compiler rewrites functions with an implicit type argument
    (`fn Foo[InterfaceName:$ T](...)`) to have an actual argument with type
    determined by the interface, and supplied at the callsite using a value
    determined by the impl.

Context:
[Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types -->

For the example above, [the Vector interface](#interfaces) could be thought of
defining a witness table type like:

```
struct Vector {
  var Type:$ Self;  // Self is the representation type.
  // Self is a member, not a parameter, so there is a single witness table type
  // across different `Self` types.
  var fn(Self: a, Self: b) -> Self : Add;
  var fn(Self: a, Double: v) -> Self : Scale;
}
```

The [impl of Vector for Point](#implementing-interfaces) would be a value of
this type:

```
var Vector : VectorForPoint = (
    .Self = Point,
    .Add = fn(Point: a, Point: b) -> Point {
      return Point(.x = a.x + b.x, .y = a.y + b.y);
    },
    .Scale = fn(Point: a, Double: v) -> Point {
      return Point(.x = a.x * v, .y = a.y * v);
    },
);
```

Finally we can define a generic function and call it, like
<code>[AddAndScale from the "Generics" section above](#generics)</code> by
making the witness table a regular argument to the function:

```
fn AddAndScale[Type:$ T](T: a, T: b, Double: s, Ptr(Vector(T)):$ impl) -> T {
  return impl->Scale(impl->Add(a, b), s);
}
// Point implements Vector.
var Point: v = AddAndScale(a, w, 2.5, &VectorForPoint);
```

The rule is that generic arguments (declared using `:$`) are passed at compile
time, so the actual value of the `impl` argument here can be used to generate
the code for `AddAndScale`. So `AddAndScale` is using a
[static-dispatch witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#static-dispatch-witness-table).

## Adapting types

We also provide a way to create new types
[compatible with](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#compatible-types)
existing types with different APIs, in particular with different interface
implementations, by
[adapting](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#compatible-types)
them:

```
interface A { method (Self: this) F(); }
interface B { method (Self: this) G(); }
struct C {
  impl A { method (Self: this) F() { Print("CA"); } }
}
adaptor D for C {
  impl B { method (Self: this) G() { Print("DB"); } }
}
adaptor E for C {
  impl A { method (Self: this) F() { Print("EA"); } }
}
adaptor F for C {
  impl A = E as A;  // Possibly we'd allow "impl A = E;" here.
  impl B = D as B;
}
```

This allows us to provide implementations of new interfaces (as in `D`), provide
different implementations of the same interface (as in `E`), or mix and match
implementations from other compatible types (as in `F`). The rules are:

-   You may only add APIs, not change the representation of the type, unlike
    extending a type where you may add fields.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `C`, `D`, `E`, and `F` end up compatible
    with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit casting between these
    types (unlike between a type and one of its facet types / impls).

### Example: Defining an impl for use by other types

Let's say we want to provide a possible implementation of an interface for use
by types for which that implementation would be appropriate. We can do that by
defining an adaptor implementing the interface that is parameterized on the type
it is adapting. That impl may then be pulled in using the `"impl ... = ...;"`
syntax.

```
interface Comparable {
  fn operator<(Self: this, Self: that) -> Bool;
  ... // And also for >, <=, etc.
}
adaptor ComparableFromDifferenceFn(Type:$ T, fn(T, T)->Int:$ Difference) for T {
  impl Comparable {
    fn operator<(Self: this, Self: that) -> Bool {
      return Difference(this, that) < 0;
    }
    ... // And also for >, <=, etc.
  }
}
struct MyType {
  var Int: x;
  fn Difference(Self: this, Self: that) { return that.x - this.x; }
  impl Comparable = ComparableFromDifferenceFn(MyType, Difference) as Comparable;
}
```

## Associated types

For context, see
["Interface type parameters versus associated types" in the Carbon: Generics Terminology doc](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-versus-associated-types).
In some cases, we will want the signatures of methods to vary from
implementation to implementation. We already have one example of this: the
`Self` type discussed [above in the "Interfaces" section](#interfaces). For
other cases, we can say that the interface declares that each implementation
will provide a type under a specific name. For example:

```
interface Stack {
  var Type:$ ElementType;
  method (Ptr(Self): this) Push(ElementType: value);
  method (Ptr(Self): this) Pop() -> ElementType;
  method (Ptr(Self): this) IsEmpty() -> Bool;
}
```

Here we have an interface called `Stack` which defines two methods, `Push` and
`Pop`. The signatures of those two methods declared as accepting or returning
values with the type `ElementType`, which any implementer of `Stack` must also
define. For example, maybe `DynamicArray` implements `Stack`:

```
struct DynamicArray(Type:$ T) {
  // DynamicArray methods
  // Could use either `Self` or `DynamicArray(T)` here.
  method (Ptr(Self): this) PushBack(T: value);
  method (Ptr(Self): this) PopBack() -> T;
  method (Ptr(Self): this) IsEmpty() -> Bool;

  impl Stack {
    var Type:$ ElementType = T;
    // `Self` and `DynamicArray(T)` are still equivalent here.
    method (Ptr(Self): this) Push(ElementType: value) {
      this->PushBack(value);
    }
    method (Ptr(Self): this) Pop() -> ElementType {
      return this->PopBack();
    }
    // Use default version of IsEmpty() from DynamicArray.
  }
}
```

Now we can write a generic function that operates on anything implementing the
`Stack` interface, for example:

```
fn PeekAtTopOfStack[Stack:$ StackType](Ptr(StackType): s) -> StackType.ElementType {
  var StackType.ElementType: top = s->Pop();
  s->Push(top);
  return top;
}

var DynamicArray(Int): my_array = (1, 2, 3);
// PeekAtTopOfStack's StackType is set to (DynamicArray(Int) as Stack).
// StackType.ElementType becomes Int.
Assert(PeekAtTopOfStack(my_array) == 3);
```

**Aside:** In general, any field declared as "generic" (using the `:$` syntax),
will only have compile-time and not runtime storage associated with it.

### Constraints on associated types in interfaces

Now note that inside `PeekAtTopOfStack` we don't know anything about
`StackType.ElementType`, so we can't perform any operations on values of that
type, other than pass them to `Stack` methods. We can define an interface that
has an associated type constrained to satisfy an interface (or any
[other type-type](#adapting-types)). For example, we might say interface
`Container` has a `Begin` method returning values with type satisfying the
`Iterator` interface:

```
interface Iterator {
  method (Ptr(Self): this) Advance();
  ...
}
interface Container {
  var Iterator:$ IteratorType;
  method (Ptr(Self): this) Begin() -> IteratorType;
  ...
}
```

With this additional information, a function can now call `Iterator` methods on
the return value of `Begin`:

```
fn OneAfterBegin[Container:$ T](Ptr(T): c) -> T.IteratorType {
  var T.IteratorType: iter = c->Begin();
  iter.Advance();
  return iter;
}
```

#### Model

The associated type is modeled by a witness table field in the interface.

```
struct Iterator(Type:$ Self) {
  var fn(Ptr(Self): this): Advance;
  ...
}
struct Container(Type:$ Self) {
  var Type:$ IteratorType;  // Representation type for the iterator.
  // Witness that IteratorType implements Iterator.
  var Ptr(Iterator(IteratorType)): iterator_impl;
  method (Ptr(Self): this) Begin() -> IteratorType;
  ...
}
```

#### External constraints by way of optional parameters

Users of an interface may also want to introduce constraints on an associated
type, in addition to constraints in the definition of the interface. To support
this, we automatically make every associated type an optional named parameter to
the interface. For example, we could make a the `ElementType` of an `Iterator`
interface equal to the `ElementType` of a `Container` interface as follows:

```
interface Iterator {
  var Type:$ ElementType;
  ...
}
interface Container {
  var Type:$ ElementType;
  var Iterator(.ElementType = ElementType):$ IteratorType;
  ...
}
```

Functions accepting a generic type might also want to constrain an associated
type. For example, we might want to have a function only accept stacks
containing integers:

```
fn SumIntStack[Stack(.ElementType = Int):$ T](Ptr(T): s) -> Int {
  var Int: sum = 0;
  while (!s->IsEmpty()) {
    sum += s->Pop();
  }
  return sum;
}
```

Or you might constrain the element type to satisfy an interface (`Comparable` in
this example) without saying exactly what type it is:

```
fn SortContainer[TypeImplements(Comparable):$ ElementType,
                 Container(.ElementType = ElementType):$ ContainerType]
    (Ptr(ContainerType): container_to_sort);
```

These different types of constraints can be combined. For example, this example
expresses a constraint that two associated types are equal and satisfy an
interface:

```
// CT1.ElementType == CT2.ElementType
// CT1.ElementType implements HasEquality
fn EqualContainers[TypeImplements(HasEquality):$ ET,
                   Container(.ElementType = ET):$ CT1,
                   Container(.ElementType = ET):$ CT2]
    (Ptr(CT1): c1, Ptr(CT2): c2) -> Bool;
```

This approach has a few advantages:

-   There is one mechanism for constraining an interface: passing in arguments.
    We use this both for
    [interface parameters](#parameterized-interfaces-optional-feature)
    (required, typically positional parameters) and associated types (optional,
    named parameters).
-   You can express a variety of constraints, including that two things must be
    the same, something must take on a specific value, or something must satisfy
    an interface.
-   Deciding if two types must always be the same can be done just by
    normalizing their expressions by substituting any parameters. For example,
    `Container.IteratorType.ElementType` normalizes to `Container.ElementType`.
    In the case of the `SortContainer` function, there is an additional
    constraint making `ContainerType.IteratorType.ElementType` and
    `ContainerType.ElementType` both equal to `ElementType`.
-   Many constrained interfaces are naturally represented as type-types, which
    is useful for constructs like [`DynPtr`](#dynamic-pointer-type) or
    [`DynBoxed`](#dynboxed) that are parameterized by type-types.

**Rejected alternative:** Other languages use `requires` clauses to expressed
constraints, as discussed in
[this appendix](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/appendix-requires-constraints.md).

### Constraints that are hard to express

**Concern:** It is difficult to express some kinds of constraints in this
framework: mathematical constraints on values (for example, "`NTuple` where `N`
is at least 2"), and inequality type constraints (for example "type is not
`Bool`").

You might need an inequality type constraint, for example, to control overload
resolution:

```
fn F[Type:$ T](T: x) -> T { return x; }
fn F(Bool: x) -> String {
  if (x) return "True"; else return "False";
}

fn G[Type:$ T](T: x) -> T {
  // We need a T != Bool constraint for this to type check.
  return F(x);
}
```

Another use case for inequality type constraints would be to say something like
"define `ComparableTo(T1)` for `T2` if `ComparableTo(T2)` is defined for `T1`
and `T1 != T2`".

## Parameterized interfaces [optional feature]

(This feature is optional: we may not want this extra complexity. I do recommend
implementing this feature, for the reasons discussed below.)

Reasons:

-   In Carbon we are consistently allowing constructs to be parameterized.
-   Some things are more naturally represented by required, positional
    parameters (as opposed to associated types, which are optional and named).
-   Parameterized interfaces more naturally support cases where one type
    naturally implements an interface multiple times with different parameters.
    For example, a type might be comparable with multiple other types.

Some type constraints would be more conveniently expressed by moving from
[associated types](#associated-types) to
[type parameters for interfaces](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-versus-associated-types).
The syntax for type parameters is that we allow a parameter list after the name
of the interface:

```
interface Stack(Type:$ ElementType) {
  method (Ptr(Self): this) Push(ElementType: value);
  method (Ptr(Self): this) Pop() -> ElementType;
  method (Ptr(Self): this) IsEmpty() -> Bool;
}
struct DynamicArray(Type:$ T) {
  ...
  // References to a parameterized interface must be followed by an argument list
  // with specific values for every parameter.
  impl Stack(T) { ... }
}
// Equivalent out-of-line definition:
impl[Type:$ T] Stack(T) for DynamicArray(T) { ... }
```

It is a little more convenient to express type constraints on type parameters
than associated types, since you don't need to specify the name:

```
// Stack.ElementType constrained to be Int.
fn SumIntStack[Stack(Int):$ T](Ptr(T): s) -> Int { ... }

interface Iterator(Type:$ ElementType) { ... }
interface Container(Type:$ ElementType) {
  // IteratorType constrained to implement the Iterator interface,
  // with matching ElementType.
  var Iterator(ElementType):$ IteratorType;
  ...
}

// ElementType of the two containers constrained to match and
// implement HasEquality interface.
fn EqualContainers[HasEquality:$ ElementType,
                   Container(ElementType):$ CT1,
                   Container(ElementType):$ CT2]
    (Ptr(CT1): c1, Ptr(CT2): c2) -> Bool { ... }
```

Note that the way to express constraints on type parameters is consistent with
treating associated types as optional parameters.

But it is more awkward in the unconstrained case, since you still need to pass
something in that position. That is both more ceremony for something you didn't
care about, and creates an issue of how that parameter is determined.

**Proposal:** A type will be by default only allowed one implementation of an
interface, not one per interface & type parameter combination.

The advantage of this approach is that it makes inferring the type parameters
unambiguous. This allows us to use a parameterized interface in cases where we
don't know the type parameter. For example, to write `PeekAtTopOfStack` from the
[associated types section](#associated-types) with the parameterized version of
`Stack`, we need to infer the `ElementType`:

```
fn PeekAtTopOfStack[Type:$ ElementType, Stack(ElementType):$ StackType]
    (Ptr(StackType): s) -> ElementType { ... }
```

The alternative of one implementation per interface & type parameter combination
is perhaps more natural. It seems useful for something like a `ComparableTo(T)`
interface, where a type might be comparable with multiple other types. It does
have a problem where you need to be certain that every impl of an interface for
a parameterized type can be distinguished:

```
interface Map(Type:$ FromType, Type:$ ToType) {
  method (Ptr(Self): this) Map(FromType: needle) -> Optional(ToType);
}
struct Bijection(Type:$ FromType, Type:$ ToType) {
  impl Map(FromType, ToType) { ... }
  impl Map(ToType, FromType) { ... }
}
// Error: Bijection has two impls of interface Dictionary(String, String)
var Bijection(String, String): oops = ...;
```

In this case, it would be better to have an adapting type to contain the impl
for the reverse map lookup:

```
struct Bijection(Type:$ FromType, Type:$ ToType) {
  impl Map(FromType, ToType) { ... }
}
adaptor ReverseLookup(Type:$ FromType, Type:$ ToType)
    for Bijection(FromType, ToType) {
  impl Map(ToType, FromType) { ... }
}
```

This would be the preferred approach to use instead of multiple impls of the
same interface.

**Proposal:** You can opt in to allowing multiple impls for a type by using
templated type parameters to an interface. Templated type parameters to
interfaces generally won't be inferred / deduced (at least not in a context
where only one answer is allowed).

This will be the approach used for `ComparableTo(T)`, `ConstructibleFrom(...)`,
or other operators that might have multiple overloads. Example:

```
interface EqualityComparableTo(Type:$$ T) {  // Note: $$ instead of $
  fn operator==(Self: this, T: that) -> Bool;
  ...
}
struct Complex {
  var Float64: real;
  var Float64: imag;
  // Can implement this interface more than once as long as it has different
  // arguments.
  impl EqualityComparableTo(Complex) { ... }
  impl EqualityComparableTo(Float64) { ... }
}
```

**Question:** Should we use something other than the generic/template
distinction to specify whether different values to the parameter create
different interfaces for purposes of inference and implementation?

## Conditional conformance

The problem we are trying to solve here is expressing that we have an impl of
some interface for some type, but only if some additional type restrictions are
met. To do this, we leverage
[impl arguments](#out-of-line-impl-arguments-for-parameterized-types):

-   We can provide the same impl argument in two places to constrain them to be
    the same.
-   We can declare the impl argument with a more-restrictive type, to for
    example say this impl can only be used if that type satisfies an interface.

**Example:** [Interface constraint] Here we implement the `Printable` interface
for arrays of `N` elements of `Printable` type `T`, generically for `N`.

```
interface Printable {
  method (Ptr(Self): this) Print() -> String;
}
struct FixedArray(Type:$ T, Int:$ N) { ... }

// By saying "Printable:$ T" instead of "Type:$ T" here, we constrain
// T to be Printable for this impl.
// Note: this should probably be "TypeImplements(Printable):$ T", but
// "TypeImplements" is described later in this document.
impl[Printable:$ T, Int:$ N] Printable for FixedArray(T, N) {
  method (Ptr(Self): this) Print() -> String {
    var Bool: first = False;
    var String: ret = "";
    for (auto: a) in *this {
      if (!first) {
        ret += ", ";
      }
      ret += a.Print();
    }
    return ret;
  }
}
```

**Example:** [Same-type constraint] We implement interface `Foo(T)` for
`Pair(T, U)` when `T` and `U` are the same.

```
interface Foo(Type:$ T) { ... }
struct Pair(Type:$ T, Type:$ U) { ... }
impl[Type:$ T] Foo(T) for Pair(T, T) { ... }
```

**Proposal:** [Other boolean condition constraints] Just like we support
conditions when pattern matching (for example in overload resolution), we should
also allow them when defining an impl:

```
impl[Type:$$ T] if (sizeof(T) <= 16) Foo for T { ... }
```

**Concern:** The conditional conformance feature makes the question "is this
interface implemented for this type" undecidable in general.
[This feature in Rust has been shown to allow implementing a Turing machine](https://sdleffler.github.io/RustTypeSystemTuringComplete/).
This means we will likely need some heuristic like a limit on how many steps of
recursion are allowed.

**Rejected alternative:** We could also have a syntax for defining these impls
inline in the struct definition, but I haven't found a satisfactory solution
here. For example, it is hard to express the "two types are actually the same"
constraint from the previous example. It also causes issues where you either
have to introduce a new name for the constrained type or have the same name mean
different things in the inner scope with the impl definition versus the
containing struct scope. This was discussed in
[Carbon meeting Nov 27, 2019 on Generics & Interfaces (TODO)](#broken-links-footnote)<!-- T:Carbon meeting Nov 27, 2019 on Generics & Interfaces --><!-- A:#heading=h.gebr4cdi0y8o -->.

## Templated impls for generic interfaces

Some things going on here:

-   Our syntax for out-of-line impls already allows you to have a templated type
    parameter. This can be used to provide a general impl that depends on
    templated access to the type, even when the interface itself is defined
    generically.
-   We very likely will want to restrict the impl in some ways.
    -   Easy case: An impl for a family of parameterized types.
    -   Trickier is "structural conformance": we might want to say "here is an
        impl for interface `Foo` for any class implementing a method `Bar`".
        TODO: real use case.

### Structural conformance

**Question:** How do you say: "restrict this impl to types that have a member
function with a specific name & signature"?

An important use case is to restrict templated definitions to an appropriate set
of types.

**Decision:** We don't want to support the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
of C++ because it does not let the user clearly express the intent of which
substitution failures are meant to be constraints and which are bugs.
Furthermore, the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
leads to problems where the constraints can change accidentally as part of
modifications to the body that were not intended to affect the constraints at
all. As such, constraints should only be in the impl signature rather than be
determined by anything in the body.

**Decision:** We don't want anything like `LegalExpression(...)` for turning
substitution success/failure into True/False at this time, since we believe that
it introduces a lot of complexity, and we would rather lean on conforming to an
interface or the reflection APIs. However, we feel less strongly about this
position than the previous position and we may revisit (say because of needing a
bridge for C++ users). One nice property of the `LegalExpression(...)` paradigm
for expressing a constraint is that it would be easy for the constraint to
mirror code from the body of the function.

**Additional concern:** We want to be able to express "method has a signature"
in terms of the types involved, without necessarily any legal example values for
those types. For example, we want to be able to express that "`T` is
constructible from `U` if it has a `operator create` method that takes a `U`
value", without any way to write down an particular value for `U` in general:

```
interface ConstructibleFrom(...:$$ Args) { ... }
impl[Type:$$ T, Type:$$ U] if (LegalExpression(T.operator create(???)))
    ConstructibleFrom(U) for T { ... }
```

This is a problem for the `LegalExpression(...)` model, another reason to avoid
it.

**Possible answer:** likely we will write constraints using the reflection API,
though it isn't written at this time. The advantage of this approach is that we
likely will want a reflection API anyway for use in metaprogramming, so reusing
that instead of introducing another mechanism reduces complexity.

TODO: Go interface structural matching -- do we really want that? Could really
help with decoupling dependencies, breaking cycles, cleaning up layering.
Example: two libraries can be combined without knowing about each other.

**Possible answer:** Another approach would be define a function that returns a
type-type that matches types satisfying a structural constraint. This structural
constraint might be represented by a list of name, signature pairs encoded as
keyword arguments, as in:

> `HasFunction(.Name1 = FnType1, .Name2 = FnType2)` defines a type-type whose
> values are types with at least a method named `Name1` with signature matching
> the function type `FnType1` and a method name `Name2` with signature matching
> the function type `FnType2`.

Example:

```
fn CallsFooAndBar[HasFunction(.Foo = fn(Int:_)->String,
                              .Bar = fn(String:_)->Bool):$$ T]
    (T: x, Int: y) -> Bool {
  return x.Bar(x.Foo(y));
}
```

One downside of this approach is that it does not provide a way to write down
signatures that involve `Self`. Supporting that may require something a bit more
cumbersome:

```
HasMethod(fn(Type:$$ Self)->(.Foo = fn(Self:_, Int:_)->String,
                             .Bar = fn(Self:_, String:_)->Bool))
```

**Possible answer:** A last possibility: anonymous interfaces would match
structurally. Just as types with names use nominative typing and types without
names are compared structurally, we could say that an interface defined without
a name is automatically implemented for any type that structurally matches the
API defined in the interface.

```
interface {
  fn Foo(Int: _) -> String;
  fn Bar(String: _) -> Bool;
}
```

This has the benefit that signatures involving `Self` are much more natural to
write using the machinery and syntax already associated with defining
interfaces.

```
interface {
  fn Foo(Self: _, Int: _) -> String;
  fn Bar(Self: _, String: _) -> Bool;
}
```

To give a name to this type-type while staying with structural conformance, use
the `alias` facility:

```
alias HasFooAndBar = interface {
  fn Foo(Int: _) -> String;
  fn Bar(String: _) -> Bool;
}
fn CallsFooAndBar[HasFooAndBar:$$ T](T: x, Int: y) -> Bool { ... }
```

Unfortunately, the difference between `alias Foo = interface { ... }` and
`interface Foo { ... }` is pretty subtle but important. It might be better to
instead have a keyword to draw attention to this difference, like:

```
structural interface HasFooAndBar {
  fn Foo(Int: _) -> String;
  fn Bar(String: _) -> Bool;
}
```

### Bridge for C++ templates

#### Calling C++ template code from Carbon

Let's say we want to call some templated C++ code from generic Carbon code.

```
// In C++
template<class T>
struct S {
  void F(T* t);
};
```

We first define the common API for the template:

```
// In Carbon
interface SInterface(Type:$ T) {
  method (Ptr(Self): this) F(Ptr(T): t);
}
```

and once we implement that interface for the C++ type `S`:

```
// Note: T has to be a templated argument to be usable with the C++ template `S`.
// There is no problem passing a template argument `T` to the generic argument of
// `SInterface`.
impl[Type:$$ T] SInterface(T) for C++::S(T) {
  method (Ptr(Self): this) F(Ptr(T): t) { this->F(t); }
}
```

we can then call it from a generic Carbon function:

```
fn G[Type:$ T, SInterface(T):$ SType](Ptr(SType): s, Ptr(T): t) {
  s->F(t);
}
var C++::S(Int) : x;
var Int : y = 3;
G(&x, &y);  // C++::S(Int) implements SInterface(Int) by way of templated impl
```

#### Moving a C++ template to Carbon

Imagine we have a C++ templated type with (possibly templated) consumers in C++,
and we want to migrate that type to Carbon. For example, say we have a template
`C++::Foo` in C++, and are moving it to Carbon generic `Foo`. Let's say the
`C++::Foo` template takes optional parameters, `C++::std::optional<T>` for any
`T`, but of course the way template code is typically written is to make it work
with anything that has the `C++::std::optional<T>` API. When we move it to
generic `Foo` in Carbon, we need both the `T` argument, and a
[higher-ranked](#bookmark=id.kayz42hh0s7j) type parameter to represent the
optional type. Some C++ users will continue to use this type with C++'s
`std::optional<T>`, which by virtue of being a C++ template, can't take generic
arguments. We still can make a templated implementation of a generic interface
for it:

```
interface Optional(Type:$ T) { ... }
impl[Type:$$ T] Opt(T) for C++::std::optional(T);
```

### Subtlety around templated interfaces

One subtlety around templated interfaces is that they may have multiple
implementations for a single type. Templated impls could take these each of
these multiple implementations for one interface and manufacture an impl for
another interface, as in this example:

```
// Some interfaces with templated type parameters.
interface EqualityComparableTo(Type:$$ T) { ... }
// Types can implement templated interfaces more than once as long as the
// templated arguments differ.
struct Complex {
  var Float64: r;
  var Float64: i;
  impl EqualityComparableTo(Complex) { ... }
  impl EqualityComparableTo(Float64) { ... }
}
// Some other interface with a templated type parameter.
interface Foo(Type:$$ T) { ... }
// This provides an impl of Foo(T) for U if U is EqualityComparableTo(T).
// In the case of Complex, this provides two impls, one for T == Complex, and one for T == Float64.
impl[EqualityComparableTo(Type:$$ T):$ U] Foo(T) for U { ... }
```

One tricky part of this is that you may not have visibility into all the impls
of an interface for a type since they may be
[defined with one of the other types involved](#impl-lookup). Hopefully this
isn't a problem -- you will always be able to see the _relevant_ impls given the
types that have been imported / have visible definitions.

### Lookup resolution

**Rule:** Can have multiple impl definitions that match, as long as there is a
single best match. Best is defined using the "more specific" partial ordering:

-   Matching a descendant type or descendant interface is more specific and
    therefore a closer match than a parent type or interface.
-   Matching an exact type (`Foo` or `Foo(Bar)`) is more specific than a
    parameterized family of types (`Foo(T)` for any type `T`) is more specific
    than a generic type (`T` for any type `T`).
-   TODO: others?

TODO: Examples

**Implication:** Can't do impl lookup with generic arguments, even if you can
see a matching templated definition, since there may be a more-specific match
and we want to be assured that we always get the same result any time we do an
impl lookup.

TODO: Example

## Composition of type-types

So we have one way of defining a type-type: every interface is represented by a
type-type. In addition, Carbon defines `Type`, the type-type whose values
include every type. We now introduce ways of defining new type-types in terms of
other type-types.

### Interface extension [optional feature]

(I'm not sure that we need this extra complexity, but it is easy to define in
case we decide this feature is sufficiently useful.)

We can define a new interface as the extension of another interface, adding
additional API. Anything implementing the extended interface also implements the
base interface, but is required to give definitions/implementations for all the
functions defined in the base interface.

```
interface A {
  method (Self: this) F();
}
interface B extends A {
  method (Self: this) G();
}
struct S {
  impl B {
    method (Self: this) F() { ... }
    method (Self: this) G() { ... }
  }
}
var S: x;
(x as (S as B)).F();
(x as (S as B)).G();
(x as (S as A)).F();
```

Interface extension supports a form of subsumption:

```
fn TakesA[A: T](Ptr(T): a) { ... }
TakesA(&x);  // Okay: S implements B so it also implements A.

fn TakesB[B: T](Ptr(T): b) { ... }
struct SA {
  impl A { ... }
}
var SA: y;
TakesB(&y);  // Error: SA implements A but not B.
```

TODO: State the rules of covariance/contravariance for inheritance.

#### Use case: overload resolution

Implementing an extended interface is an example of a more specific match for
[lookup resolution](#lookup-resolution). For example, this could be used to
provide different implementations of an algorithm depending on the capabilities
of the iterator being passed in:

```
interface ForwardIterator(Type:$ T) {
  fn operator*(Self: this) -> Ref(T);
  fn operator++(Ptr(Self): this);
  fn operator==(Self: this, Self: that) -> Bool;
  fn operator!=(Self: this, Self: that) -> Bool;
}
interface BidirectionalIterator(Type:$ T) extends ForwardIterator(T) {
  fn operator--(Ptr(Self): this);
}
interface RandomAccessIterator(Type:$ T) extends BidirectionalIterator(T) {
  fn operator+(Self: this, Int: offset) -> Self;
  fn operator-(Self: this, Int: offset) -> Self;
  fn operator-(Self: this, Self: that) -> Int;
  fn operator+=(Ptr(Self): this, Int: offset);
  fn operator-=(Ptr(Self): this, Int: offset);
}
fn SearchInSortedList[Comparable:$ T, ForwardIterator(T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  // does linear search
}
// Will prefer the following overload when it matches since it is more specific.
fn SearchInSortedList[Comparable:$ T, RandomAccessIterator(T): IterT]
    (IterT: begin, IterT: end, T: needle) -> Bool {
  // does binary search
}
```

#### Covariant return type constraints

In addition to adding functions, the extended interface can have more
restrictive constraints on any associated types used only in covariant positions
in the interface, such as return values.

```
interface ForwardContainer(Type:$ T) {
  var ForwardIterator(T):$ IteratorType;
  method (Ptr(Self): this) Begin() -> IteratorType;
  method (Ptr(Self): this) End() -> IteratorType;
}
interface BidirectionalContainer(Type:$ T) extends ForwardContainer(T) {
  var BidirectionalIterator(T):$ IteratorType;
}
```

Any consumer of a `ForwardContainer(T)` can call all the `ForwardIterator(T)`
methods on values of `IteratorType`, and all of those methods will also be
present for any value of type `BidirectionalIterator(T)`.

#### Model

The extended interface just appends on to the end of the existing interface,
just like as is done
[with structs](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#question-extension--inheritance).
For the above example, this would be represented as:

```
struct A(Type:$ Self) {
  var fn(Self): F;
}
struct B(Type:$ Self) extends A(Self) {
  var fn(Self): G;
}
```

### Type implementing multiple interfaces

Let's define a type-type constructor called `TypeImplements`. Given a list
`(TT1, ..., TTn)` of type-types (typically interfaces), we define a new
type-type `TypeImplements(TT1, ..., TTn)` according to this rule:

> `TypeImplements(TT1, ..., TTn)` is a type whose values are types `T` such that
> `T as TT1`, ..., `T as TTn` are all legal expressions.

Note that the order of the arguments does not matter (so `TypeImplements(A, B)`
is the same as `TypeImplements(B, A)`).

This would be used as follows:

```
interface A { method (Self: this) F(); }
interface B { method (Self: this) G(); }
fn H[TypeImplements(A, B):$ T](T: x) {
  // Can't call any methods on "x" directly.
  (x as (T as A)).F();
  (x as (T as B)).G();
}
struct S {
  impl A { method (Self: this) F() { ... } }
  impl B { method (Self: this) G() { ... } }
}
var S: y = ...;
H(y); // H's T is set to S
```

Note that `TypeImplements(A)` is different from `A` when it comes to the
function body, even if the difference is invisible to callers of the function
(through the magic of implicit casts):

```
fn K1[A:$ T](T: x) {
  x.F();  // Legal
  // T has type A, so (T as A) == T, so (x as (T as A)) == (x as T) == x.
  // Since all the those casts are trivial, this is exactly equivalent to the
  // x.F() statement above:
  (x as (T as A)).F();
}
fn K2[TypeImplements(A):$ T](T: x) {
  x.F();  // Error: "x" doesn't have a method named "F".
  // In this case T does not have type A, so these casts are not trivial:
  (x as (T as A)).F();  // Legal
}
// K1 and K2 accept the same values
K1(y);  K2(y);
K1(y as (S as A));  K2(y as (S as A));
```

Furthermore, there are cases where you must use `TypeImplements(A)` instead of
`A`:

```
fn GetWithDefault
    [Type:$ K, TypeImplements(HasDefault):$ V]
    (HashMap(K, V): map, K: key) -> Ptr(V) {
  if (not map.has_key(key)) {
    map.insert(key, (V as HasDefault).default());
  }
  return &map[key];
}
```

The reason we need `V` to be `TypeImplements(HasDefault)` instead of just
`HasDefault` is so it can match the parameter to `HashMap(K, V)` which typically
won't be the `HasDefault` facet of whatever type you are using.

#### Subsumption

We have the following two subsumption rules:

> If `T` has type `TypeImplements(A1, ..., Am)`, it may be implicitly cast to
> `TypeImplements(B1, ..., Bn)` if for every `Bi` there is a `Aj` such that
> `Bi == Aj` or `Aj` extends `Bi`.

> If `T` has type `TypeImplements(A1, ..., Am)`, `T` may be implicitly cast to
> `T as Aj` for any `Aj` in `(A1, ..., Am)`.

This subsumption rule allows generic functions to call other generic functions
with equal or less-strict requirements.

```
fn L[TypeImplements(A, B):$ T](T: x) {
  H(x);  // Same requirements, no casting
  K1(x);  // Implicitly casts: x as (T as A)
  K2(x);  // Implicitly casts: x as (T as TypeImplements(A))
}
```

The subsumption rule implies that the composition of `TypeImplements`
expressions can be flattened:
`TypeImplements(TypeImplements(A, B), TypeImplements(C, D))` is the same as
`TypeImplements(A, B, C, D)`.

#### Model

The `TypeImplements` construction essentially creates an unnamed interface that
[contains the other interfaces](#interface-nestingcontainment-optional-feature).
Since the interface doesn't have a name, it is just matched structurally. So
`TypeImplements(A, B)` for interfaces `A` and `B` would be approximately
equivalent to

```
interface {
  impl A;
  impl B;
}
```

### Type compatible with another type

Given a type-type `TT` and a type `U`, define the type-type
`CompatibleWith(TT, U)` as follows:

> `CompatibleWith(TT, U)` is a type whose values are types `T` such that:
>
> -   `T` has type `TT`.
> -   `T` and `U` are
>     [compatible](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#compatible-types).
>     That is values of types `T` and `U` can be cast back and forth without any
>     change in representation (for example `T` is an [adaptor](#adapting-types)
>     for `U`).

**Note:** We require the user to supply `TT` and `U`, they may not be inferred.
Specifically, this code would be illegal:

```
fn Illegal[Type:$ U, CompatibleWith(TT, U):$ T](Ptr(T): x) ...
```

In general there would be multiple choices for `U` given a specific `T` here,
and no good way of picking one. However, similar code is allowed if there is
another way of determining `U`:

```
fn Allowed[Type:$ U, CompatibleWith(TT, U):$ T](Ptr(U): x, Ptr(T): y) ...
```

#### Example: Multiple implementations of the same interface

This allows us to represent functions that accept multiple implementations of
the same interface for a type.

```
enum CompareResult { Less, Equal, Greater }
interface Comparable {
  method (Self: this) Compare(Self: that) -> CompareResult;
}
fn CombinedLess[Type:$ T](T: a, T: b,
                          CompatibleWith(Comparable, T):$ U,
                          CompatibleWith(Comparable, T):$ V) -> Bool {
  match ((a as U).Compare(b)) {
    case CompareResult.Less => { return True; }
    case CompareResult.Greater => { return False; }
    case CompareResult.Equal => {
      return (a as V).Compare(b) == CompareResult.Less;
    }
  }
}
```

Used as:

```
struct Song { ... }
adaptor SongByArtist for Song { impl Comparable { ... } }
adaptor SongByTitle for Song { impl Comparable { ... } }
assert(CombinedLess(Song(...), Song(...), SongByArtist, SongByTitle) == True);
```

We might generalize this to a list of implementations:

```
fn CombinedCompare[Type:$ T]
    (T: a, T: b, List(CompatibleWith(Comparable, T)):$ CompareList)
    -> CompareResult {
  for (auto: U) in CompareList {
    var CompareResult: result = (a as U).Compare(b);
    if (result != CompareResult.Equal) {
      return result;
    }
  }
  return CompareResult.Equal;
}

assert(CombinedCompare(Song(...), Song(...), (SongByArtist, SongByTitle)) ==
       CompareResult.Less);
```

#### Example: Creating an impl out of other impls

And then to package this functionality as an implementation of `Comparable`, we
combine `CompatibleWith` with [type adaptation](#adapting-types):

```
adaptor ThenCompare(Type:$ T,
                    List(CompatibleWith(Comparable, T)):$ CompareList) for T {
  impl Comparable {
    method (Self: this) Compare(Self: that) -> CompareResult {
      for (auto : U) in CompareList {
        var CompareResult: result = (this as U).Compare(that);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

alias SongByArtistThenTitle = ThenCompare(Song, (SongByArtist, SongByTitle));
var Song: song = ...;
var SongByArtistThenTitle: song2 = Song(...) as SongByArtistThenTitle;
assert((song as SongByArtistThenTitle).Compare(song2) == CaompareResult.Less);
```

### Other type constraints

Some constraints, such as that some
[type parameter](#parameterized-interfaces-optional-feature) or
[associated type](#associated-types) must implement an interface or that two
must be equal, are [represented using multiple clauses](#external-constraints-by
way of-optional-parameters) rather than a single type-type. Sometimes we may
need a single type-type, such as to define a `DynPtr(TT)` (as
[described in the following dynamic pointer type section](#dynamic-pointer-type)).
There are a couple of approaches we could use to form a single type-type that
includes such constraints.

#### Recommendation: interface adapter

Imagine a construct for naming a new type-type that captures a set of
constraints on a type. For example, the type could be restricted to implementing
an interface with particular constraints on its type parameters. Analogous to
[adapting a type](#adapting-types), this could be called "adapting an
interface".

TODO

**Alternative approach:** unnamed interfaces, using structural conformance, may
also be able to express this. TODO

#### Rejected alternative: `ForSome(F)`

Another way to solve this problem would be to have a `ForSome(F)` construct,
where `F` is a function from types to type-types.

> `ForSome(F)`, where `F` is a function from type `T` to type-type `TT`, is a
> type whose values are types `U` with type `TT=F(T)` for some type `T`.

**Example:** Pairs of values where both values have the same type might be
written as

```
fn F[ForSome(lambda (Type:$ T) => PairInterface(T, T)):$ MatchedPairType]
    (Ptr(MatchedPairType): x) { ... }
```

This would be equivalent to:

```
fn F[Type:$ T, PairInterface(T, T):$ MatchedPairType]
    (Ptr(MatchedPairType): x) { ... }
```

**Example:** Containers where the elements implement the `HasEquality` interface
might be written as:

```
fn F[ForSome(lambda (TypeImplements(HasEquality):$ T) => Container(T)):$ ContainerType]
  (Ptr(ContainerType): x) { ... }
```

This would be equivalent to:

```
fn F[TypeImplements(HasEquality):$ T, Container(T):$ ContainerType]
  (Ptr(ContainerType): x) { ... }
```

### Sized types and type-types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`Int32`, `Float64`, etc.) most other concrete types (for
    example most
    [structs](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md)).
-   It could be known generically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a
    [dynamic type](#dynamic-pointer-type) such as `Dynamic(TT)`, a
    [variable-sized type](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#control-over-allocation),
    or you could dereference a pointer to a base type that could actually point
    to a
    [descendant](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#question-extension--inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

I'm going to call a type "sized" if it is in the first two categories, and
"unsized" otherwise. (Note: something with size 0 is still considered "sized".)
Given a type-type `TT`, define the type-type `Sized(TT)` as follows:

    `Sized(TT)` is a type whose values are types `T` with type `TT` that are "sized" -- that is the size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring (member/local) variables
of that type, taking values of that type as parameters, returning values of that
type, and defining arrays of that type. There will be other requirements as
well, such as being movable, copyable, or constructible from some types.

Example:

```
interface Foo {
  impl DefaultConstructible;  // See "interface nesting/containment" below.
}
struct Bar {  // Structs are "sized" by default.
  impl Foo;
}
fn F[Foo: T](Ptr(T): x) {  // T is unsized.
  var T: y;  // Illegal: T is unsized.
}
fn G[Sized(Foo): T](Ptr(T): x) { // T is sized, but its size is only known generically.
  var T: y;  // Allowed: T is sized and default constructible.
}
var Bar: z;
G(&z);  // Allowed: Bar is sized and implements Foo.
```

**Question:** Even if the size is fixed, it won't be known at the time of
compiling the generic function if we are using the dynamic strategy. Should we
automatically [box](#boxed) local variables when using the dynamic strategy? Or
should we only allow `MaybeBox` values to be instantiated locally?

#### Model

This requires a special integer field be included in the witness table type to
hold the size of the type. This field will only be known generically, so if its
value is used for type checking, we need some way of evaluating those type tests
symbolically.

## Dynamic types

Two different goals here:

-   Reducing code size at the expense of more runtime dispatch.
-   Increasing expressivity by allowing types to vary at runtime, AKA
    "[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch)".

We address these two different use cases with two different mechanisms. What
they have in common is using a runtime/dynamic type value (using
`InterfaceName: type_name`, no `$`) instead of a generic type value (using
`InterfaceName:$ type_name`, with a `$`). In the first case,
[we make the type parameter to a function dynamic](#runtime-type-parameters). In
the second case,
[we use a dynamic type value as a field in a struct](#runtime-type-fields). In
both cases, we have a name bound to a runtime type value, which is modeled by a
[dynamic-dispatch witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#dynamic-dispatch-witness-table)
instead of the
[static-dispatch witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#static-dispatch-witness-table)
used with generic type values.

### Runtime type parameters

If we pass in a type as an ordinary parameter (using `:` instead of `:$`), this
means passing the witness table as an ordinary parameter -- that is a
[dynamic-dispatch witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#dynamic-dispatch-witness-table)
-- to the function. This means that there will be a single copy of the generated
code for this parameter.

**Restrictions:** The type's size will only be known at runtime, so patterns
that use a type's size such as declaring local variables of that type or passing
values of that type by value are forbidden. Essentially the type is considered
[unsized](#sized-types-and-type-types), even if the type-type uses the
`Sized(TT)` function.

**Note:** In principle you could imagine supporting values with a dynamic size,
but it would add a large amount of implementation complexity and would not have
the same runtime performance in a way that would likely be surprising to users.
Without a clear value proposition, it seems better just to ask the user to
allocate anything with a dynamic size on the heap using something like
<code>[Boxed](#boxed)</code> below.

**Question:** Should we prevent interfaces that have functions that accept
`Self` parameters or return `Self` values (and therefore violate the unsized
restriction) from being used as the type of runtime type parameters, or should
just those functions be blacklisted?

TODO examples

### Runtime type fields

Instead of
[passing in a single type parameter to a function](#runtime-type-parameters), we
could store a type per value. This changes the data layout of the value, and so
is a somewhat more invasive change. It also means that when a function operates
on multiple values they could have different real types, and so
[there are additional restrictions on what functions are supported, like no binary operations](#bookmark=id.3atpcs1p4un9).

**Terminology:** Not quite
["Late binding" on Wikipedia](https://en.wikipedia.org/wiki/Late_binding), since
this isn't about looking up names dynamically. It could be called
"[dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch)", but that
does not distinguish it from [runtime type parameter](#runtime-type-parameters)
(both use
[dynamic-dispatch witness tables](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#dynamic-dispatch-witness-table))
or normal
[virtual method dispatch](https://en.wikipedia.org/wiki/Virtual_function).

#### Dynamic pointer type

Given a type-type `TT` (with some restrictions described below), define
`DynPtr(TT)` as a type that can hold a pointer to any value `x` with type `T`
satisfying `TT`. Variables of type `DynPtr(TT)` act like pointers:

-   They do not own what they point to.
-   They have an assignment operator which allows them to point to new values
    (with potentially different types as long as they all satisfy `TT`).
-   They may be copied or moved.
-   They have a fixed size (unlike the values they point to), though that size
    is larger than a regular pointer.

Example:

```
interface Printable {
  method (Ptr(Self): this) Print();
}
struct AnInt {
  var Int: x;
  impl Printable { method (Ptr(Self): this) Print() { PrintInt(this->x); } }
}
struct AString {
  var String: x;
  impl Printable { method (Ptr(Self): this) Print() { PrintString(this->x); } }
}

var AnInt: i = (.x = 3);
var AString: s = (.x = "Hello");

var DynPtr(Printable): i_dynamic = &i;
i_dynamic->Print();  // Prints "3".
var DynPtr(Printable): s_dynamic = &s;
s_dynamic->Print();  // Prints "Hello".

var DynPtr(Printable)[2]: dynamic = (&i, &s);
for (DynPtr(Printable): iter) in dynamic {
  // Prints "3" and then "Hello".
  iter->Print();
}
```

This corresponds to
[a trait object reference in Rust](https://doc.rust-lang.org/book/ch17-02-trait-objects.html).

**Restrictions:** The member functions in the `TT` interface must only have
`Self` in the "receiver" or "this" position.

This is similar to
[the "object safe" restriction in Rust](https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md)
and for the same reasons. Consider an interface that takes `Self` as an
argument:

```
interface EqualCompare {
  method (Ptr(Self): this) IsEqual(Ptr(Self): that) -> Bool;
}
```

and implementations of this interface for our two types:

```
impl EqualCompare for AnInt {
  method (Ptr(AnInt): this) IsEqual(Ptr(AnInt): that) -> Bool {
    return this->x == that->x;
  }
}
impl EqualCompare for AString {
  method (Ptr(AString): this) IsEqual(Ptr(AString): that) -> Bool {
    return this->x == that->x;
  }
}
```

Now given two values of type `Dynamic(EqualCompare)`, what happens if we try and
call `IsEqual`?

```
var DynPtr(EqualCompare): i_dyn_eq = &i;
var DynPtr(EqualCompare): s_dyn_eq = &s;
i_dyn_eq->IsEqual(&*s_dyn_eq);  // Unsound: runtime type confusion
s_dyn_eq->IsEqual(&*i_dyn_eq);  // Unsound: runtime type confusion
```

For `*i_dyn_eq` to implement `EqualCompare.IsEqual`, it needs to accept any
`Ptr(DynPtr(EqualCompare).T)` value for `that`, including `&*s_dyn_eq`. But
`i_dyn_eq->IsEquals(...)` is going to call `AnInt.EqualCompare.IsEqual` which
can only deal with values of type `AnInt`. So this construction is unsound.

Similarly, we can't generally convert a return value using a specific type (like
`AnInt`) into a value using the dynamic type, that has a different
representation.

**Concern:** "object safe" removes one of the expressivity benefits of generics
over inheritance, the ability to use `Self` arguments and return values like
`Compare(Self, Self)` and `Clone(Self) -> Self`. Is this going to be too
restrictive or prevent too many interfaces from being object safe?

Rust has a way of defining some methods only being present in an interface if
there is also a "sized" restriction on the type. Since Rust's trait objects are
not sized, this provides a mechanism for having some methods in the interface
only in situations where you know the type isn't dynamic.

**Context:** See
["Proposal: Fat pointers" section of Carbon: types as function tables, interfaces as type-types (TODO)](#broken-links-footnote)<!-- T:Carbon: types as function tables, interfaces as type-types --><!-- A:#heading=h.t96sqqyjs8xu -->,
[Existential types](https://en.wikipedia.org/wiki/Type_system#Existential_types)
or
[Dependent pair types](https://en.wikipedia.org/wiki/Dependent_type'%22%60UNIQ--postMath-00000012-QINU%60%22_type).

##### Model

TODO

```
// Note: InterfaceType is essentially "TypeTypeType".
struct DynPtr(InterfaceType:$$ TT) {  // TT is any interface
  struct DynPtrImpl {
    private TT: t;
    private Ptr(Void): p;  // Really Ptr(t) instead of Ptr(Void).
    impl TT {
      // Defined using meta-programming.
      // Forwards this->F(...) to (this->p as Ptr(this->t))->F(...)
      // or equivalently, this->t.F(this->p as Ptr(this->t), ...).
    }
  }
  var TT:$ T = (DynPtrImpl as TT);
  private DynPtrImpl: impl;
  fn operator->(Ptr(Self): this) -> Ptr(T) { return &this->impl; }
  fn operator=[TT:$ U](Ptr(Self): this, Ptr(U): p) { this->impl = (.t = U, .p = p); }
}
```

#### Deref

To make a function work on either regular or dynamic pointers, we define an
interface `Deref(T)` that both `DynPtr` and `Ptr(T)` implement:

```
// Types implementing `Deref(T)` act like a pointer to `T`.
interface Deref(Type:$ T) {
  // This is used for the `->` and `*` dereferencing operators.
  method (Self: this) Deref() -> Ptr(T);
  impl Copyable;
  impl Movable;
}

// Implementation of Deref() for DynPtr(TT).
impl[InterfaceType:$$ TT] Deref(DynPtr(TT).DynPtrImpl as TT) for DynPtr(TT);
// or equivalently:
impl[InterfaceType:$$ TT] Deref(DynPtr(TT).T) for DynPtr(TT);

// Implementation of Deref(T) for Ptr(T).
impl[Type:$ T] Deref(T) for Ptr(T) {
  method (Ptr(T): this) Deref() -> Ptr(T) { return this; }
}
```

Now we can implement a function that takes either a regular pointer to a type
implementing `Printable` or a `DynPtr(Printable)`:

```
// This is equivalent to `fn PrintIt[Printable:$ T](T*: x) ...`,
// except it also accepts `DynPtr(Printable)` arguments.
fn PrintIt[Printable:$ T, Sized(Deref(T)): PtrT](PtrT: x) {
  x->Print();
}
PrintIt(&i); // T == (AnInt as Printable), PtrT == T*
// Prints "3"
PrintIt(&s); // T == (AString as Printable), PtrT == T*
// Prints "Hello"
PrintIt(dynamic[0]);  // T == DynPtr(Printable).T, PtrT == DynPtr(Printable)
// Prints "3"
PrintIt(dynamic[1]);  // T == DynPtr(Printable).T, PtrT == DynPtr(Printable)
// Prints "Hello"
```

#### Boxed

One way of dealing with unsized types is by way of a pointer, as with `T*` and
`DynPtr` above. Sometimes, though, you would like to work with something closer
to value semantics. For example, the `Deref` interface and `DynPtr` type
captures nothing about ownership of the pointed-to value, or how to destroy it.

So we are looking for the equivalent of C++'s `unique_ptr<T>`, that will handle
unsized types, and then later we will add a variation that supports dynamic
types. `Boxed(T)` is like `unique_ptr<T>`:

-   it has a fixed size
-   is movable even if `T` is not
-   will destroy what it points to when it goes out of scope.

It differs, though, in that `Boxed(T)` has an allocator and so can support
copying if `T` does.

```
// Boxed is sized and movable even if T is not.
struct Boxed(Type:$ T,
             // May be able to add more constraints on AllocatorType (like
             // sized & movable) so we could make it a generic argument?
             AllocatorInterface:$$ AllocatorType = DefaultAllocatorType) {
  private var T*: p;
  private var AllocatorType: allocator;
  operator create(T*: p, AllocatorType: allocator = DefaultAllocator) { ... }
  impl Movable { ... }
}

// TODO: Should these just be constructors defined within Boxed(T)?
// If T is constructible from X, then Boxed(T) is constructible from X ...
impl[ConstructibleFrom(...:$$ Args): T] ConstructibleFrom(Args) for Boxed(T) {
  ...
}
// ... and Boxed(X) as well.
impl[ConstructibleFrom(...:$$ Args): T] ConstructibleFrom(Boxed(Args))
    for Boxed(T) { ... }

// This allows you to create a Boxed(T) value inferring T so you don't have to
// say it explicitly.
fn Box[Type:$ T](T*: x) -> Boxed(T) { return Boxed(T)(x); }
fn Box[Type:$ T, AllocatorInterface:$$ AllocatorType]
    (T*: x, AllocatorType: allocator) -> Boxed(T, AllocatorType) {
  return Boxed(T, AllocatorType)(x, allocator);
}
```

NOTE: Chandler requests that boxing be explicit so that the cost of indirection
is visible in the source (and in fact visible wherever the dereference happens).
This solution also accomplishes that but may not address all use cases for
boxing.

#### DynBoxed

`DynBoxed(TT)` is to `Boxed(T)` as `DynPtr(TT)` is to `T*`. Like `DynPtr(TT)`,
it holds a pointer to a value of any type `T` that satisfies the interface `TT`.
Like `Boxed(T)`, it owns that pointer.

TODO

```
struct DynBoxed(InterfaceType:$$ TT,
                AllocatorInterface:$$ AllocatorType = DefaultAllocatorType) {
  private DynPtr(TT): p;
  private var AllocatorType: allocator;
  ...  // Constructors, etc.
  // Destructor deallocates this->p.
  impl Movable { ... }
}
```

**Question:** Should there be some mechanism to have values be dynboxed in
fast-compile builds, but not boxed in release builds?

**Answer:** Right now we are going with the static strategy for both, and are
just going to focus on making that fast.

#### MaybeBoxed

We have a few different ways of making types with value semantics:

-   `Boxed(T)`: Works with sized and unsized concrete types, `T` need not be
    movable. Even if `T` is movable, it may be large or expensive to move so you
    rather used `Boxed(T)` instead.
-   `DynBoxed(TT)`: Can store values of any type satisfying the interface (so
    definitely unsized). Performs
    [dynamic dispatch](https://en.wikipedia.org/wiki/Dynamic_dispatch).
-   `T`: Regular values that are sized and movable. The extra
    indirection/pointer and heap allocation for putting `T` into a box would
    introduce too much overhead / cost.

In all cases we end up with a sized, movable value that is not very large. Just
like we did with <code>[Deref(T) above](#deref)</code>, we can create an
interface to abstract over the differences, called <code>MaybeBoxed(T)</code>:

```
interface MaybeBoxed(Type:$ T) {
  fn operator->(Ptr(Self): this) -> Ptr(T);
  // plus other smart pointer operators
  impl Movable;
}
// TODO: Want some way to say that MaybeBoxed(T) should be sized, to avoid all
// users having to say so?

impl[Type:$ T] MaybeBoxed(T) for Boxed(T) {
  fn operator->(Ptr(Self): this) -> Ptr(T) { return this->p; }
}

impl[InterfaceType:$$ TT] MaybeBoxed(DynBoxed(TT).T) for DynBoxed(TT) {
  ...  // TODO
}
```

For the case of values that we can efficiently move without boxing, we implement
a new type `NotBoxed(T)` that adapts `T` and so has the same representation and
supports zero-runtime-cost casting.

```
// Can pass a T to a function accepting a MaybeBoxed(T) value without boxing by
// first casting it to NotBoxed(T), as long as T is sized and movable.
adaptor NotBoxed(Sized(TypeImplements(Movable)):$ T) for T {  // :$ or :$$ here?
  impl Movable = T as Movable;
  impl MaybeBoxed(T) {
    fn operator->(Ptr(Self): this) -> Ptr(T) { return this as Ptr(T); }
  }
}
// TODO: Should this just be a constructor defined within NotBoxed(T)?
// Says NotBoxed(T) is constructible from a value of type Args if T is.
impl[ConstructibleFrom(...:$$ Args): T] ConstructibleFrom(Args) for NotBoxed(T) {
  ...
}

// This allows you to create a NotBoxed(T) value inferring T so you don't have to
// say it explicitly. TODO: Could probably replace "Type:$$ T" with
// "Sized(TypeImplements(Movable)):$ T", here.
fn DontBox[Type:$$ T](T: x) -> NotBoxed(T) inline { return x as NotBoxed(T); }
// Use NotBoxed as the default implementation of MaybeBoxed for small & movable
// types. TODO: Not sure how to write a size <= 16 bytes constraint here.
impl[Sized(TypeImplements(Movable)):$$ T] if (sizeof(T) <= 16)
    MaybeBoxed(T) for T = NotBoxed(T);
```

This allows us to write a single generic function using that interface and have
the caller decide which of these mechanisms is the best fit for the specific
types being used.

```
interface Foo { method (Ptr(Self): this) F(); }
fn UseBoxed[Foo:$ T, Sized(MaybeBoxed(T)):$ BoxType](BoxType: x) {
  x->F();  // Possible indirection is visible
}
struct Bar { impl Foo { ... } }
var DynBoxed(Foo): y = new Bar(...);
UseBoxed(y);
// DontBox might not be needed, if Bar meets the requirements to use the
// default NotBoxed impl of MaybeBox.
UseBoxed(DontBox(Bar()));
```

## Implicit interface arguments [rejected optional feature]

(Right now I'm leaning against the extra complexity of this feature.)

An alternative mechanism for
[functions requiring a type to implement multiple interfaces](#type-implementing-multiple-interfaces),
would be to have extra arguments in the implicit argument list without colons
(`:`). For example:

```
// Instead of: fn H[TypeImplements(A, B):$ T](T: x) { ... }
fn H[Type:$ T, T as A, T as B](T: x) { ... }
```

The idea here is that the pattern matching syntax used inside the parens
(`(`...`)`) allows you to either match any value with a type by using a colon
(&lt;type> `:` &lt;name>) or match a specific value without (&lt;value>). We
would extend that idea to the syntax used in the implicit argument list inside
the brackets (`[`...`]`) to also allow values without a colon (`:`), to (a)
assert that an expression evaluates to a legal value and (b) have the caller
pass whatever information is needed to so that same expression can be used
inside the function (for example a
[witness table](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#witness-tables-for-example-swift-and-carbon-generics)).

**Argument against this feature:** It is redundant with `TypeImplements()`, and
the `TypeImplements()` feature is more broadly applicable / more composable.
Consider this example from the
["Example: Multiple implementations of the same interface" section](#example-multiple-implementations-of-the-same-interface):

```
fn CombinedCompare[Type:$ T]
    (..., List(CompatibleWith(Comparable, T)):$ CompareList) ...
```

What if we wanted `CompareList` to have types that implement two interfaces
`Foo` and `Bar` instead of just `Comparable`? With `TypeImplements`, this is
straightforward:

```
fn CombinedCompare[Type:$ T]
    (..., List(CompatibleWith(TypeImplements(Foo, Bar), T)):$ CompareList) ...
```

This construction comes up with the type of variadic arguments, when their type
can vary but they all need to implement some interfaces. I don't see how to
represent the same thing with extra implicit arguments.

## Interface nesting/containment [optional feature]

Just as we might want to support
[interface extension](#interface-extension-optional-feature), we may also want
to support a containment relationship between interfaces & implementations. This
would have the advantage that each interface would get a separate namespace, and
you could easily contain more than one impl (while extending multiple interfaces
is scarier). Example:

```
interface Inner1 {
  method (Self: this) K();
}
interface Inner2 {
  method (Self: this) L();
}
interface Outer {
  impl Inner1;
  impl Inner2;
}
struct S {
  impl Inner2 {
    method (S: this) L() { ... }
  }
  impl Outer {
    impl Inner1 {
      method (S: this) K() { ... }
    }
    // impl of Inner2 here uses (S as Inner2) by default.
  }
}
var S: y = ...;
(y as ((S as Outer) as Inner1)).K();
```

The fact that the `Outer` interface requires an implementation of two other
interfaces is now well captured in a compositional way. If `S` directly
implements `Inner1` or `Inner2`, it could use that as the default in the impl of
`Outer`.

This, combined with unnamed/structural interfaces, would be a building block for
the
[`TypeImplements` construction above](#type-implementing-multiple-interfaces).

**Question:** A similar feature that is perhaps a bit simpler would be to say
that interfaces can require other interfaces, but there is no containment.
Instead, types would contain a flat list of implementations.

For the example above, we could leave the definitions of the interfaces alone,
but the interpretation would be different. Instead of `Outer` containing
`Inner1` and `Inner2`, it would simply have a requirement that the type
implement them separately.

```
interface Inner1 {
  method (Self: this) K();
}
interface Inner2 {
  method (Self: this) L();
}
interface Outer {
  impl Inner1;
  impl Inner2;
}
struct S {
  impl Inner1 {
    method (S: this) K() { ... }
  }
  impl Inner2 {
    method (S: this) L() { ... }
  }
  impl Outer {
    // Requirements satisfied by impl of Inner1 and Inner2 above.
  }
}
var S: y = ...;
(y as (S as Inner1)).K();
```

## Index of examples

Specifically, how does this proposal address
[the use cases from the "problem statement" doc](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/use-cases.md)?

-   Define an
    [interface](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface).
    -   [in "Interfaces" section](#interfaces) (and most other sections).
-   Define an interface with
    [type parameters](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-versus-associated-types)
    (maybe) and/or
    [associated types](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-versus-associated-types)
    (almost certainly).
    -   [associated types](#associated-types),
    -   [type parameters](#parameterized-interfaces-optional-feature).
-   Define an interface with
    [type constraints](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#type-constraints),
    such as associated types or type parameters satisfying some interface. Type
    constraints will also be needed as part of generic function definitions, to
    define relationships between type parameters and associated types.
    -   [associated types](#bookmark=id.y76bpmnyjm7k),
    -   [type parameters](#bookmark=id.arne8eq43vmm).
-   Optional, but probably straightforward if we want it: Define an interface
    that
    [extends/refines](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#extendingrefining-an-interface)
    another interface:
    -   [interface extension](#interface-extension-optional-feature).
-   Similarly we probably want a way to say an interface requires an
    implementation of one or more other interfaces:
    -   [interface nesting/containment](#interface-nestingcontainment-optional-feature).
-   Define how a type
    [implements](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#impls-implementations-of-interfaces)
    an interface
    ([semantic conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#semantic-versus-structural-interfaces)).

    -   [implementing interfaces](#implementing-interfaces)

    It should address
    [the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
    for example by allowing the impl definition to be completely out of line as
    long as it is defined with either the type or the interface.

    -   [Out-of-line impl](#out-of-line-impl-arguments-for-parameterized-types)
    -   [Impl lookup](#impl-lookup)

-   Define a parameterized implementation of an interface for a family of types.
    This is both for
    [structural conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#semantic-versus-structural-interfaces)
    by way of
    [templated impls](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#templated-impl),
    and
    [conditional conformance](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#conditional-conformance).
    That family of types may have generic or regular parameters, so that for
    example you could implement a `Printable` interface for arrays of `N`
    elements of `Printable` type `T`, generically for `N` (not separately
    instantiated for each `N`).
    -   [Conditional conformance](#conditional-conformance)
    -   [Templated impls for structural conformance](#templated-impls-for-generic-interfaces)
        (TODO)
-   Control how an interface may be used in order to reserve or abandon rights
    to evolve the interface. See
    [the relevant open question in "Carbon closed function overloading proposal" (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal --><!-- A:#bookmark=id.hxvlthy3z3g1 -->.
    -   TODO
-   Specify a generic explicit (non-type or type) argument to a function:
    -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)..
-   Specify a generic
    [implicit argument](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#implicit-argument)
    to a function:
    -   ["generics" section](#generics).
-   Specify a generic type argument constrained to conform to an interface. And
    in the function, call methods defined in the the interface on a value of
    that type.
    -   ["generics" section](#generics),
    -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface).
-   Specify a generic type argument constrained to conform to multiple
    interfaces. And in the function, call methods defined in each interface on a
    value of that type, and pass the value to functions expecting any subset of
    those interfaces. Ideally this would be convenient enough that we could
    favor fewer narrow interfaces and combine them instead of having a large
    number of wide interfaces.
    -   [Type implementing multiple interfaces](#type-implementing-multiple-interfaces).
-   Define multiple implementations of an interface for a single type, be able
    to pass those multiple implementations in a single function call, and have
    the function body be able to control which implementation is used when
    calling interface methods. This should work for any interface, without
    requiring cooperation from the interface definition. For example, have a
    function sort songs by artist, then by album, and then by title given those
    three orderings separately.
    -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface).
-   In general, ways of specifying new combinations of interface implementations
    for a type. For example, a way to call a generic function with a value of
    some type, even if the interface and type are defined in different libraries
    unknown to each other, by providing an implementation for that interface in
    some way. This problem is described in
    "[The trouble with typeclasses](https://pchiusano.github.io/2018-02-13/typeclasses.html)".
    -   [Adapting types](#adapting-types).
-   A value with a type implementing a superset of the interfaces required by a
    generic function may be passed to the function without additional syntax
    beyond passing the same value to a non-generic function expecting the exact
    type of the value
    ([subsumption](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#subsumption-and-casting)).
    This should be true for values with types only known generically, as long as
    it is generically known that the type implements a sufficient set of
    interfaces.
    -   [Subsumption](#subsumption)
-   Define a parameterized entity (such as a function) such that code for it
    will only be generated once.
    -   [Runtime type parameters](#runtime-type-parameters)
-   Define a parameterized entity such that code for it will be generated
    separately for each distinct combination of arguments.
    -   [Generics](#generics)
-   Convert values of arbitrary types implementing an interface into values of a
    single type that implements that same interface, for a sufficiently
    well-behaved interface.
    -   [Runtime type fields](#runtime-type-fields)

Stretch goals:

-   A way to define one or a few functions and get an implementation for an
    interface that has more functions (like defining `<`, `>`, `<=`, `>=`, `==`,
    and `!=` in terms of `<=>`, or `++`, `--`, `+`, `-`, and `-=` from `+=`).
    Possibly the "one or few functions" won't even be part of the interface.
    -   [Example: Defining an impl for use by other types](#example-defining-an-impl-for-use-by-other-types)
-   Define an interface implementation algorithmically -- possibly by way of a
    function returning an impl, or by defining an
    [adapting type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#adapting-a-type)
    that implements that interface. This could be a solution to the previous
    bullet. Another use case is when there are few standard implementation
    strategies for an interface, and you want to provide those implementations
    in a way that makes it easy for new types to adopt one.
    -   [Example: Defining an impl for use by other types](#example-defining-an-impl-for-use-by-other-types)
-   Support a way to switch between algorithms based on the capabilities of a
    type. For example, we may want to use different algorithms for random-access
    versus bidirectional iterators. Similarly, a way to have specialization
    based on type information in a generic like you might do in a template
    function for performance but still would allow type checking. Example: In
    C++, `std::vector<T>::resize()` can use a more efficient algorithm if `T`
    has a `noexcept` move constructor. Can this optimization be allowed from
    generic code since it does not affect the signature of `resize()`, and
    therefore type checking? In a non-release build, it would be semantically
    equivalent but slower to ignore the optimized implementation.
    -   [Use case: overload resolution](#use-case-overload-resolution)
-   As much as possible, switching a templated function to a generic one should
    involve minimal changes to the function body. It should primarily just
    consist of adding constraints to the signature. When changes are needed, the
    compiler will not accept the code without them. No semantics of any code
    will change merely as the result of switching from template to generics. See
    ["Carbon principle: Generics"](https://github.com/josh11b/carbon-lang/blob/principle-generics/docs/project/principles/principle-generics.md).
    -   No semantic changes to code when switching from template -> generic.
    -   Only changes needed are additional casts when there are
        [multiple interface requirements for a single type](#type-implementing-multiple-interfaces).
        Code will not compile without those changes.
    -   No change to function body code going from generic -> template.

Very stretch goals (these are more difficult, and possibly optional):

-   Define an interface where the relationship between the input and output
    types is a little complicated. For example, widening multiplication from an
    integer type to one with more bits, or
    `Abs: Complex(SomeIntType) -> SomeFloatType`. One possible strategy is to
    have the return type be represented by an
    [associated type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/terminology.md#interface-type-parameters-versus-associated-types).
    -   [Constraints on associated types in interfaces](#constraints-on-associated-types-in-interfaces)
-   Define an interface that has multiple related types, like Graph/Nodes/Edges.
    TODO: A concrete combination of `Graph`, `Edge`, and `Node` types that we
    would like to define an interface for. Is the problem when you `Edge` and
    `Node` refer to each other, so you need a forward declaration to break the
    cycle?
    -   TODO
-   Impls where the impl itself has state. (from richardsmith@) Use case:
    implementing interfaces for a flyweight in a Flyweight pattern where the
    Impl needs a reference to a key -> info map.
    -   TODO: Difficult! Not clear how this would work.
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
    (Ptr(StackLike(T)): x, fn (T)->V: f) -> StackLike(V) { ... }
```

    -   TODO: Challenging! Probably needs something like [Dependent function types](https://en.wikipedia.org/wiki/Dependent_type#Pi_type)

These mechanisms need to have an underlying programming model that allows users
to predict how to do these things, how to compose these things, and what
expressions are legal.

-   See the [main "Model" section](#model), in addition to "Model" subsections
    throughout the doc

## Notes

These are notes from discussions after this document was first written that have
not yet been incorporated into the main text above.

-   Can use IDE tooling to show all available methods, automatically inserting
    casting to a facet type to get that method
-   "Functional dependencies" in Haskell about the unique/generic versus
    non-unique/template arguments distinction for type parameters of interfaces
-   Open question: is it okay for interface definer to decide (a) whether type
    used in constraints, (b) whether different parameters make different
    interfaces that a type can implement multiple of.
-   Example: what is the type of the subsequence of a sequence, from Dmitri
-   Imagine I want to define a new interface that has the type implements
    property but also implements a new method.
-   Static/type fields holding non-type values: Should have these, should not
    have distinction between types/functions/other types.
-   Instance fields: capabilities include read, write, address-of (implies read
    & write?). Swift also has a modify capability implemented using coroutines.
    If we have address-of, it must be a real address.
-   Question: C++ maybe gets wrong that you can take address of any member.
    Greatly simplifies sanitizers, makes reasoning about what side effects can
    affect members for correctness easier. Maybe opt-in feature? Similarly for
    local variables. Maybe can call function taking a pointer from a member
    function as long as it doesn't capture? Need to firm up design for example
    fields before interfaces for example fields.
-   Concern about interfaces for operator overloading: Point + Vector = Point,
    Point - Point = Vector
-   Concern about type-type model: adds friction to binary operators -- is left
    or right type is self? Couple of problems with the idea of interface
    implemented for the (LeftType, RightType) tuple. How does the impl get
    passed in? How do you say that an interface is only for pairs?
-   Use case problem: Have interface `MultipliesBy(R)` for Self \* R. Want to
    write a constraint that a function can take any `R` type such that type
    `Foo` implements `MultipliesBy(R)`.
-   Want inheritance with virtual functions to be modeled by interface
    extension. Example showing the interaction between Dynamic pointer types and
    interface extension.

How do we represent binary operations like `Addable`? Could the interface be
defined on the pair of types of the two arguments somehow?

```
fn F(A: a, B: b, ..., Addable(A, B):$ T) requires (A,B) : Addable(A, B) {
  ((A, B) as T).DoTheAdd(x, y)
}
```

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
