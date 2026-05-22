# Implicit conversions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/820)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [C++ conversions](#c-conversions)
        -   [Array-to-pointer conversions](#array-to-pointer-conversions)
        -   [Function-to-pointer conversions](#function-to-pointer-conversions)
        -   [Qualification conversions](#qualification-conversions)
        -   [Integral promotions](#integral-promotions)
        -   [Floating-point promotions](#floating-point-promotions)
        -   [Integral conversions](#integral-conversions)
        -   [Floating-point conversions](#floating-point-conversions)
        -   [Pointer conversions](#pointer-conversions)
        -   [Pointer-to-member conversions](#pointer-to-member-conversions)
        -   [Function pointer conversions](#function-pointer-conversions)
        -   [Boolean conversions](#boolean-conversions)
    -   [No conversions](#no-conversions)
    -   [No extensibility](#no-extensibility)
    -   [Transitivity](#transitivity)

<!-- tocstop -->

## Problem

Frequently, an expression provided as input to an operation has a type that does
not exactly match the expected type. To improve the language ergonomics, we do
not want to require explicit conversions in all such cases. However, there is
strong evidence from C++ that allowing certain kinds of implicit conversion is
dangerous and harmful in practice. We need to find a reasonable balance.

## Background

C++ permits many kinds of implicit conversion, some of which are generally
considered good, and others are sometimes harmful. For example:

-   `int` implicitly converts to `long`. This is useful and seldom harmful.
-   `long` implicitly converts to `int` and to `unsigned int`. This can result
    in data loss.
-   `int*` implicitly converts to `bool`. This can be useful in some contexts,
    such as `if (p)`, but surprising and harmful in others.

See also
[implicit conversions in C++](https://en.cppreference.com/w/cpp/language/implicit_conversion).

## Proposal

See changes to design.

## Rationale based on Carbon's goals

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Disallowing implicit conversions that lose information reduces the risk
        that existing code will be reinterpreted in a harmful way as libraries
        in use evolve.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Permitting a limited, safe set of implicit conversions reduces the
        boilerplate work necessary to write code.
    -   Generics rely on performing implicit conversions between different
        type-of-types for deduced type parameters. Applying the same rules
        consistently for all expressions makes the language simpler.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Providing some of the same implicit conversions as C++ reduces the need
        to add explicit casts when migrating. However, explicit casts will still
        be required when the C++ code was performing an operation that we don't
        consider safe.
    -   Support for implicit conversions provides a path to expose converting
        constructors and conversion functions defined in C++ code to Carbon.

## Alternatives considered

### C++ conversions

We could permit more of the conversions that C++ does. This section considers
each kind of implicit conversion in C++ and provides a description of the
deviation and a rationale.

#### Array-to-pointer conversions

Array types have not yet been designed yet, so this is out of scope for now.

One possible design would be for pointers to not support arithmetic, and for
arrays to provide "array iterators" that do supply such arithmetic. In this
design, an implicit conversion from arrays to array iterators would likely be
surprising.

#### Function-to-pointer conversions

Function pointer types have not been designed yet, and might not exist in the
same form as in C++, so this is out of scope for now.

One possible design would be to have no function pointer types, and instead
model functions as values of a unique type that implements a certain `Callable`
interface. Then a function pointer could be modeled as a type-erased generic
implementing `Callable`. In this model, there would be an implicit conversion
from a function value to such a type-erased generic value.

#### Qualification conversions

So far, Carbon has no notion of cv-qualification. However, these conversions
would likely be covered by the permission to convert from `T*` to `U*` if `T` is
a subtype of `U`.

#### Integral promotions

Carbon disallows implicit conversion from `bool` to integral types. We could
permit such implicit conversions.

Advantages:

-   Improves C++ compatibility.
-   Permits constructs to count how many of a set of predicates were true:
    `if (cond1 + cond2 + cond3 >= 2)`.

Disadvantages:

-   Treating truth values as the integers 0 and 1 results in code that is harder
    to read and understand.
-   This conversion can result in unexpected overloads being called, when a
    `bool` argument is passed to a parameter of some other type.

#### Floating-point promotions

This conversion is permitted.

#### Integral conversions

These conversions are only permitted when they are known to preserve the
original value. These are the conversions that are considered non-narrowing in
C++.

We could permit narrowing integer conversions.

Advantages:

-   Improves C++ compatibility.
-   Allows implicitly undoing implicit widening in constructs such as
    `char n; char c = '0' + n;` where C++ promotes `'0' + n` to `int`.
    -   However, Carbon is unlikely to implicitly widen to `i32` here.

Disadvantages:

-   Introduces the potential for implicit data loss.

#### Floating-point conversions

Carbon disallows implicit conversion from a more-precise floating-point type to
a less-precise floating-point type, such as from `f64` to `f32`. We could permit
these implicit conversions.

Advantages:

-   Improves C++ compatibility.
-   Allows implicitly undoing implicit widening in constructs such as
    `float a, b; float c = a + b;` where C++ promotes `a + b` to `double`.
    -   However, Carbon might not implicitly widen to `f64` here.

Disadvantages:

-   Introduces the potential for implicit loss of precision.
-   Introduces the risk that a low-precision operation might be selected when
    given higher-precision operands.

#### Pointer conversions

Carbon permits the equivalent conversions, except for the conversion from
`nullptr` to pointer type. We anticipate that Carbon pointers will not be
nullable by default.

Once nullable pointers are designed, we would expect an expression representing
the null state would be implicitly convertible to the nullable pointer type.

#### Pointer-to-member conversions

Carbon does not yet have pointer-to-member types. This is out of scope for now.

#### Function pointer conversions

Carbon does not yet have function pointer types. This is out of scope for now.

#### Boolean conversions

An implicit conversion from arithmetic types and pointer types to `bool` is not
provided. Pointer types are expected to not be nullable by default, so that part
is out of scope for now.

We could permit implicit conversion from arithmetic types to `bool`.

Advantages:

-   Improves C++ compatibility and familiarity to C++ programmers.

Disadvantages:

-   Harms type safety by permitting an implicit lossy conversion.
    -   Invites bugs where the wrong overload is selected, where an argument of
        arithmetic type is passed to a `bool` parameter.
-   Harms the mental model of `bool` being a choice type rather than an integer
    type.
-   Allowing an implicit conversion would permit this kind of conversion
    everywhere, whereas it is likely only desirable in a select few places, such
    as where C++ performs a "contextual conversion to `bool`".

### No conversions

We could permit no implicit conversions at all, or restrict the set of
conversions from those proposed.

Advantages:

-   Code might be easier to understand, because all conversions would be fully
    explicit.

Disadvantages:

-   Code is likely to be harder to read and harder to write due to casts being
    inserted frequently.
-   Creates tension for generics, where implicit conversions between
    type-of-types are a central part of the model.

### No extensibility

We could provide only built-in conversions and no user-defined implicit
conversions.

Advantages:

-   Ensures that programmers don't add irresponsible implicit conversions.

Disadvantages:

-   Creates an artificial distinction between built-in and user-defined types.
-   Creates problems for interoperation with C++ and migration from C++, because
    certain forms of user-defined implicit conversion are common in C++ code.
-   Disallows useful functionality without sufficient justification.

### Transitivity

We could apply implicit conversions transitively. If an implicit conversion from
`A` to `B` is provided and an implicit conversion from `B` to `C` is provided,
we could try to infer an implicit conversion from `A` to `C`.

This leads to practical problems, as there would be an unbounded search space
for intermediate `B` types. For example:

```
impl [T:! Constraint1] A as ImplicitAs(T);
impl [T:! Constraint2] T as ImplicitAs(B);
let x: A = ...;
let y: B = x as B;
```

There is a potentially unbounded space of types to search here (anything that
satisfies both `Constraint1` and `Constraint2` at once. Similarly:

```
class X(N: i32, M: i1) {}
impl [template N:! i32] X(N, 0) as ImplicitAs(X(N+1, 0));
impl [template N:! i32] X(N, 0) as ImplicitAs(X(N+1, 1));
impl [template N:! i32] X(N, 1) as ImplicitAs(X(N+1, 1));
let z: auto = ({} as X(0, 0)) as X(100, 0);
```

This could lead to a very long implicit conversion sequence (which will
presumably need exponential runtime to find).

We could support partial transitivity, for only unparameterized intermediate
types, by ignoring all blanket impls. But that would be arbitrary, and we can
provide better results by first matching the overall source and destination
types and then asking them what intermediate type we should be converting to,
which is supported by this proposal. For example, for `Optional`:

```
impl [T:! Type, U:! ImplicitAs(T)] U as ImplicitAs(Optional(T)) {
  fn Convert[me: T]() -> Optional(T) { return ...; }
}
```
