# `as` expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/845)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Future work](#future-work)
    -   [Provide a mechanism for unsafe conversions](#provide-a-mechanism-for-unsafe-conversions)
        -   [Casting operator for conversions with domain restrictions](#casting-operator-for-conversions-with-domain-restrictions)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow `as` to perform some unsafe conversions](#allow-as-to-perform-some-unsafe-conversions)
    -   [Allow `as` to perform two's complement truncation](#allow-as-to-perform-twos-complement-truncation)
    -   [`as` only performs implicit conversions](#as-only-performs-implicit-conversions)
    -   [Integer to bool conversions](#integer-to-bool-conversions)
    -   [Bool to integer conversions](#bool-to-integer-conversions)

<!-- tocstop -->

## Problem

We would like to provide a notation for the following operations:

-   Requesting a type conversion in order to select an operation to perform, or
    to resolve an ambiguity between possible operations:
    ```
    fn Ratio(a: i32, b: i32) -> f64 {
      // Note that a / b would invoke a different / operation.
      return a / (b as f64);
    }
    ```
-   Specifying the type that an expression will have or will be converted into,
    for documentation purposes.
    ```
    class Thing {
      var id: i32;
    }
    fn PrintThing(t: Thing) {
      // 'as i32' reminds the reader what type we're printing.
      Print(t.id as i32);
    }
    ```
-   Specifying the type that an expression is expected to have, potentially
    after implicit conversions, as a form of static assertion.
    ```
    fn Munge() {
      // I expect this expression to produce a Widget but I'm getting compiler
      // errors and I'd like to narrow down why.
      F(Some().Complex().Expression() as Widget);
    }
    ```

In general, the developer wants to specify that an expression should be
considered to produce a value of a particular type, and that type might be more
general than the type of the expression, the same as the type of the expression,
or perhaps might represent a different way of viewing the value.

The first of the above problems is especially important in Carbon due to the use
of facet types for generics. Explicit conversions of types to interfaces will be
necessary in order to select the meaning of operations, because the same member
name on different facet types for the same underlying type will in general have
different meanings.

For this proposal, the following are out of scope:

-   Requesting a type conversion that changes the value, such as by truncation.
-   Converting a value to a narrower type or determining whether such a
    conversion is possible -- `try_as` or `as?` operations.

## Background

C++ provides a collection of different kinds of casts and conversions from an
expression `x` to a type `T`:

-   Copy-initialization: `T v = x;`
-   Direct-initialization: `T v(x);`
-   Named casts:
    -   `static_cast<T>(x)`
    -   `const_cast<T>(x)`
    -   `reinterpret_cast<T>(x)`
    -   `dynamic_cast<T>(x)`
-   C-style casts: `T(x)` or equivalently `(T)x`
    -   These can do anything that `static_cast`, `const_cast`, and
        `reinterpret_cast` can do, but ignore access control on base classes.
-   List-initialization: `T{x}`
    -   This can do anything that implicit conversion can do, and can also
        initialize a single -- real or notional -- subobject of `T`.
    -   Narrowing conversions are disallowed.

These conversions are all different, and each of them has some surprising or
unsafe behavior.

Swift provides four forms of type casting operation:

-   `x as T` performs a conversion from subtype to supertype.
    -   `pattern as T` in a pattern matching context converts a pattern that
        matches a subtype to a pattern that matches a supertype, by performing a
        runtime type test. This effectively results in a checked supertype to
        subtype conversion.
-   `x as! T` performs a conversion from supertype to subtype, with the
    assumption that the value inhabits the subtype.
-   `x as? T` performs a conversion from supertype to subtype, producing an
    `Optional`.
-   `T(x)` and similar construction expressions are used to convert between
    types without a subtyping relationship, such as between integer and
    floating-point types.

In Swift, `x as T` is always unsurprising and safe.

Rust provides the following:

-   `x as T` performs a conversion to type `T`.
    -   When there is no corresponding value, some specified value is produced:
        this conversion will perform 2's complement truncation on integers and
        will saturate when converting large floating-point values to integers.
    -   Conversions between distinct pointer types, and between pointers and
        integers, are permitted. Rust treats accesses through pointers as
        unsafe, but not pointer arithmetic or casting.

This cast can perform some conversions with surprising results, such as integer
truncation. It can also have surprising performance implications, because it
defines the behavior of converting an out-of-range value -- for example, when
converting from floating-point to integer -- in ways that aren't supported
across all modern targets.

Haskell and Scala support type ascription notation, `x : T`. This has also been
proposed for Rust. This notation constrains the type checker to find a type for
the expression `x` that is consistent with `T`, and is used:

-   for documentation purposes,
-   to guide the type checker to select a particular meaning of the code in the
    presence of ambiguity, and
-   as a diagnostic tool when attempting to understand type inference failures.

## Proposal

Carbon provides a binary `as` operator.

`x as T` performs an unsurprising and safe conversion from `x` to type `T`.

-   This can be used to perform any implicit conversion explicitly. As in Swift,
    this can therefore be used to convert from subtype to supertype.
-   This can also be used to perform an unsurprising and safe conversion that
    cannot be performed implicitly because it's lossy, such as from `i32` to
    `f32`.

This operator does not perform conversions with domain restrictions, such as
converting from `f32` to `i64`, where sufficiently large values can't be
converted. It does not perform operations in which there are multiple different
reasonable interpretations, such as converting from `i64` to `i32`, where a
two's complement truncation might sometimes be reasonable but where the intent
is more likely that it is an error to convert a value that does not fit into an
`i32`.

See changes to the design for details.

## Rationale based on Carbon's goals

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Providing only unsurprising built-in `as` conversions, and encouraging
        user-defined types to do the same, makes code easier to understand.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Syntactically distinguishing between always-safe `as` conversions and
        potentially-unsafe conversions being performed by other syntax makes it
        clearer which code should be the subject of more scrutiny when reasoning
        about safety.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The `As` interface provides the same functionality as single-argument
        `explicit` constructors and `explicit` conversion functions in C++, and
        can be used to expose those operations for interoperability purposes and
        as a replacement for those operations during migration.

## Future work

### Provide a mechanism for unsafe conversions

We need to provide additional conversions beyond those proposed for `as`. In
particular, to supply the same set of conversions as C++, we would need at least
the following conversions that don't match the rules for `as`:

Conversions with a domain restriction:

-   Conversions from pointer-to-supertype to pointer-to-subtype.
-   Conversions from floating-point to integer types that assume the input is
    in-range.
-   (Not in C++.) Conversions between any two integer types that assume the
    input is in-range.

Conversions that modify some values:

-   Truncating conversions between any two integer types.

Conversions that reinterpret values:

-   Conversions between arbitrary pointer types.
-   Conversions between integers and pointers.
-   Bit-casts between arbitrary, sufficiently-trivial types of the same size.

Special cases:

-   Some analogue of `dynamic_cast`.
-   Some analogue of `const_cast`.

We will need to decide which of these we wish to provide -- in particular,
depending on our plans for mutability and RTTI, `const_cast` and `dynamic_cast`
may or may not be appropriate.

For the operations we do supply, we could provide either named functions or
dedicated language syntax. While this proposal suggests that the `as` operator
should not be the appropriate language syntax for the above cases, that decision
should be revisited once we have more information from examining the
alternatives.

#### Casting operator for conversions with domain restrictions

We could provide an additional casting operator, such as `assume_as` or
`unsafe_as`, to model conversions that have a domain restriction, such as
`i64 -> i32` or `f32 -> i64` or `Base*` -> `Derived*`.

Advantage:

-   Provides additional important but unsafe functionality.
-   Gives this functionality the appearance of being a central language feature.
-   Separates safe conversions from unsafe ones.

Disadvantage:

-   Increases complexity.
-   The connection between these conversions may not be obvious, and the kind
    and amount of unsafety in practice differs substantially between them.

If we don't follow this direction, we will need to provide these operations by
another mechanism, such as named function calls.

## Alternatives considered

### Allow `as` to perform some unsafe conversions

We could provide a single type-casting operator that can perform some
conversions that have a domain restriction, treating values out of range as
programming errors.

One particularly appealing option would be to permit `as` to convert freely
between integer and floating-point types, but not permit it to convert from
supertype to subtype.

Advantage:

-   Developers many not want to be reminded about the possibility of overflow in
    conversions to integer types.
-   This would make `as` more consistent with arithmetic operations, which will
    likely have no overt indication that they're unsafe in the presence of
    integer overflow.
-   If we don't do this, then code mixing differently-sized types will need to
    use a syntactic notation other than `as`, even if all conversions remain
    in-bounds. If such code is common, as it is in C++ (for example, when mixing
    `int` and `size_t`), developers may become accustomed to using that "assume
    in range" notation and not consider it to be a warning sign, thereby eroding
    the advantage of using a distinct notation.

Disadvantage:

-   If we allow this conversion, there would be no clear foundation for which
    conversions can be performed by `as` and which cannot in general.
-   An `as` expression would be less suitable for selecting which operation to
    perform if it can be unsafe.
-   Under maintenance, every usage of `as` would need additional scrutiny
    because it's not in general a safe operation.
-   This risks being surprising to developers coming from C and C++ where
    integer type conversions are always safe.

The choice to not provide these operations with `as` is experimental, and should
be revisited when we have more information about the design of integer types and
their behavior.

### Allow `as` to perform two's complement truncation

We could allow `as` to convert between any two integer types, performing a two's
complement conversion between these types.

Advantage:

-   Familiar to developers from C++ and various other systems programming
    languages.

Disadvantage:

-   Makes `as` conversions have behavior that diverges from the behavior of
    arithmetic, where we expect at least signed overflow to be considered a
    programming error rather than being guaranteed to wrap around.
-   Introducing a common and easy notation for conversion with wraparound means
    that this notation will also be used in the -- likely much more common --
    case of wanting to truncate a value that is already known to be in-bounds.
    Compared to having distinct notation for these two operations:
    -   This removes the ability to distinguish between programming errors due
        to overflow and intentional wraparound by using the same syntax for
        both, both for readers of the code and for automated checks in debugging
        builds.
    -   This removes the ability to optimize on the basis of knowing that a
        value is expected to be in-bounds when performing a narrowing
        conversion.

The choice to not provide these operations with `as` is experimental, and should
be revisited when we have more information about the design of integer types and
their behavior.

### `as` only performs implicit conversions

We could limit `as` to performing only implicit conversions. This would mean
that `as` cannot perform lossy conversions.

Advantage:

-   One fewer set of rules for developers to be aware of.

Disadvantage:

-   Converting between integer and floating-point types is common, and providing
    built-in syntax for it seems valuable.

### Integer to bool conversions

We could allow a conversion of integer types (and perhaps even floating-point
types) to `bool`, converting non-zero values to `true` and converting zeroes to
`false`.

Advantage:

-   This treatment of non-zero values as being "truthy" and zero values as being
    "falsy" is familiar to developers of various other languages.
-   Uniform treatment of types that can be notionally converted to a Boolean
    value may be useful in templates and generics in some cases.

Disadvantage:

-   The lossy treatment of all non-zero values as being "truthy" is somewhat
    arbitrary and can be confusing.
-   An `as bool` conversion is less clear to a reader than a `!= 0` test.
-   An `as bool` conversion is more verbose than a `!= 0` test.

### Bool to integer conversions

We could disallow conversions from `bool` to `iN` types.

Advantage:

-   More clearly demarcates the intended semantics of `bool` as a truth value
    rather than as a number.
-   Avoids making a choice as to whether `true` should map to 1 (zero-extension)
    or -1 (sign-extension).
    -   But there is a strong established convention of using 1.
-   Such conversions are a known source of bugs, especially when performed
    implicitly. `as` conversions will likely be fairly common and routine in
    Carbon code due to their use in generics. As such, they may be written
    without much thought and not given much scrutiny in code review.
    ```
    var found: bool = false;
    var total_found: i32 = 0;
    for (var (key: i32, value: i32) in list) {
      if (key == expected) {
        found = true;
        total_found += value;
      }
    }
    // Include an explicit `as i64` to emphasize that we're widening the
    // total at this point.
    // Bug: meant to pass `total_found` not `found` here.
    add_to_total(found as i64);
    ```

Disadvantage:

-   Removes a sometimes-useful operation for which there isn't a similarly terse
    alternative expression form.
    -   But we could add a member function `b.AsBit()` if we wanted.
-   Does not expose the intended connection between the `bool` type and bits.

We could disallow conversion from `bool` to `i1`.

Advantage:

-   Avoids a surprising behavior where this conversion converts `true` to -1
    whereas all others convert `true` to 1.

Disadvantage:

-   Results in non-uniform treatment of conversion from `bool`, and an awkward
    special case that may get in the way of generics.
-   A conversion from `bool` that produces -1 for a `true` value is useful when
    producing a mask, for example in `(b as i1) as u32`.
