# Numeric Literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Defined Types](#defined-types)
    -   [Integers](#integers)
    -   [Floats](#floats)
    -   [Implicit conversions](#implicit-conversions)
    -   [Examples](#examples)
-   [Numeric literal semantics](#numeric-literal-semantics)
-   [Open questions](#open-questions)
    -   [Primitive types as code vs built-in](#primitive-types-as-code-vs-built-in)
    -   [Syntax for wrapping operations](#syntax-for-wrapping-operations)
    -   [Non-power-of-two sizes](#non-power-of-two-sizes)
-   [References](#references)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Numeric literals are used to define a numeric value in a program. This value can
be of any type, such as an integral type or a real number type. Numeric literals
have a type derived from their value. Two integer literals have the same type if
and only if they represent the same integer. Two real number literals have the
same type if and only if they represent the same real number.

That is:

-   For every integer, there is a type representing literals with that integer
    value.
-   For every rational number, there is a type representing literals with that
    real value.
-   The types for real numbers are distinct from the types for integers, even
    for real numbers that represent integers. `var x: i32 = 1.0;` is invalid.

Primitive operators are available between numeric literals, and produce values
with numeric literal types. For example, the type of `1 + 2` is the same as the
type of `3`.

Numeric types can provide conversions to support initialization from numeric
literals. Because the value of the literal is carried in the type, a type-level
decision can be made as to whether the conversion is valid.

The integer types defined in the standard library permit conversion from integer
literal types whose values are representable in the integer type. The
floating-point types defined in the Carbon library permit conversion from
integer and rational literal types whose values are between the minimum and
maximum finite value representable in the floating-point type.

### Defined Types

The following types are defined in the Carbon prelude:

-   An arbitrary-precision integer type.

    ```
    class BigInt;
    ```

-   A rational type, parameterized by a type used for its numerator and
    denominator.

    ```
    class Rational(T:! Type);
    ```

    The exact constraints on `T` are not yet decided.

-   A type representing integer literals.

    ```
    class IntLiteral(N:! BigInt);
    ```

-   A type representing floating-point literals.

        ```
        class FloatLiteral(X:! Rational(BigInt));
        ```

    All of these types are usable during compilation. `BigInt` supports the same
    operations as `Int(n)`. `Rational(T)` supports the same operations as
    `Float(n)`.

The types `IntLiteral(n)` and `FloatLiteral(x)` also support primitive integer
and floating-point operations such as arithmetic and comparison, but these
operations are typically heterogeneous: for example, an addition between
`IntLiteral(n)` and `IntLiteral(m)` produces a value of type
`IntLiteral(n + m)`.

### Integers

Integer types can be either signed or unsigned, much like in C++. Signed
integers are represented using 2's complement and notionally modeled as
unbounded natural numbers. Signed overflow in either direction is an error.
Specific sizes are available, for example: `i8`, `u16`, `i32`, and `u128`.

There is an upper bound on the size of an integer, most likely initially set to
128 bits due to LLVM limitations.

### Floats

Floating point types are based on the binary floating point formats provided by
IEEE-754. `f16`, `f32`, `f64` and, if available, `f128` correspond exactly to
those sized IEEE-754 formats, and have the semantics defined by IEEE-754.

### Implicit conversions

`IntLiteral(n)` converts to any sufficiently large integer type, as if by:

```
impl [template N:! BigInt, template M:! BigInt]
    IntLiteral(N) as ImplicitAs(Int(M))
    if N >= Int(M).MinValue as BigInt and N <= Int(M).MaxValue as BigInt {
  ...
}
impl [template N:! BigInt, template M:! BigInt]
    IntLiteral(N) as ImplicitAs(Unsigned(M))
    if N >= Int(M).MinValue as BigInt and N <= Int(M).MaxValue as BigInt {
  ...
}
```

The above is for exposition purposes only; various parts of this syntax are not
yet decided.

Similarly, `IntLiteral(x)` and `FloatLiteral(x)` convert to any sufficiently
large floating-point type, and produce the nearest representable floating-point
value. Conversions in which `x` lies exactly half-way between two values are
rejected, as
[previously decided](/docs/design/lexical_conventions/numeric_literals.md).
Conversions in which `x` is outside the range of finite values of the
floating-point type are also represented, rather than saturating to the finite
range or producing an infinity.

### Examples

```carbon
// This is OK: the initializer is of the integer literal type with value
// -2147483648 despite being written as a unary `-` applied to a literal.
var x: i32 = -2147483648;

// This initializes y to 2^60.
var y: i64 = 1 << 60;

// This forms a rational literal whose value is one third, and converts it to
// the nearest representable value of type `f64`.
var z: f64 = 1.0 / 3.0;

// This is an error: 300 cannot be represented in type `i8`.
var c: i8 = 300;

fn f[template T:! Type](v: T) {
  var x: i32 = v * 2;
}

// OK: x = 2_000_000_000.
f(1_000_000_000);

// Error: 4_000_000_000 can't be represented in type `i32`.
f(2_000_000_000);

// No storage required for the bound when it's of integer literal type.
struct Span(template T:! Type, template BoundT:! Type) {
  var begin: T*;
  var bound: BoundT;
}

// Returns 1, because 1.3 can implicitly convert to f32, even though conversion
// to f64 might be a more exact match.
fn G() -> i32 {
  match (1.3) {
    case _: f32 => { return 1; }
    case _: f64 => { return 2; }
  }
}

// Can only be called with a literal 0.
fn PassMeZero(_: IntLiteral(0));

// Can only be called with integer literals in the given range.
fn ConvertToByte[template N:! BigInt](_: IntLiteral(N)) -> i8
    if N >= -128 and N <= 127 {
  return N as i8;
}

// Given any int literal, produces a literal whose value is one higher.
fn OneHigher(L: IntLiteral(template _:! BigInt)) -> auto {
  return L + 1;
}
// Error: 256 can't be represented in type `i8`.
var v: i8 = OneHigher(255);
```

## Numeric literal semantics

Numeric Literal semantics are covered in the
[numeric_literals](lexical_conventions/numeric_literals.md) lexical conventions
doc. Decimal and Real-Number semantics are defined, with decimal, hexadecimal
and binary integer literals, and decimal and hexadecimal real number literals.

## Open questions

### Primitive types as code vs built-in

There are open questions about the extent to which these types should be defined
in Carbon code rather than special. Clearly they can't be directly implemented
w/o help, but it might still be useful to force the programmer-observed
interface to reside in code. However, this can cause difficulty with avoiding
the need to import things gratuitously.

### Syntax for wrapping operations

Open question around allowing special syntax for wrapping operations (even on
signed types) and/or requiring such syntax for wrapping operations on unsigned
types.

### Non-power-of-two sizes

Supporting non-power-of-two sizes is likely needed to have a clean model for
bitfields, but requires more details to be worked out around memory access.

## References

> -   Proposal
>     [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
> -   Proposal
>     [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
> -   [#2015: Numeric type literal syntax](https://github.com/carbon-language/carbon-lang/pull/2015)
