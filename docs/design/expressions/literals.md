# Literal expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

> **STATUS:** Up-to-date on 2022-Dec-9.

<!-- toc -->

## Table of contents

-   [Numeric literals](#numeric-literals)
    -   [Numeric literal syntax](#numeric-literal-syntax)
    -   [Defined Types](#defined-types)
    -   [Implicit conversions](#implicit-conversions)
    -   [Examples](#examples)
    -   [Alternatives Considered](#alternatives-considered)
-   [Numeric type literals](#numeric-type-literals)
    -   [Meaning](#meaning)
    -   [Usage](#usage)
    -   [Alternatives considered](#alternatives-considered-1)
-   [String literals](#string-literals)
-   [References](#references)

<!-- tocstop -->

## Numeric literals

Numeric Literals are defined on Wikipedia
[here](<https://en.wikipedia.org/wiki/Literal_(computer_programming)>).

In Carbon, numeric literals have a type derived from their value. Two integer
literals have the same type if and only if they represent the same integer. Two
real number literals have the same type if and only if they represent the same
real number.

That is:

-   For every integer, there is a type representing literals with that integer
    value.
-   For every rational number, there is a type representing literals with that
    real value.
-   The types for real numbers are distinct from the types for integers, even
    for real numbers that represent integers. For example, `1 / 2` results in
    `0`, due to integer arithmetic, whereas `1.0 / 2` results in `0.5`. This is
    due to `1` having an integral type, while `1.0` has a real number type, even
    though it represents the same numeric value.

Primitive operators are available between numeric literals, and produce values
with numeric literal types. For example, the type of `1 + 2` is the same as the
type of `3`.

Numeric types can provide conversions to support initialization from numeric
literals. Because the value of the literal is carried in the type, a type-level
decision can be made as to whether the conversion is valid.

The integer types defined in the standard library permit conversion from integer
literal types whose values are representable in the integer type. The
floating-point types defined in the standard library permit conversion from
integer and rational literal types whose values are between the minimum and
maximum finite value representable in the floating-point type.

### Numeric literal syntax

Numeric literal syntax is covered in the
[numeric literal lexical conventions](../lexical_conventions/numeric_literals.md)
doc. Both Integer and Real-Number syntax is defined, with decimal, hexadecimal
and binary integer literals, and decimal and hexadecimal real number literals.

### Defined Types

The following types are defined in the Carbon prelude:

-   `Core.BigInt`, an arbitrary-precision integer type;
-   `Core.Rational(T:! type)`, a rational type, parameterized by a type used for
    its numerator and denominator -- the exact constraints on `T` are not yet
    decided;
-   `Core.IntLiteral(N:! Core.BigInt)`, a type representing integer literals;
    and
-   `Core.FloatLiteral(X:! Core.Rational(Core.BigInt))`, a type representing
    floating-point literals.

All of these types are usable during compilation. `Core.BigInt` supports the
same operations as [`Core.Int(N)`](#meaning). `Core.Rational(T)` supports the
same operations as [`Core.Float(N)`](#meaning).

The types `Core.IntLiteral(N)` and `Core.FloatLiteral(X)` also support primitive
integer and floating-point operations such as arithmetic and comparison, but
these operations are typically heterogeneous: for example, an addition between
`Core.IntLiteral(N)` and `Core.IntLiteral(M)` produces a value of type
`Core.IntLiteral(N + M)`.

### Implicit conversions

`Core.IntLiteral(N)` converts to any sufficiently large integer type, as if by:

```
impl forall [template N:! Core.BigInt, template M:! Core.BigInt]
    Core.IntLiteral(N) as ImplicitAs(Core.Int(M))
    if N >= Core.Int(M).MinValue as Core.BigInt
    and N <= Core.Int(M).MaxValue as Core.BigInt {
  ...
}
impl forall [template N:! Core.BigInt, template M:! Core.BigInt]
    Core.IntLiteral(N) as ImplicitAs(Core.UInt(M))
    if N >= Core.UInt(M).MinValue as Core.BigInt
    and N <= Core.UInt(M).MaxValue as Core.BigInt {
  ...
}
```

The above is for exposition purposes only; various parts of this syntax are not
yet decided.

Similarly, `Core.IntLiteral(X)` and `Core.FloatLiteral(X)` convert to any
sufficiently large floating-point type, and produce the nearest representable
floating-point value.

Conversions in which `X` lies exactly half-way between two values are rounded to
the value in which the mantissa is even, as defined in the IEEE 754 standard and
as was decided in
[proposal #866](https://github.com/carbon-language/carbon-lang/pull/866).

Conversions in which `X` is outside the range of finite values of the
floating-point type are rejected rather than saturating to the finite range or
producing an infinity.

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

fn F[template T:! type](v: T) {
  var x: i32 = v * 2;
}

// OK: x = 2_000_000_000.
F(1_000_000_000);

// Error: 4_000_000_000 can't be represented in type `i32`.
F(2_000_000_000);

// No storage required for the bound when it's of integer literal type.
struct Span(template T:! type, template BoundT:! type) {
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
fn PassMeZero(_: Core.IntLiteral(0));

// Can only be called with integer literals in the given range.
fn ConvertToByte[template N:! Core.BigInt](_: Core.IntLiteral(N)) -> i8
    if N >= -128 and N <= 127 {
  return N as i8;
}

// Given any int literal, produces a literal whose value is one higher.
fn OneHigher(L: Core.IntLiteral(template _:! Core.BigInt)) -> auto {
  return L + 1;
}
// Error: 256 can't be represented in type `i8`.
var v: i8 = OneHigher(255);
```

### Alternatives Considered

-   [Use an ordinary integer or floating-point type for literals](/proposals/p0144.md#use-an-ordinary-integer-or-floating-point-type-for-literals)
-   [Use same type for all literals](/proposals/p0144.md#use-same-type-for-all-literals)
-   [Allow leading `-` in literal tokens](/proposals/p0144.md#allow-leading---in-literal-tokens)
-   [Forbidding floating-point ties](/proposals/p0866.md#alternatives-considered)

## Numeric type literals

Carbon has a simple keyword-like syntax of `iN`, `uN`, and `fN` for two's
complement signed integers, unsigned integers, and
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) floating-point numbers,
respectively. Here, `N` can be a positive multiple of 8, including the common
power-of-two sizes (for example, `N = 8, 16, 32`).

Examples of this syntax include:

-   `i16` - A 16-bit two's complement signed integer type
-   `u32` - A 32-bit unsigned integer type
-   `f64` - A 64-bit IEEE-754 binary floating-point number type

### Meaning

These type literals are aliases for parameterized types defined in the `Core`
package that is automatically imported by the prelude:

-   `iN` is an alias for `Core.Int(N)`
-   `uN` is an alias for `Core.UInt(N)`
-   `fN` is an alias for `Core.Float(N)`

### Usage

```carbon
package sample api;

fn Sum(x: i32, y: i32) -> i32 {
  return x + y;
}

fn Main() -> i32 {
  return Sum(4, 2);
}
```

In the above example, `Sum` has parameters `x` and `y`, each of which is typed
as a 32-bit two's complement signed integer. `Main` then returns the output of
`Sum` as a 32-bit two's complement signed integer.

### Alternatives considered

-   [C++ LP64 convention](/proposals/p2015.md#c-lp64-convention)
-   [Type name with length suffix](/proposals/p2015.md#type-name-with-length-suffix)
-   [Uppercase suffixes](/proposals/p2015.md#uppercase-suffixes)
-   [Additional bit sizes](/proposals/p2015.md#additional-bit-sizes)

## String literals

String literal syntax is covered in the
[string literal lexical conventions](../lexical_conventions/string_literals.md).

No design for string types has been through the proposal process yet.

## References

-   Proposal
    [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
-   Proposal
    [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
-   Question-for-leads issue
    [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
-   Proposal
    [#866: Allow ties in floating literals](https://github.com/carbon-language/carbon-lang/pull/866)
-   Proposal
    [#2015: Numeric type literal syntax](https://github.com/carbon-language/carbon-lang/pull/2015)
-   Question-for-leads issue
    [#2113: Structure, scope, and naming of the prelude and syntax aliases](https://github.com/carbon-language/carbon-lang/issues/2113)
