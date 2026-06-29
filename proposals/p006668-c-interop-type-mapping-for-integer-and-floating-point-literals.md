# C++ interop type mapping for integer and floating-point literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6668)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [C++ literals](#c-literals)
        -   [Integer literals](#integer-literals)
            -   [Type of C++ integer literals](#type-of-c-integer-literals)
            -   [Clang](#clang)
            -   [Gcc](#gcc)
        -   [Floating-point literals](#floating-point-literals)
    -   [Carbon literals](#carbon-literals)
        -   [Integer literals](#integer-literals-1)
        -   [Floating-point literals](#floating-point-literals-1)
        -   [Literal types](#literal-types)
        -   [No suffixes](#no-suffixes)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Carbon to C++ type mapping](#carbon-to-c-type-mapping)
        -   [Integer literals](#integer-literals-2)
        -   [Floating-point literals](#floating-point-literals-2)
    -   [C++ to Carbon type mapping](#c-to-carbon-type-mapping)
        -   [Integer literals](#integer-literals-3)
        -   [Floating-point literals](#floating-point-literals-3)
-   [Future work](#future-work)
    -   [More robust hexadecimal and binary literals in Carbon](#more-robust-hexadecimal-and-binary-literals-in-carbon)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Carbon to C++ type mapping](#carbon-to-c-type-mapping-1)
        -   [Use the C++ standard rules for hexadecimal and binary literals](#use-the-c-standard-rules-for-hexadecimal-and-binary-literals)
        -   [Use the Clang way of determining the type](#use-the-clang-way-of-determining-the-type)
    -   [C++ to Carbon type mapping](#c-to-carbon-type-mapping-1)
        -   [Use Carbon literal types](#use-carbon-literal-types)

<!-- tocstop -->

## Abstract

Provides bidirectional mappings for types of integer and floating-point literals
between Carbon and C++. For example, given a literal `123`, defines the interop
type.

## Problem

This document addresses Carbon &lt;-> C++ type mapping of integer and floating
literals. This comes to use for example during overloading resolution of C++
functions, when a C++ function is called from Carbon with a literal as a call
argument. It also appears for example when importing C++ macros to Carbon for
example macros which define constants where the constant is a literal.

## Background

### C++ literals

#### Integer literals

##### Type of C++ integer literals

As specified in the
[C++ standard](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/n4950.pdf#page=31&zoom=100,85,693),
the type of the literal is the first type from the list below in which the value
can fit based on its **suffix** and the **base** of the literal.

| Suffix                                  | Decimal bases                                     | Binary, octal or hexadecimal bases                                            |
| --------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| (none)                                  | `int`, `long`, `long long`                        | `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long` |
| `u` or `U`                              | `unsigned`, `unsigned long`, `unsigned long long` | `unsigned`, `unsigned long`, `unsigned long long`                             |
| `l` or `L`                              | `long`, `long long`                               | `long`, `unsigned long`, `long long`, `unsigned long long`                    |
| `u` or `U` and `l` or `L`               | `unsigned long`, `unsigned long long`             | `unsigned long`, `unsigned long long`                                         |
| `ll` or `LL`                            | `long long`                                       | `long long`, `unsigned long long`                                             |
| `u` or `U` and `ll` or `LL`             | `unsigned long long`                              | `unsigned long long`                                                          |
| `z` or `Z` (since C++23)                | the signed version of `std::size_t`               | the signed version of `std::size_t`, `std::size_t`                            |
| `u` or `U` and `z` or `Z` (since C++23) | `std::size_t`                                     | `std::size_t`                                                                 |

For details on `std::size_t`, see
[cppreference](https://en.cppreference.com/w/cpp/types/size_t).

If the value is too big to fit in any of these types, and an extended integer
type exists, then it may be assigned an extended type. If all types are signed,
then it may fit into a signed extended type. If all types are unsigned then it
may be fitted into an unsigned extended type. If there are both signed and
unsigned types in the list, then both signed and unsigned extended types are
possible types. If the value can’t fit in any of the types, the program is
ill-formed.

##### Clang

Clang diverges from the standard in that if the decimal integer literal doesn’t
fit to a type from the list, instead of assigning an extended integer type, it
assigns `unsigned long long`.

Example:

```cpp
#include <iostream>

inline auto foo(long a) -> void { printf("hello from foo_long(%ld)", a); }
inline auto foo(unsigned long a) -> void { printf("hello from foo_unsigned_long(%lu)", a); }

inline auto foo(long long a) -> void { printf("hello from foo_long_long(%lld)", a); }
inline auto foo(unsigned long long a) -> void { printf("hello from foo_unsigned_long_long(%llu)", a); }

inline auto foo(__int128 a) -> void { printf("hello from foo___int128");}

int main() {
   foo(9223372036854775808);  // MAX_LONG + 1
   return 0;
}
```

Output:

```
<source>:12:9: warning: integer literal is too large to be represented in a signed integer type, interpreting as unsigned [-Wimplicitly-unsigned-literal]
  12 |     foo(9223372036854775808);
     |        ^
1 warning generated.
Execution build compiler returned: 0
Program returned: 0
hello from foo_unsigned_long_long(9223372036854775808)
```

##### Gcc

Gcc shows a warning that the value will be treated as an unsigned integer, but
it actually assigns the type `__int128` to it.

Output of the example above:

```
<source>:12:9: warning: integer constant is so large that it is unsigned
  12 |     foo(9223372036854775808);
     |         ^~~~~~~~~~~~~~~~~~~
Execution build compiler returned: 0
Program returned: 0
hello from foo___int128
```

#### Floating-point literals

The type of a floating literal is `double` unless explicitly specified by a
suffix. When there is a suffix, then the suffix determines the type.

| Suffix                         | Floating-point literal type |
| ------------------------------ | --------------------------- |
| (none)                         | `double`                    |
| `f` or `F`                     | `float`                     |
| `l` or `L`                     | `long double`               |
| `f16` or `F16` (since C++23)   | `std::float16_t`            |
| `f32` or `F32` (since C++23)   | `std::float32_t`            |
| `f64` or `F64` (since C++23)   | `std::float64_t`            |
| `f128` or `F128` (since C++23) | `std::float128_t`           |
| `bf16` or `BF16` (since C++23) | `std::bfloat16_t`           |

If the value doesn’t fit the type, the program is ill-formed.

### Carbon literals

#### Integer literals

Carbon has **decimal, hexadecimal and binary** integer literals. Example:

-   `123` (decimal)
-   `0x1FE` (hexadecimal)
-   `0b10` (binary)

There are no suffixes for the integer literal types.

#### Floating-point literals

Carbon supports **decimal and hexadecimal** floating-point literals. Example:

1.  Decimal:

     -   `123.456`
     -   `1.23456e791`

2.  Hexadecimal:
     -   `0x1.Ap123`

#### Literal types

Carbon has literal types, currently `Core.IntLiteral` and `Core.FloatLiteral`.
These are convertible to any type where they fit without any truncation or loss
of precision.

At present, type deduction would retain the type of the literal. For example,
`let x: auto = 1;` would result in `x` having type `Core.IntLiteral`. This is
not the desired behavior, and so can be expected to change in the future.

#### No suffixes

Carbon literals have no suffix. The reasons for this are covered in
[Proposal #144's alternative "Use an ordinary integer or floating-point type for literals"](p000144-numeric-literal-semantics.md#use-an-ordinary-integer-or-floating-point-type-for-literals).

## Proposal

-   **Carbon literal to C++ type**: A Carbon literal will be given a C++ type
    according to the C++ rules when used in C++ context, such as calling a C++
    function.
-   **C++ literal to Carbon type**: A C++ literal will be given a C++ type
    following the C++ rules, which will then be mapped to a Carbon type as
    defined in
    [primitive types mapping](p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).
    -   For example, `1` becomes `int` according to C++ rules, and `int` maps to
        `i32` in Carbon.

## Details

### Carbon to C++ type mapping

#### Integer literals

There are no suffixes in Carbon for the integer types, so a Carbon decimal
integer literal will follow the C++ rules for decimal integers. Carbon also
doesn't make a distinction between hexadecimal and binary literals, so these
literals will also follow the C++ rules for decimal integers without a suffix.

The first type from this list in which the value can fit will be selected:

-   `int`
-   `long`
-   `long long`
-   `__int128`

If the value doesn’t fit in any of these types, the program will be ill-formed
and diagnosed with an error. This is intended to match the C++ standard
behavior, instead of Clang's `-Wimplicitly-unsigned-literal` behavior.

Decimal numbers are most commonly used as integer literals, so this should match
most of the existing C++ calls. To match the C++ calls with a non-decimal
literal argument, an explicit `unsigned` type will need to be provided. For
example, `0xDEADBEEF as u32`.

#### Floating-point literals

As there are no suffixes in Carbon for the floating-point literals, a Carbon
floating-point literal will map to C++ `double`. If the value doesn’t fit to
double, the program is ill-formed.

### C++ to Carbon type mapping

#### Integer literals

A C++ integer literal will be given a C++ integer type following the C++ rules,
which will then be mapped to a Carbon type as defined in
[primitive types mapping](p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).

| C++ literal suffix                      | Carbon type with decimal C++ integer literals                     | Carbon type with hexadecimal, binary and octal C++ integer literals                                       |
| --------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| (none)                                  | `Cpp.int`, `Cpp.long`, `Cpp.long_long`                            | `Cpp.int`, `Cpp.unsigned_int`, `Cpp.long`, `Cpp.unsigned_long`, `Cpp.long_long`, `Cpp.unsigned_long_long` |
| `u` or `U`                              | `Cpp.unsigned_int`, `Cpp.unsigned_long`, `Cpp.unsigned_long_long` | `Cpp.unsigned_int`, `Cpp.unsigned_long`, `Cpp.unsigned_long_long`                                         |
| `l` or `L`                              | `Cpp.long`, `Cpp.long_long`                                       | `Cpp.long`, `Cpp.unsigned_long`, `Cpp.long_long`, `Cpp.unsigned_long_long`                                |
| `u` or `U` and `l` or `L`               | `Cpp.unsigned_long`, `Cpp.unsigned_long_long`                     | `Cpp.unsigned_long`, `Cpp.unsigned_long_long`                                                             |
| `ll` or `LL`                            | `Cpp.long_long`                                                   | `Cpp.long_long`, `Cpp.unsigned_long_long`                                                                 |
| `u` or `U` and `ll` or `LL`             | `Cpp.unsigned_long_long`                                          | `Cpp.unsigned_long_long`                                                                                  |
| `z` or `Z` (since C++23)                | `Cpp.uintptr_t`                                                   | `Cpp.uintptr_t`, `Cpp.size_t`                                                                             |
| `u` or `U` and `z` or `Z` (since C++23) | `Cpp.size_t`                                                      | `Cpp.size_t`                                                                                              |

#### Floating-point literals

A C++ floating literal will be given a C++ type following the C++ rules, which
will then be mapped to a Carbon type as defined in
[primitive types mapping](p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).
That means:

| C++ floating-point literal suffix | Carbon floating-point literal type |
| --------------------------------- | ---------------------------------- |
| (none)                            | `f64`                              |
| `f` or `F`                        | `f32`                              |
| `l` or `L`                        | `Cpp.long_double`                  |
| `f16` or `F16` (since C++23)      | `f16`                              |
| `f32` or `F32` (since C++23)      | TBD                                |
| `f64` or `F64` (since C++23)      | TBD                                |
| `f128` or `F128` (since C++23)    | `f128`                             |
| `bf16` or `BF16` (since C++23)    | TBD                                |

## Future work

### More robust hexadecimal and binary literals in Carbon

We may want to use C++ standard rules for Carbon's hexadecimal and binary
literals, per the
[alternative below](#use-the-c-standard-rules-for-hexadecimal-and-binary-literals).
However, that requires addressing how that should be handled in the integer
literal space; for example, either adding more information to `Core.IntLiteral`
or a new type, as well as addressing what kind of type is produced when
performing arithmetic on mixed literal types. That may be desirable, but
addressing it here would substantially increase the scope of this proposal.

## Rationale

This work will contribute to Carbon’s goal for seamless interoperability with
C++
([Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)),
by keeping the consistency with the existing C++ usage.

## Alternatives considered

### Carbon to C++ type mapping

#### Use the C++ standard rules for hexadecimal and binary literals

For example, when dealing with `0xFFFF_FFFF`, we could make that be interpreted
as `unsigned` instead of `long`. This will be discernible in cases of overload
resolution and type deduction.

For example:

```cpp
import Cpp inline '''
void f(unsigned);

void g(unsigned);
void g(long);

template<typename T> void h(T);
''';

fn CallF() {
  // OK in both Carbon and C++.
  // We select the f(unsigned) overload, and can call it.
  Cpp.f(0xFFFF_FFFF);
}

fn CallG() {
  // Would call g(unsigned) in C++.
  // Calls g(long) in Carbon.
  Cpp.g(0xFFFF_FFFF);
}

fn CallH() {
  // Would call h<unsigned> in C++.
  // Calls h<long> in Carbon.
  Cpp.h(0xFFFF_FFFF);
}
```

Advantages:

-   This would allow using the unsigned types for the cases where it’s important
    (for example bit manipulation). This affects understandability for
    interactions on interop boundaries, where developers might reasonably expect
    that `0xFFFF_FFFF` would behave identically in Carbon and C++.

Disadvantages:

-   The current integer literal design doesn't support this distinction.
-   These cases seem to be less often used than the decimal literals.

This decision is probably good enough for now, although we should consider it
for [future work](#more-robust-hexadecimal-and-binary-literals-in-carbon).

#### Use the Clang way of determining the type

As described in [background](#clang), we could use the biggest unsigned integer
type currently supported for C++ (`unsigned __int128`) if the value doesn’t fit
to `int`, `long`, or `long long`.

Advantages:

-   This will match the C++ calls for the users that use Clang.

Disadvantages:

-   Using `__int128` gives a consistently signed type interpretation regardless
    of bit width. This is also consistent to how we plan to require an explicit
    `unsigned` conversion.

It's not clear how much this issue would arise, so we believe a consistent
signed type interpretation is a good default.

### C++ to Carbon type mapping

#### Use Carbon literal types

In other words, import a literal to a Carbon literal type; `Core.IntLiteral` or
`Core.FloatLiteral`.

Advantages:

-   This would allow determining the type later according to the Carbon rules.

Disadvantages:

-   This may not be always feasible. For example, sometimes a variable or
    constant needs to be initialized with the value of the literal.

Rather than trying to detect feasibility, we are choosing to assign a sized
numeric type.
