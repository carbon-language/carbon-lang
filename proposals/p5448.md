# Carbon <-> C++ Interop: Primitive Types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5448)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Data models](#data-models)
    -   [Carbon Primitive Types](#carbon-primitive-types)
    -   [C++ Fundamental Types](#c-fundamental-types)
        -   [void](#void)
        -   [std::nullptr_t](#stdnullptr_t)
        -   [std::byte](#stdbyte)
        -   [Character types](#character-types)
        -   [Signed integer types](#signed-integer-types)
        -   [Unsigned integer types](#unsigned-integer-types)
        -   [Floating-point types](#floating-point-types)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [C++ -> Carbon mapping details](#c---carbon-mapping-details)
    -   [Carbon -> C++ mapping details](#carbon---c-mapping-details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
-   [Open questions](#open-questions)

<!-- tocstop -->

## Abstract

Define the type mapping of the primitive types between Carbon and C++.

## Problem

Interoperability of Carbon with C++ is one of the Carbon language goals (see
[Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)).
Providing
[unsurprising mappings between C++ and Carbon types](/docs/design/interoperability/philosophy_and_goals.md#unsurprising-mappings-between-c-and-carbon-types)
is one of it's sub goals.

This proposal addresses the type mapping between the two languages to support
achieving this goal.

## Background

### Data models

The following data models are widely accepted:

-   32-bit systems:
    -   `LP32` (Win16 API): `int` 16-bit; `long` 32-bit; `pointer` 32-bit.
    -   `ILP32` (Win32 API; Unix and Unix-like systems): `int` 32-bit; `long`
        32-bit; `pointer` 32-bit.
-   64-bit systems:
    -   `LLP64` (Win32 API: 64-bit ARM or x86-64): `int` 32-bit; `long` 32-bit;
        `pointer` 64-bit.
    -   `LP64` (Unix and Unix-like systems (Linux, macOS)): `int` 32-bit; `long`
        64-bit; `pointer` 64-bit.

[Carbon supported platforms](/docs/project/principles/success_criteria.md#modern-os-platforms-hardware-architectures-and-environments)

Carbon will prioritize supporting modern OS, 64-bit little endian platforms (for
example [LLP64](/proposals/p5448.md#data-models),
[LP64](/proposals/p5448.md#data-models)). Historic platforms like
[LP32](/proposals/p5448.md#data-models) won't be supported.

For clarity, the text below omits [LP32](/proposals/p5448.md#data-models)
relevant information and focuses only on the Carbon supported platforms.

### Carbon Primitive Types

Carbon has the following
[primitive types](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/README.md#primitive-types):

-   `bool`: boolean type taking `true` or `false`
-   integer types:
    -   signed integer types: `iN` (`N` - bit width, a positive multiple of 8)
        -   `i8`, `i16`, `i32`, `i64`, `i128`, `i256`
    -   unsigned integer types: `uN` (`N` - bit width, a positive multiple of 8)
        -   `u8`, `u16`, `u32`, `u64`, `u128`, `u256`
-   floating-point types: `fN` (`N` - bit width, a positive multiple of 8),
    IEEE-754 format
    -   `f16`, `f32`, and `f64` - always available
    -   `f80`, `f128`, or `f256` may be available, depending on the platform

### C++ Fundamental Types

C++ calls the primitive types
[fundamental types](https://en.cppreference.com/w/cpp/language/types). The
following fundamental types exist in C++:

-   `void`
-   `std::nullptr_t`
-   `std::byte`
-   integral types (also integer types):
    -   `bool`
    -   character types:
        -   narrow character types: `signed char`, `unsigned char`, `char`,
            `char8_t` (c++20)
        -   wide character types: `char16_t`, `char32_t`, `wchar_t`
    -   signed integer types:
        -   standard signed integer types: `signed char`, `short`, `int`,
            `long`, `long long`
        -   extended signed integer types (implementation-defined)
    -   unsigned integer types:
        -   standard unsigned integer types: `unsigned char`, `unsigned short`,
            `unsigned int`, `unsigned long`, `unsigned long long`
        -   extended unsigned integer types
-   floating-point types:
    -   standard floating-point types: `float`, `double`, `long double`
    -   extended floating-point types:
        -   fixed width floating-point types (since C++23): `float16_t`,
            `float32_t`, `float64_t`, `float128_t`, `bfloat16_t`
        -   other implementation-defined extended floating-point types

#### void

Objects of type `void` are not allowed, neither are arrays of `void`, nor
references to `void`. Pointers to `void` and functions returning `void` are
allowed.

#### std::nullptr_t

The type of `nullptr` (the null pointer literal). It's a distinct type that is
not itself a pointer type.

#### std::byte

| Type        | Width in bits | Notes                                                                                                                  |
| ----------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `std::byte` | 8-bit         | can be used to access raw memory, same as `unsigned char`, but it's not a character type and is not an arithmetic type |

#### Character types

| Type            | Width in bits                      | Notes                                                                                                                                                         |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `char`          | 8-bit                              | multibyte characters; same representation, alignment and signedness as either `signed char` or `unsigned char` (platform-dependent), but it's a distinct type |
| `signed char`   | 8-bit                              | signed character representation                                                                                                                               |
| `unsigned char` | 8-bit                              | unsigned character representation; raw memory access                                                                                                          |
| `char8_t`       | 8-bit                              | UTF-8 character representation; same size, alignment and signedness as `unsigned char`, but a distinct type                                                   |
| `char16_t`      | 16-bit                             | UTF-16 character representation; same size, alignment and signedness as `std::uint_least16_t`, but a distinct type                                            |
| `char32_t`      | 32-bit                             | UTF-32 character representation; same size, alignment and signedness as `std::uint_least32_t`, but a distinct type                                            |
| `wchar_t`       | 32-bit on Linux, 16-bit on Windows | wide character representation, holds UTF-32 on Linux and other non-Windows platforms, UTF-16 on Windows.                                                      |

#### Signed integer types

**Standard signed integer types**

| Type          | Width in bits                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `signed char` | 8-bit                                                                                             |
| `short`       | 16-bit                                                                                            |
| `int`         | 32-bit                                                                                            |
| `long`        | [LLP64](/proposals/p5448.md#data-models): 32-bit; [LP64](/proposals/p5448.md#data-models): 64-bit |
| `long long`   | 64-bit                                                                                            |

**Exact-width integer types**

Typically aliases of the standard integer types.

| Type           | Width in bits | Defined as                                                                   |
| -------------- | ------------- | ---------------------------------------------------------------------------- |
| `std::int8_t`  | 8-bit         | `typedef signed char int8_t`                                                 |
| `std::int16_t` | 16-bit        | `typedef signed short int16_t`                                               |
| `std::int32_t` | 32-bit        | `typedef signed int int32_t`                                                 |
| `std::int64_t` | 64-bit        | [LLP64](/proposals/p5448.md#data-models): `typedef signed long long int64_t` |
|                |               | [LP64](/proposals/p5448.md#data-models): `typedef signed long int64_t`       |

**Fastest minimum-width integer types**

Integer types that are usually fastest to operate with among all integer types
that have the minimum specified width.

| Type                | Width in bits | Defined as                                                                        |
| ------------------- | ------------- | --------------------------------------------------------------------------------- |
| `std::int_fast8_t`  | >=8-bit       | `typedef signed char int_fast8_t`                                                 |
| `std::int_fast16_t` | >=16-bit      | implementation dependent                                                          |
| `std::int_fast32_t` | >=32-bit      | implementation dependent                                                          |
| `std::int_fast64_t` | >=64-bit      | [LLP64](/proposals/p5448.md#data-models): `typedef signed long long int_fast64_t` |
|                     |               | [LP64](/proposals/p5448.md#data-models): `typedef signed long int_fast64_t`       |

**Minimum-width integer types**

Smallest signed integer type with width of at least N-bits.

| Type                 | Width in bits | Defined as                                                                         |
| -------------------- | ------------- | ---------------------------------------------------------------------------------- |
| `std::int_least8_t`  | >=8-bit       | `typedef signed char int_least8_t`                                                 |
| `std::int_least16_t` | >=16-bit      | `typedef short int_least16_t`                                                      |
| `std::int_least32_t` | >=32-bit      | `typedef int int_least32_t`                                                        |
| `std::int_least64_t` | >=64-bit      | [LLP64](/proposals/p5448.md#data-models): `typedef signed long long int_least64_t` |
|                      |               | [LP64](/proposals/p5448.md#data-models): `typedef signed long int_least64_t`       |

**Greatest-width integer types**

Maximum-width signed integer type.

| Type            | Width in bits | Defined as                                                                    |
| --------------- | ------------- | ----------------------------------------------------------------------------- |
| `std::intmax_t` | >=32-bit      | [LLP64](/proposals/p5448.md#data-models): `typedef signed long long intmax_t` |
|                 |               | [LP64](/proposals/p5448.md#data-models): `typedef signed long intmax_t`       |

**Integer types capable of holding object pointers**

Signed integer type, capable of holding any pointer.

| Type            | Width in bits | Defined as                                                           |
| --------------- | ------------- | -------------------------------------------------------------------- |
| `std::intptr_t` | >=16-bit      | most platforms: `typedef long intptr_t`                              |
|                 |               | some [ILP32](/proposals/p5448.md#data-models):`typedef int intptr_t` |

**Other signed integer types**

| Type        | Width in bits | Defined as                                        |
| ----------- | ------------- | ------------------------------------------------- |
| `ptrdiff_t` | >=16-bit      | most platforms: `typedef std::intptr_t ptrdiff_t` |
|             |               | Holds the result of subtracting two pointers.     |

#### Unsigned integer types

The unsigned integer types have the same sizes as their
[signed counterparts](/proposals/p5448.md#signed-integer-types).

| Type     | Width in bits | Defined as                                 |
| -------- | ------------- | ------------------------------------------ |
| `size_t` | >=16-bit      | most platforms: `typedef uintptr_t size_t` |
|          |               | Holds the result of the `sizeof` operator. |

#### Floating-point types

**Standard floating-point types**

| Type          | Format                                                                                                              | Width in bits    | Note                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------------------------------------------------------------- |
| `float`       | usually [IEEE-754 binary32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)                   | 32-bits          | The format or the size can vary depending on the compiler and the platform. |
| `double`      | usually [IEEE-754 binary64](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)                   | 64-bits          | The format or the size can vary depending on the compiler and the platform. |
| `long double` | [IEEE-754 binary128](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format)                       | 128-bit          | used by some SPARC, MIPS, ARM64 implementations.                            |
|               | [IEEE-754 binary64-extended format](https://en.wikipedia.org/wiki/Extended_precision)                               | 80-bit or 64-bit | 80-bit (most x86 and x86-64 implementations); 64-bit used by MSVC.          |
|               | [`double-double`](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) | 128-bit          | used on PowerPC.                                                            |

**Fixed-width floating-point types (C++23)**

They aren’t aliases to the standard floating-point types (`float`, `double`,
`long double`), but to an extended floating-point type.

| Type              | Width in bits | Defined as                     |
| ----------------- | ------------- | ------------------------------ |
| `std::float16_t`  | 16-bit        | `using float16_t = _Float16`   |
| `std::float32_t`  | 32-bit        | `using float32_t = _Float32`   |
| `std::float64_t`  | 64-bit        | `using float64_t = _Float64`   |
| `std::float128_t` | 128-bit       | `using float128_t = _Float128` |
| `std::bfloat16_t` | 16-bit        |                                |

## Proposal

-   The C++ fixed-width integer types `intN_t` will be the same type as Carbon
    integer types `iN`. Likewise for `uintN_t` <-> `uN`.
-   A C++ `builtin type` will be available in Carbon as `Cpp.builtin_type`, for
    the standard C++ signed/unsigned integer and floating-point types.
-   A C++ integer `builtin type` that is not the same as `intN_t` or `uintN_t`
    for any N, will be nameable in Carbon only as `Cpp.builtin_type`.
-   Different C++ types will be considered different in Carbon, so C++ overload
    resolution can be handled without issues.

## Details

The table of Carbon <-> C++ mappings is as follows:

| Carbon                   | C++                                    |
| ------------------------ | -------------------------------------- |
| `()` as a return type    | `void`                                 |
| `bool`                   | `bool`                                 |
| `i8`                     | `int8_t`                               |
| `i16`                    | `int16_t`                              |
| `i32`                    | `int32_t`                              |
| `i64`                    | `int64_t`                              |
| `i128`                   | `int128_t`                             |
| `u8`                     | `uint8_t`                              |
| `u16`                    | `uint16_t`                             |
| `u32`                    | `uint32_t`                             |
| `u64`                    | `uint64_t`                             |
| `u128`                   | `uint128_t`                            |
| `Cpp.signed_char`        | `signed char`                          |
| `Cpp.short`              | `short`                                |
| `Cpp.int`                | `int`                                  |
| `Cpp.long`               | `long`                                 |
| `Cpp.long_long`          | `long long`                            |
| `Cpp.unsigned_char`      | `unsigned char`                        |
| `Cpp.unsigned_short`     | `unsigned short`                       |
| `Cpp.unsigned_int`       | `unsigned int`                         |
| `Cpp.unsigned_long`      | `unsigned long`                        |
| `Cpp.unsigned_long_long` | `unsigned long long`                   |
| `Cpp.float`              | `float`                                |
| `Cpp.double`             | `double`                               |
| `Cpp.long_double`        | `long double`                          |
| `f16`                    | `std::float16_t (_Float16)`            |
| `f128`                   | `std::float128_t (_Float128)`          |
| TBD                      | `float32_t`, `float64_t`, `bfloat16_t` |
| TBD                      | `char`, `charN_t`, `wchar_t`           |
| TBD                      | `std::byte`                            |
| TBD                      | `std::nullptr_t`                       |

In addition to the exact mappings above, the following are expected to be the
same type due to the different spellings of the types in C++ being the same:

| Carbon type | C++ type                                |
| ----------- | --------------------------------------- |
| `i8`        | `signed char`                           |
| `u8`        | `unsigned char`                         |
| `i16`       | `short`                                 |
| `u16`       | `unsigned short`                        |
| `i32`       | `int`                                   |
| `u32`       | `unsigned int`                          |
| `i64`       | `long` or `long long`                   |
| `u64`       | `unsigned long` or `unsigned long long` |

### C++ -> Carbon mapping details

-   C++ `intN_t` type will be considered the same type as Carbon's `iN` type.
    Likewise for `uintN_t` <-> `uN`.
-   C++ `builtin type` will be available in Carbon inside the `Cpp` namespace
    under the name `Cpp.builtin_type`, for the standard signed/unsigned integer
    and floating-point types.

    -   The names will follow the pattern:

        -   `Cpp.[unsigned_](long_long|long|int|short|double|float)`

        that is signedness, then size keyword(s), then a type keyword only if
        there are no size keywords. For example `Cpp.unsigned_int` not
        `Cpp.unsigned`, `Cpp.long` not `Cpp.long_int`.

    -   They will be available when an `import Cpp` declaration is present.

    -   Name collision: This naming may cause name collisions if such a name
        already exist in the unnamed C++ namespace. We consider this not to be a
        common case and would not support such cases, for the benefit of having
        the C++-specific stuff in the package `Cpp`.

    -   `Cpp.builtin_type` will be the same type as `iN`/`uN`, if the
        corresponding C++ `builtin type` is the same as `intN_t`/`uintN_t` on
        that platform. Otherwise it will be available in Carbon as a new,
        distinct type that is compatible with some of the `iN`/`uN` types. For
        example:

        -   If `int32_t` is the same type as `int`, then `Cpp.int` will be the
            same type as `i32`.

        -   If `int64_t` is the same type as `long`, then `Cpp.long` will be the
            same type as `i64`. `Cpp.long_long` will be a different type,
            compatible with `i64`.

    -   `Cpp.float` and `Cpp.double` will be the same type as `f32` and `f64`
        correspondingly.

-   The type aliases `[u]int_fastN_t`, `[u]int_leastN_t`, `[u]intmax_t`,
    `[u]intptr_t`, `ptrdiff_t` and `size_t` will be available in Carbon in the
    `Cpp` namespace if the C++ header declaring them is imported (for example
    `<stdint.h>`, `<cstdint>` etc), with names like `Cpp.[u]int_fastN_t`,
    `Cpp.[u]int_leastN_t`, `Cpp.size_t` etc. No special support will be
    provided.

### Carbon -> C++ mapping details

-   Same as above, Carbon `iN`/`uN` types will map to the C++ `intN_t`/`uintN_t`
    types.
-   `f32`/`f64` will map to `float`/`double` correspondingly.
-   `f16`/`f128` will map to
    `std::float16_t (_Float16)`/`std::float128_t (_Float128)` correspondingly.
-   Some Carbon types may not have direct mappings in C++: `i256`, `u256` ,
    `f80`, `f256`.

## Rationale

One of Carbon's goals is seamless interoperability with C++ (see
[Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)),
calling for clarity of the calls and high performance.

The proposal maps the Carbon types to their direct equivalents in C++, with zero
overhead, supporting the request for unsurprising mappings between C++ and
Carbon types with high performance.

## Alternatives considered

Naming of new types:

-   Allow all keyword permutations.
    -   Reason not to do this: unnecessary and complicated.
-   Only include the keywords, and provide some syntax for combining them (eg,
    `Cpp.unsigned` & `Cpp.long` or `Cpp.unsigned(Cpp.long)`).
    -   Reason to do this: avoids taking any identifiers from Cpp that are not
        C++ keywords.
    -   Reason not to do this: overly complicated.
-   Use `Core.Cpp.T` instead of `Cpp.T`.
    -   Reason to do this: avoid name collisions with C++ code.
    -   Reason not to do this: The name collisions should not be a problem in
        practice, and would prefer to keep C++-specific stuff in package Cpp.

`long`

-   `Cpp.long` and `Cpp.long_long` both map to Carbon types that are distinct
    from `iN` for any `N`, but are compatible with either `i32` or `i64` as
    appropriate.
    -   Reason to not do this: unnecessary conversions and handling `long` and
        `long long` differently than the other C++ types.
-   Provide platform-dependent conversion functions for `long`.
    -   Reason to do this: the conversions will be clearly outlined.
    -   Reason not to do this: performance overhead for certain platforms.
-   Map `long` always to a fixed-sized Carbon type depending on the platform
    (for example to either `i32` or `i64`)
    -   Reason to do this: all the code will be using fixed-sized types.
    -   Reason not to do this: the same C++ function may map differently on
        different platforms and the Carbon code should compensate for that to
        make the code compile.

`float32_t`, `float64_t`

-   Map `f32` <-> `float32_t` and `f64` <-> `float64_t`
    -   Reason to do this: follow the same analogy as for the integer types
        (`iN` <-> `intN_t`)
    -   Reason not to do this:
        -   `float32_t`, `float64_t` are new types since C++23, so this won't be
            directly achievable, but the corresponding `_FloatN` types will need
            to be used for the older C++ versions.
        -   they are not aliases for the standard floating-point types (`float`,
            `double`, `long double`), but for extended floating-point types, so
            type conversions will be needed for the standard types.

## Open questions

The mapping of the following types remains open and will be discussed at a later
point:

-   `char`, `char8_t`, `char16_t`, `char32_t`, `wchar_t`
    -   Carbon still doesn't have character types, so the mapping of these types
        will be discussed once they are available.
    -   These are all distinct types in C++, which should be taken into account
        to prevent any issues for overloading.
-   `std::byte`
-   `std::nullptr_t`
-   `void*`
-   `Cpp.long_double` - details of this new type is still to be discussed.
-   `float32_t`, `float64_t`, `bfloat16_t`.
