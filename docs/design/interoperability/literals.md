# C++ interop type mapping for integer and floating-point literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Carbon to C++ type mapping](#carbon-to-c-type-mapping)
    -   [Integer literals](#integer-literals)
    -   [Floating-point literals](#floating-point-literals)
-   [C++ to Carbon type mapping](#c-to-carbon-type-mapping)
    -   [Integer literals](#integer-literals-1)
    -   [Floating-point literals](#floating-point-literals-1)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

The behavior of literals that participate in interop depends on the direction of
interop, but generally follows C++ rules:

-   **Carbon literal to C++ type**: A Carbon literal will be given a C++ type
    according to the C++ rules when used in C++ context, such as calling a C++
    function.
-   **C++ literal to Carbon type**: A C++ literal will be given a C++ type
    following the C++ rules, which will then be mapped to a Carbon type as
    defined in
    [primitive types mapping](/proposals/p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).
    -   For example, `1` becomes `int` according to C++ rules, and `int` maps to
        `i32` in Carbon.

This is used:

-   During overload resolution of C++ functions, when a C++ function is called
    from Carbon with a literal as an argument.
-   When importing C++ macros to Carbon, such as macros defining constants where
    the constant is a literal.

## Carbon to C++ type mapping

### Integer literals

Because there are no suffixes in Carbon for the integer types, a Carbon decimal
integer literal will follow the C++ rules for decimal integers. Carbon also
doesn't make a distinction between hexadecimal and binary literals for interop,
so these literals will also follow the C++ rules for decimal integers without a
suffix.

The first type from the following list in which the literal's value can fit is
selected:

-   `int`
-   `long`
-   `long long`
-   `__int128`

If the value doesn’t fit in any of these types, the program is ill-formed and
diagnosed with an error. This matches the C++ standard behavior rather than
Clang's `-Wimplicitly-unsigned-literal` behavior.

Because decimal numbers are most commonly used as integer literals, this should
match most existing C++ calls. To match a C++ call with a non-decimal literal
argument expecting an unsigned type, an explicit unsigned type must be provided
in Carbon. For example:

```carbon
Cpp.f(0xDEADBEEF as u32);
```

### Floating-point literals

As there are no suffixes in Carbon for the floating-point literals, a Carbon
floating-point literal will map to C++ `double`. If the value doesn’t fit in
`double`, the program is ill-formed.

## C++ to Carbon type mapping

### Integer literals

A C++ integer literal will be given a C++ integer type following the C++ rules,
which is then mapped to a Carbon type as defined in the
[primitive types mapping](/proposals/p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).

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

### Floating-point literals

A C++ floating literal will be given a C++ type following the C++ rules, which
is then mapped to a Carbon type as defined in
[primitive types mapping](/proposals/p005448-carbon-c-interop-primitive-types.md#carbon-primitive-types).
:

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

## Alternatives considered

Carbon to C++ type mapping

-   [Use the C++ standard rules for hexadecimal and binary literals](/proposals/p006668-c-interop-type-mapping-for-integer-and-floating-point-literals.md#use-the-c-standard-rules-for-hexadecimal-and-binary-literals)
-   [Use the Clang way of determining the type](/proposals/p006668-c-interop-type-mapping-for-integer-and-floating-point-literals.md#use-the-clang-way-of-determining-the-type)

C++ to Carbon type mapping

-   [Use Carbon literal types](/proposals/p006668-c-interop-type-mapping-for-integer-and-floating-point-literals.md#use-carbon-literal-types)

## References

-   [Proposal #144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
-   [Proposal #5448: Carbon / C++ interop primitive types](https://github.com/carbon-language/carbon-lang/pull/5448)
-   [Proposal #6668: C++ interop type mapping for integer and floating-point literals](https://github.com/carbon-language/carbon-lang/pull/6668)
