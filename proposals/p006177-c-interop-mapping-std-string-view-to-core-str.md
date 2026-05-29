# C++ Interop: Mapping `std::string_view` to `Core.Str`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6177)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [1. Provide a Wrapper Type](#1-provide-a-wrapper-type)
    -   [2. Do Nothing](#2-do-nothing)

<!-- tocstop -->

## Abstract

This proposal defines a direct, zero-cost mapping between C++'s
`std::string_view` and Carbon's `Core.Str` for C++ interoperability. The goal is
to make C++ APIs that use `std::string_view` feel native and seamless when used
from Carbon. This mapping relies on the two types having an identical memory
representation, a condition that we will work to ensure across all supported
platforms.

## Problem

Seamless interoperability with C++ is a core goal for Carbon. `std::string_view`
has become a ubiquitous, fundamental type in modern C++ for passing non-owning
string data. Without a direct mapping, Carbon developers would be forced to work
with ABI-incompatible wrapper types or manually unpack `std::string_view`
instances into pointers and sizes.

This creates significant friction:

1.  **Ergonomics:** C++ APIs would not feel idiomatic. Developers would need to
    perform constant, boilerplate conversions.
2.  **Performance:** Any wrapper-based solution would break the zero-cost
    abstraction principle, potentially introducing overhead at the boundary.
3.  **Adoption:** The lack of seamless integration for such a basic type would
    be a significant barrier to migrating or integrating C++ codebases.

To provide a truly smooth migration path and interoperability story, Carbon must
treat `std::string_view` as a first-class citizen, ideally as the same type as
its native string view.

## Background

Carbon's `Core.Str` is, per
[#5969](https://github.com/carbon-language/carbon-lang/issues/5969), the
language's fundamental non-owning view of a sequence of bytes. It is
[currently implemented as a pair of a pointer to the data and a 64-bit integer representing the size in bytes](https://github.com/carbon-language/carbon-lang/blob/7c13bddc92be8ceac758189df76ebbb048e1a9d5/core/prelude/types/string.carbon#L19-L22).
The assumed memory layout is the pointer followed by the size.

C++'s `std::string_view` serves the same purpose. However, its memory layout is
not standardized and varies between standard library implementations. This is
critical for ABI compatibility.

The layouts for major C++ standard library implementations are as follows:

| Standard Library | Platform/Compiler    | Member Order         | Size Type (`size_t`) | Notes                         | Source                                                                                                                                                |
| ---------------- | -------------------- | -------------------- | -------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **libc++**       | Clang (macOS, etc.)  | `__data_`, `__size_` | 64-bit (on 64-bit)   | Pointer first, then size.     | [`string_view`](https://github.com/llvm/llvm-project/blob/fd5bc6033e521b946f04cb9c473d9cca3da2da9b/libcxx/include/string_view#L711-L712)              |
| **MSVC STL**     | Microsoft Visual C++ | `_Mydata`, `_Mysize` | 64-bit (on 64-bit)   | Pointer first, then size.     | [`__msvc_string_view.hpp`](https://github.com/microsoft/STL/blob/ba64eaaa8592c700949f3c09a0d8570b932828f5/stl/inc/__msvc_string_view.hpp#L1924-L1925) |
| **libstdc++**    | GCC (Linux, etc.)    | `_M_len`, `_M_str`   | 64-bit (on 64-bit)   | **Size first, then pointer.** | [`string_view`](https://github.com/gcc-mirror/gcc/blob/6b999bf40090f356c5bb5ff8a82e7e0dc4c4ae05/libstdc%2B%2B-v3/include/std/string_view#L590-L591)   |

As the table shows, there is a key difference in member ordering, with
`libstdc++` being the outlier compared to the assumed layout of `Core.Str`.

## Proposal

We propose to map C++ `std::string_view` directly to Carbon's `Core.Str` when
importing C++ headers. This means the Carbon compiler will treat
`std::string_view` as a type alias for `Core.Str` at the ABI level.

To achieve this, the following conditions must be met:

1.  **Identical Representation:** The memory layout (sequence of fields, size,
    and alignment) of `std::string_view` and `Core.Str` must be identical for
    the target platform and C++ standard library.
2.  **Platform-Wide Compatibility:** The ultimate goal is for this mapping to
    work seamlessly across all Carbon-supported architectures.

Initially, this direct mapping will be enabled only for targets where the
representation is known to match. For other platforms, we will pursue one of two
strategies:

-   Work with standard library vendors (for example, `libstdc++`) to align on a
    consistent representation for `std::string_view` across architectures,
    falling back to providing a patched version of these libraries if necessary.
-   Adapt the memory layout of Carbon's `Core.Str` on a per-platform basis to
    match the target's native `std::string_view` ABI.

This ensures that while the initial implementation may be constrained, the
long-term design is for universal, zero-cost compatibility.

## Details

The initial implementation will assume `Core.Str` has a `(pointer, size)`
layout. This means the direct mapping will work out-of-the-box on platforms
using `libc++` (Clang) and MSVC STL.

For platforms using `libstdc++`, the current `(size, pointer)` layout is
incompatible. The direct mapping will be disabled on these platforms by default
until compatibility is achieved. Our strategy is to first align `Core.Str`'s
layout with the dominant `(pointer, size)` convention used by Clang and MSVC. We
will then engage with the `libstdc++` community to explore standardizing this
layout for better cross-compiler compatibility.

Furthermore, `Core.Str` is defined with a 64-bit size field. C++
`std::string_view` uses `size_t`, which is 32-bit on 32-bit targets. Therefore,
this direct mapping will initially be restricted to 64-bit targets, which are
Carbon's primary focus.

It is essential to recognize that both `Core.Str` and `std::string_view` are
fundamentally views over bytes, not Unicode characters. This proposal maintains
that semantic alignment. If Carbon requires a string type that understands
character boundaries, it should be a separate, distinct type in the standard
library and not interfere with this fundamental C++ interoperability mechanism.

## Rationale

This proposal directly supports the following Carbon goals:

-   **Interoperability with and migration from existing C++ code:**
    `std::string_view` is one of the most common types in modern C++ interface
    design. A seamless mapping is not a luxury but a requirement for effective
    interoperability.
-   **Performance-critical software:** By ensuring a direct, zero-cost mapping,
    we avoid any performance penalties at the C++/Carbon boundary for string
    data. This is critical for systems programming where such overhead is
    unacceptable.
-   **Code that is easy to read, understand, and write:** A direct mapping
    allows developers to think of `std::string_view` and `Core.Str` as the same
    concept, reducing cognitive load and eliminating the need for manual
    conversions.
-   **Naming of `Core.Str`:** The choice of `Str` over `String` for the
    non-owning view type is intentional. It avoids confusion with owning types
    like C++'s `std::string`. `Str` is introduced as a new term of art for
    Carbon, providing a concise and readable name for this fundamental type. The
    shorter name is preferred for its clarity and reduced verbosity, especially
    for a type that will be used frequently. This is based on the decision in
    leads question
    [#5969](https://github.com/carbon-language/carbon-lang/issues/5969).

By defining a clear path toward universal ABI compatibility for this type, we
are building a solid foundation for deep and performant integration with the
existing C++ ecosystem.

## Alternatives considered

### 1. Provide a Wrapper Type

We could choose to always import `std::string_view` as an opaque Carbon struct,
for example, `Core.Cpp.string_view`.

-   **Advantages:** This would be ABI-safe on all platforms immediately.
-   **Disadvantages:** This approach is not seamless. It would require explicit
    conversions between `Core.Str` and `Core.Cpp.string_view`, adding
    boilerplate and potential performance overhead. It violates the goal of
    making C++ APIs feel native to Carbon.

### 2. Do Nothing

We could leave it to the developer to manually handle `std::string_view` by
accepting it as an opaque type and using C++ helper functions to extract the
pointer and size.

-   **Advantages:** Simplest to implement in the compiler.
-   **Disadvantages:** This provides a terrible developer experience and runs
    directly counter to Carbon's core goal of excellent C++ interoperability. It
    would make using a vast number of modern C++ libraries prohibitively
    difficult.

The proposed approach of a direct mapping is superior as it prioritizes the
long-term goals of performance and ergonomics, even if it requires a phased
implementation to achieve full platform support.
