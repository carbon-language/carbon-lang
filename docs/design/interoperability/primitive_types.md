# Primitive types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [32-bit/64-bit conversion APIs](#32-bit64-bit-conversion-apis)
  - [Platforms-specific sizes](#platforms-specific-sizes)
- [Alternatives](#alternatives)
  - [32-bit vs 64-bit and platform compatibility](#32-bit-vs-64-bit-and-platform-compatibility)
    - [Provide variable size types](#provide-variable-size-types)
    - [Do nothing](#do-nothing)
  - [size_t and signed vs unsigned](#size_t-and-signed-vs-unsigned)
    - [Map size_t to UInt64](#map-size_t-to-uint64)
  - [char/unsigned char and byte vs character](#charunsigned-char-and-byte-vs-character)
    - [Support + and - on Byte](#support--and---on-byte)
    - [Create a Char8 type](#create-a-char8-type)
    - [Use Int8](#use-int8)

<!-- tocstop -->

| Carbon    | C/C++                                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------------- |
| `Void`    | `void`                                                                                                |
| `Byte`    | `unsigned char`, `char`, `std::byte`                                                                  |
| `Bool`    | `_Bool`, `bool`                                                                                       |
| `Int8`    | `int8_t`, `signed char`                                                                               |
| `Int16`   | `int16_t`, `short`                                                                                    |
| `Int32`   | `int32_t, int` <br>If 32-bit: `long`, `ptrdiff_t`, `size_t`, `rsize_t`, `ssize_t`                     |
| `Int64`   | `int64_t`, `long long`, `intptr_t` <br>If 64-bit: `long`, `ptrdiff_t`, `size_t`, `rsize_t`, `ssize_t` |
| `Int128`  | `int128_t`                                                                                            |
| `UInt8`   | `uint8_t`                                                                                             |
| `UInt16`  | `uint16_t`, `unsigned short`                                                                          |
| `UInt32`  | `uint32_t`, `unsigned int` <br>If 32-bit: `unsigned long`                                             |
| `UInt64`  | `uint64_t`, `unsigned long long`, `uintptr_t` <br>If 64-bit: `unsigned long, unsigned long`           |
| `Float32` | `float`                                                                                               |
| `Float64` | `double`                                                                                              |

When reading this table, note:

- This list is not intended to be exhaustive, but should be indicative of
  mappings.
- In all cases where the C/C++ type comes in both `::T` and `std::T` forms in
  C++, both are implied by `T` in the table.
- When mapping from Carbon to C/C++ and there are multiple C/C++ type
  candidates, the first one in the list is intended to be the canonical answer.

## 32-bit/64-bit conversion APIs

When writing cross-platform, cross-language code, engineers will need to be
sensitive to what size they use in Carbon versus what size C++ would use.

We will provide conversion APIs, such as `ToCLong`, to improve portability.
Applications interested in maintaining cross-platform interoperability will need
to use these instead of relying on built-in mappings.

For example:

```carbon
package CppCompat;

$if platform == LP64
fn ToCLong(var Int64: val) -> Int64 { return val; }
$else
fn ToCLong(var Int64: val) -> Int32 { return (Int32)val; }
$endif

var Int64: myVal = Foo();
// retVal is always safe because an Int32 can always become an Int64.
var Int64: retVal = Cpp.ApiUsingLong(CppCompat.ToCLong(val));
```

### Platforms-specific sizes

The primary platform-specific sizes for C++ are:

|           | LP32        | ILP32   | LLP64       | LP64        |
| --------- | ----------- | ------- | ----------- | ----------- |
| `int`     | **`Int16`** | `Int32` | `Int32`     | `Int32`     |
| `long`    | `Int32`     | `Int32` | `Int32`     | **`Int64`** |
| `pointer` | `Int32`     | `Int32` | **`Int64`** | **`Int64`** |
| `size_t`  | `Int32`     | `Int32` | **`Int64`** | **`Int64`** |

In practice, we are most worried about differences between LP64 platforms,
including 64-bit Linux, and LLP64, particularly Windows x86_64. If Carbon will
support 32-bit CPUs, we are interested in the ILP32 model, which is adopted by
the vast majority of 32-bit platforms.

Similarly, `float` and `double` may end up being different sizes on particular
platforms.

## Alternatives

### 32-bit vs 64-bit and platform compatibility

At present, we plan to provide conversion functions that provide an
appropriately-sized Carbon type for a given C++ API.

We may want to consider alternatives because the layer of indirection this
creates could be forgotten, resulting in accidentally platform-specific code.

#### Provide variable size types

Carbon could provide compatibility types with matching sizes to the C++
implementation.

For example:

```carbon
package Cpp;

$if platform == LP64
struct long { private var Bytes[8]: data; }
$else
struct long { private var Bytes[4]: data; }
$endif

// This line will fail to compile if, for example, Foo returns an Int64 while
// long is 32-bit.
var Cpp.long: myVal = (Cpp.long)Foo();
var Cpp.long: retVal = Cpp.ApiUsingLong(val);
```

Pros:

- Reduces the amount of platform-specific code that authors need to provide.
- Variable-size types avoid conversion overhead.
- Unsafety around size conversions can be avoided by just staying in
  variable-size types.

Cons:

- Introduces new interoperability-specific types into Carbon.
  - These types are likely to leak beyond C++ interoperability code, through
    support in APIs.
  - Could cause issues if most types in Carbon are assumed to be constant size
    cross-platform, and API authors don't consider variable sizes correctly.
- Conversion functions are likely still necessary for code to switch between the
  variable-size types and Carbon-recommended types.

#### Do nothing

Instead of providing conversion APIs, Carbon could provide nothing.

Pros:

- Simplifies the language.
- Makes unsafety around size conversions more explicit because developers would
  need to handle platform compatibility everywhere it comes up.

Cons:

- Portability issues due to differences between C types across platforms. As a
  result, a given C or C++ API will be imported into Carbon differently
  depending on the selected target. Carbon code would have to be written in such
  a way as to compile in all modes.

The portability issues would likely lead to this only being feasible if Carbon
is rarely, if ever, used on platforms with varying sizes. Otherwise, the demand
for portable code is likely to be high.

### size_t and signed vs unsigned

At present, the plan is that `size_t` will map to the signed `Int64` type.

We may want to consider alternatives because this loses the `size_t` unsigned
semantics, which could lead to some compatibility issues.

#### Map size_t to UInt64

We could alternatively use `UInt64` for the mapping.

Pros:

- Matches C++ semantics.

Cons:

- Pushes engineers to use unsigned types when talking about lengths, risking
  errors with negative values and/or comparisons.
  - We consider signed integers more idiomatic than unsigned.

### char/unsigned char and byte vs character

At present, the plan is that `char`/`unsigned char` should map to `Byte`. `Byte`
is distinct because, while `Int8`/`UInt8` has arithmetic, `Byte` is intended to
not have arithmetic.

We may want to consider alternatives because of a couple differences in
semantics:

- Printability of a `Byte` vs `Char8` or similar.
- Developers may want character arithmetic. For example, `'A' + 15` as a way to
  get the value `'P'`, or adding `32` to capitalize.

#### Support + and - on Byte

We could plan on supporting basic `+` and `-` on `Byte`.

Pros:

- Keeps the `Byte` translation, which may be more appropriate for some APIs.

Cons:

- Adds arithmetic operations to the `Byte` type, which may be inappropriate for
  actual memory representation.

#### Create a Char8 type

We could add a `Char8` type, specifically limiting it to a single byte,
mirroring C++. Note this is `Char8` because we'll presumably have `Char32` for
UTF-32, and possibly `Char` to indicate a multi-byte Unicode character.

Pros:

- Mirrors the C++ semantic.
- Allows for character-specific behaviors, for example when printing values to
  stdout.
- Could support character arithmetic.

Cons:

- Prevents us from representing C++ memory operations as `Byte` without a
  specific type mapping.
- Encourages the concept of a single byte as a character of text, which is
  inconsistent with Unicode.

#### Use Int8

We could convert `char` to `Int8`.

Pros:

- Makes all `char` types an `Int8`.

Cons:

- It's likely that we want to provide more character-like behaviors than `Int8`
  could offer.
- We probably don't want operators like `*` or `/` to provide integer arithmetic
  for a character.
