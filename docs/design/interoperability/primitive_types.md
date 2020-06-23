# Primitive types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [32-bit vs 64-bit and platform compatibility](#32-bit-vs-64-bit-and-platform-compatibility)
  - [Alternative: Supplement mappings with platform-compatible conversion APIs](#alternative-supplement-mappings-with-platform-compatible-conversion-apis)
  - [Alternative: Provide variable size types](#alternative-provide-variable-size-types)
- [size_t and signed vs unsigned](#size_t-and-signed-vs-unsigned)
  - [Alternative: Map size_t to UInt64](#alternative-map-size_t-to-uint64)
- [char/unsigned char and byte vs character](#charunsigned-char-and-byte-vs-character)
  - [Alternative: Support + and - on Byte](#alternative-support--and---on-byte)
  - [Alternative: Create a Char8 type](#alternative-create-a-char8-type)
  - [Alternative: Use Int8](#alternative-use-int8)

<!-- tocstop -->

Note: In all cases where the C / C++ type comes in both `::T` and `std::T` forms
in C++, both are implied by writing `T` below. Also, when mapping from Carbon to
C/C++ and there are multiple C/C++ type candidates, the first one in the list is
intended to be the canonical answer.

Note: this is not intended to be exhaustive, but indicative.

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
| `UInt64`  | `uint64_t`, `unsigned long long`, `uintptr_t` <br>If 62-bit: `unsigned long, unsigned long`           |
| `Float16` | `short float` (hopefully)                                                                             |
| `Float32` | `float`                                                                                               |
| `Float64` | `double`                                                                                              |

### 32-bit vs 64-bit and platform compatibility

At present, the proposed translation for these types to Carbon is based on the
corresponding platform-specific size that C++ uses. For example:

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

When writing cross-platform, cross-language code, engineers will need to be
sensitive to what size they use in Carbon vs what size C++ would use.

Pros:

- Types always map to a Carbon type of an explicit, equal size.

Cons:

- Portability issues due to differences between C types across platforms. As a
  result, a given C or C++ API will be imported into Carbon differently
  depending on the selected target. Carbon code would have to be written in such
  a way as to compile in all modes.

#### Alternative: Supplement mappings with platform-compatible conversion APIs

Carbon could provide conversion APIs, such as `ToCLong`, to improve portability.

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

Pros:

- Reduces the amount of platform-specific code that authors need to provide.
- Platform-specific conversions are clearly annotated as being related to
  compatibility with specific C types.

Cons:

- Implies type conversions on certain platforms, with performance overhead.
  - Users may still need to write platform-specific code.
- The code uses explicitly-sized types, so users have to compile code for all
  target platforms to see all possible errors.

#### Alternative: Provide variable size types

Carbon could provide compatibility types with matching sizes to the C++
implementation.

i.e.:

```carbon
package CppCompat;

$if platform == LP64
struct CLong { private var Bytes[8]: data; }
$else
struct CLong { private var Bytes[4]: data; }
$endif

// This line will fail to compile if, for example, Foo returns an Int64 while
// CLong is 32-bit.
var CLong: myVal = (CLong)Foo();
var CLong: retVal = Cpp.ApiUsingLong(val);
```

Pros:

- Reduces the amount of platform-specific code that authors need to provide.

Cons:

- Most Carbon code is still expected to use explicit sizes. Platform-specific
  code should still be expected to crop up around APIs that expect a particular
  size.

### size_t and signed vs unsigned

At present, the proposal is that `size_t` will map to the signed `Int64` type.

Pros:

- Idiomatically represent memory sizes, container lengths, etc as `Int64`.

Cons:

- Does not match `size_t` unsigned semantics.

#### Alternative: Map size_t to UInt64

We could alternatively use `UInt64` for the mapping.

Pros:

- Matches C++ semantics.

Cons:

- Pushes engineers to use unsigned types when talking about lengths, risking
  errors with negative values and/or comparisons.

### char/unsigned char and byte vs character

At present, the proposal is that `char`/`unsigned char` should map to `Byte`.
`Byte` is distinct because, while `Int8`/`UInt8` has arithmetic, `Byte` is
intended to not have arithmetic.

Pros:

- C/C++ use char types in some cases when dealing with memory, because there
  hasn't historically been a dedicated byte type.

Cons:

- Users may do character arithmetic on `char`.
  - Using an integer offset to get a letter. For example, `'A' + 15` as a way to
    get the value `'P'`.
  - Using `32` to capitalize. For example, `'a' + 32` as a way to get the value
    `'A'`.

#### Alternative: Support + and - on Byte

We could plan on supporting basic `+` and `-` on `Byte`.

Pros:

- Keeps the `Byte` translation, which may be more appropriate for some APIs.

Cons:

- Adds arithmetic operations to the `Byte` type, which may be inappropriate for
  actual memory representation.

#### Alternative: Create a Char8 type

We could add a `Char8` type, specifically limiting it to a single byte,
mirroring C++. Note this is `Char8` because we'll presumably have `Char32` for
UTF-32, and possibly `Char` to indicate a multi-byte Unicode character.

Pros:

- Mirrors the C++ semantic.
- Allows for character-specific behaviors, for example when printing values to
  stdout.

Cons:

- Prevents us from representing C++ memory operations as `Byte` without a
  specific type mapping.

#### Alternative: Use Int8

We could convert `char` to `Int8`.

Pros:

- Makes all `char` types an `Int8`.

Cons:

- It's likely that we want to provide more character-like behaviors than `Int8`
  could offer.
- We probably don't want operators like `*` or `/` to provide integer arithmetic
  for a character.
