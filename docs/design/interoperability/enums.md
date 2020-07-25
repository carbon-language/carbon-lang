# Type mapping

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C/C++ enums in Carbon](#cc-enums-in-carbon)
  - [Stripping common prefixes](#stripping-common-prefixes)
  - [Renaming enum values](#renaming-enum-values)
  - [C++ enum classes](#c-enum-classes)
- [Carbon enums in C/C++](#carbon-enums-in-cc)
- [Open questions](#open-questions)
  - [Enum-to-integer implicit casts](#enum-to-integer-implicit-casts)
  - [Integer-to-enum casts](#integer-to-enum-casts)
  - [Enum sizes](#enum-sizes)

<!-- tocstop -->

We expect enums can be represented directly in the other language. All values in
the copy should be assumed to be explicit, to prevent any possible issues in
enum semantics.

## C/C++ enums in Carbon

C++ enums will generally translate naturally to Carbon, whether using `enum` or
`enum class`. Attributes may be used if renaming is desired to reach
conventional Carbon naming.

Given a C++ enum:

```cc
enum Direction {
  East,
  West = 20,
  North,
  South,
};
```

We would expect to generate equivalent Carbon code:

```carbon
enum Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
}

// Calling semantic:
var Direction: x = Direction.East;
```

### Stripping common prefixes

Sometimes enum names may repeat the enum identifier; for example,
`DIRECTION_EAST` instead of `East`. To help with this case, we strip common
prefixes by default, with heuristics to handle various naming styles to only
strip on word boundaries.

For example, this kind of enum would automatically strip `DIRECTION_` when
renaming:

```cc
enum Direction {
  DIRECTION_EAST,
  DIRECTION_WEST,
  DIRECTION_NORTH,
  DIRECTION_SOUTH,
};
```

The reason for detecting word boundaries may be seen in this example, where
`SHAPE_C` is the common prefix but `SHAPE_` is what should be stripped:

```cc
enum Shape {
  SHAPE_CIRCLE,
  SHAPE_CYLINDER,
};
```

### Renaming enum values

In order to change names, such as making more idiomatic casing, we can also add
a `carbon_enum` attribute:

```cc
enum Direction {
  DIRECTION_EAST,
  DIRECTION_WEST,
  DIRECTION_NORTH,
  DIRECTION_SOUTH,
} __attribute__((carbon_enum("East:West:North:South"));
```

### C++ enum classes

If using enum class, we'd expect similar behavior:

```cc
enum class Direction : char {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

With Carbon code:

```carbon
enum(Byte) Direction {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

## Carbon enums in C/C++

Carbon enums should be expected to translate to `enum class`.

For example, given a Carbon enum:

```carbon
$extern("Cpp") enum Direction {
  East,
  West = 20,
  North,
  South,
}
```

We would expect to generate equivalent C++ code:

```cc
enum class Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
};
```

## Open questions

### Enum-to-integer implicit casts

C and C++ APIs sometimes rely on enums being implicitly convertible to integers.
It might be the case that, either by default or always, Carbon enums will not be
convertible to integers in order to avoid even a remote possibility of anyone
relying on numeric values.

Ignoring this issue will lead to some APIs being non-ergonomic. That's okay, but
we should ensure these APIs are still usable.

### Integer-to-enum casts

C and C++ APIs also sometimes cast arbitrary integers into enum values, which
Carbon enums may decide to prohibit. Ignoring this issue could lead to
miscompiles.

### Enum sizes

Carbon enum sizes and C++ enum sizes may end up varying. We may need some kind
of attribute support in order to ensure sizes work correctly when bridging.
