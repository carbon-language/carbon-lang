# Type mapping

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C/C++ enums in Carbon](#cc-enums-in-carbon)
- [Carbon enums in C/C++](#carbon-enums-in-cc)

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

Sometimes enum names may repeat the enum identifier; for example,
`DIRECTION_EAST` instead of `East`. To help with this case, we may want to
support renaming of enum entries. For example, to rename in a way that results
in a match to the above Carbon calling convention, we add `carbon_enum`:

```cc
enum Direction {
  DIRECTION_EAST,
  DIRECTION_WEST,
  DIRECTION_NORTH,
  DIRECTION_SOUTH,
} __attribute__((carbon_enum("East:West:North:South"));
```

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

For example, given a C++ enum:

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
