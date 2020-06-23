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

Given an enum:

```
enum Direction {
  East,
  West = 20,
  North,
  South,
};
```

We would expect to generate equivalent Carbon code:

```
enum Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
}

// Calling semantic:
var Direction: x = Direction.East;
```

Sometimes enum names may repeat the enum identifier, e.g., `DIRECTION_EAST`
instead of `East`. To help with this case, we may want to support renaming of
enum entries. e.g., to rename in a way that results in a match to the above
Carbon calling convention:

```
enum Direction {
  DIRECTION_EAST,
  DIRECTION_WEST,
  DIRECTION_NORTH,
  DIRECTION_SOUTH,
} __attribute__((carbon_enum("East:West:North:South"));
```

If using enum class, we'd expect similar behavior:

```
enum class Direction : char {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

With Carbon code:

```
enum(Byte) Direction {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

## Carbon enums in C/C++

Given an enum:

```
$extern("Cpp") enum Direction {
  East,
  West = 20,
  North,
  South,
}
```

Because Carbon automatically groups enums, we would expect to generate
equivalent C++ code:

```
enum class Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
};
```
