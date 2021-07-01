# Carbon appendix: interface implementation syntax alternatives

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The interface implementation syntax was decided in
[question-for-leads issue #575](https://github.com/carbon-language/carbon-lang/issues/575).
This appendix has additional discussion about rejected alternatives.

## Rejected: out-of-line impl

We considered an out-of-line syntax for declaring and defining interface `impl`
blocks, to replace both the inline syntax and the `external impl` statement.
For, example:

```
struct Point { ... }
impl Vector for Point { ... }
```

The main advantage of this syntax was that it was uniform across many cases,
including [conditional conformance](details.md#conditional-conformance). It
wasn't ideal across a number of dimensions though.

-   It repeated the type name which was redundant and verbose
-   It could affect the API of the type outside of the type definition.
-   We prefer the type name before the interface name (using `as), instead of
    this ordering used by Rust.

## Rejected: extend blocks

Instead of the `external impl` statement, we considered putting all external
implementations in an `extend` block.

```
struct Point { ... }
extend Point {
  impl Vector { ... }
}
```

The `extend` approach had some disadvantages:

-   Implementations were indented more than the `external impl` approach.
-   Extra ceremony in the case of only implementing one type for an interface.
    This case is expected to be common since external implementations will most
    often be defined with the interface.
-   When implementing multiple interfaces in a single `extend` block, the name
    of the type being extended could be far from the `impl` declaration and hard
    to find.
