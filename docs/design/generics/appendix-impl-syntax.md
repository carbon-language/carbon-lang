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

This syntax also could also be used inline inside a `struct` definition for
conditional conformance use cases:

```
struct FixedArray(T:! Type, N:! Int) {
  // A few different syntax possibilities here:
  extend FixedArray(P:! Printable, N2:! Int) { impl as Printable { ... } }
  extend FixedArray(P:! Printable, N) { impl as Printable { ... } }
  extend[P:! Printable] FixedArray(P, N) { impl as Printable { ... } }
}

struct Pair(T:! Type, U:! Type) {
  extend Pair(T, T) { impl as Foo(T) { ... } }
}
```

## Rejected: other conditional conformance syntax options

Some other ideas we had considered lack the consistency between internal and
external conditional conformance:

-   One approach would be to use deduced arguments in square brackets after the
    `impl` keyword, and an `if` clause to add constraints:

```
struct FixedArray(T:! Type, N:! Int) {
  impl as [U:! Printable] Printable if T == U {
    // Here `T` and `U` have the same value and so you can freely
    // cast between them. The difference is that you can call the
    // `Print` method on values of type `U`.
  }
}

struct Pair(T:! Type, U:! Type) {
  impl as Foo(T) if T == U {
    // Can cast between `Pair(T, U)` and `Pair(T, T)` since `T == U`.
  }
}
```

-   Another approach is to use pattern matching instead of boolean conditions.
    This might look like (though it introduces another level of indentation):

```
struct FixedArray(T:! Type, N:! Int) {
  @if let P:! Printable = T {
    impl as Printable { ... }
  }
}

interface Foo(T:! Type) { ... }
struct Pair(T:! Type, U:! Type) {
  @if let Pair(T, T) = Self {
    impl as Foo(T) { ... }
  }
}
```

We can have this consistency, but lose the property that all unqualified names
for a type come from its `struct` definition:

-   We could keep `extend` statements outside of the struct block to avoid
    sharing names between scopes, but allow them to have an `internal` keyword
    (as long as it is in the same library).

```
struct FixedArray(T:! Type, N:! Int) { ... }
extend FixedArray(P:! Printable, N:! Int) internal {
  impl as Printable { ... }
}

struct Pair(T:! Type, U:! Type) { ... }
extend Pair(T:! Type, T) internal { ... }
```

Lastly, we could adopt a "flow sensitive" approach, where the meaning of names
can change in an inner scope. This would allow the `if` conditions that govern
when an `impl` is used to affect the types in that `impl`'s definition:

```
struct FixedArray(T:! Type, N:! Int) {
  impl as Printable if (T implements Printable) {
    // Inside this scope, `T` has type `Printable` instead of `Type`.
  }
}
```

This would require mechanisms to both describe these conditions and determine
how they affect types.
