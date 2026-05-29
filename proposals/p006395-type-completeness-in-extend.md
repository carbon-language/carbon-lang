# Type completeness in extend

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6395)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Require any target scopes in an `extend` declaration to be complete, since
`extend` changes the scopes which are looked in for name lookup to include those
named in the declaration.

## Problem

Proposal
[#5168](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md)
laid out rules for when a facet type needs to be identified or complete. When
`require impls X` is written, then `X` must be identified. However it does not
specify any rule for `extend require impls X`. And we lack completeness rules
for other `extend` declarations, including `extend impl as X`, `extend base: X`,
and `extend adapt X`.

An `extend` declaration always names one or more target entities which will be
included in the search for names when looking in the enclosing scope of the
`extend` declaration. In order to do name lookup into an entity, it must be
complete to avoid poisoning names in the entity.

## Background

-   Proposal
    [#5168](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md):
    Forward `impl` declaration of an incomplete interface
-   Proposal
    [#2760](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p002760-consistent-class-and-interface-syntax.md):
    Consistent `class` and `interface` syntax
-   Proposal
    [#0777](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p000777-inheritance.md):
    Inheritance

## Proposal

An `extend` declaration declares that the enclosing scope extends the scope
associated with any target entity named in the declaration. That is to say name
lookups into the enclosing scope will also look into the scopes which are
nominated by the `extend` declaration. The `extend` declaration requires that
the target scopes named in an `extend` declaration are complete, or if the
target is a generic parameter, requires the type of the parameter to be
complete.

The scope of a facet type formed by a `where` declaration extends the scope of
its first operand. And the scope of a facet type formed by a `&` operator
extends the scopes of both of its operands. In either case, the scope of the
facet type is complete if every scope it extends is complete. The facet type
`type` is associated with an empty scope and is complete.

| Facet type                   | Requirement                           |
| ---------------------------- | ------------------------------------- |
| `I`                          | Requires `I` is complete.             |
| `I & J`                      | Requires `I` and `J` are complete.    |
| `type where U impls J`       | Requires `type` is complete.          |
| `I & (type where U impls J)` | Requires `I` and `type` are complete. |

## Details

To `extend` an entity `Y` with another `Z` means that name lookups into `Y` will
also look into `Z`. Immediately after the `extend` operation, members of `Z`
should also be found when doing name lookup into `Y`, both from outside and from
inside the definition of `Y`. In order to be able to perform lookups into `Z`,
we require that `extend` operations only target scopes that are complete.

This requirement functions recursively. Given an interface `B` that extends
another interface `A`: By naming `A` in an extend declaration, we require `A` is
complete. This provides that its entire definition is known, and thus its
`extend` relationship to `B`. The `extend` relationship there also provides that
`B` is complete.

If the target scope of an `extend` declaration is a generic parameter, its type
must be complete where the `extend` declaration is written, as name lookups into
the extended scope will look into the type of the generic parameter.

```carbon
interface I {
  fn F();
}

class C(T:! I) {
  extend base: T;
  // `F` names `T.F` here, found in `I`.
  fn G() { F(); }
}
```

As any generic parameter in the enclosing scope is replaced by a more specific
value, extended scopes that depend on a generic parameter must remain complete.
This includes forming a specific for the extended scope involving the parameter
in order to surface any monomorphization errors in the resulting specific.

In the next example, the `extend` declaration in interface `A(N)` names a
symbolic facet type which can produce monomorphization errors when a negative
value is provided for `N`. When a more specific value for the target `B(N)` is
provided, we require the specific value to be complete as well by forming the
specific. A diagnostic error would be produced while checking `C(-1)` for
completeness, as it requires `A(-1)` to be complete, which requires
`B(array(i32, -1))` to be complete, and that contains an invalid type.

```carbon
interface B(T:! type) {}

interface A(N:! i32) {
  // Requires `B(N)` to be complete.
  extend require impls B(array(i32, N)) {}
}

class C(N! i32) {
  // Requires `A(N)` to be complete, which requires `B(N)` to be complete.
  extend impl as A(N);
}

fn F() {
  // Requires `C(-1)` to be complete, which requires `A(-1)` to be complete, which requires `B(array(i32, -1))` to be complete.
  var c: C(-1);
}
```

These rules prohibit an `extend` declaration from naming its enclosing scope,
since by being part of the definition of that scope, it is implied that the
enclosing scope is not complete. This seems reasonable as all names available
inside the enclosing interface or named constraint are already available or
would conflict with the ones that are.

## Rationale

This is based on the principle of
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
For code to be easy to write, the rules need to be consistent. Once an `extend`
declaration has been written, the names inside should become available through
the enclosing scope immediately. If we allow an interface named in an `extend`
declaration to be incomplete, then name lookup will fail ambiguously. Those same
names may become valid later once the interface is defined.

## Alternatives considered

We considered not requiring that the scope named in an `extend` declaration be
complete where it is written, but only when the enclosing scope is required to
be complete. This is more flexible, allowing entities to be defined later in the
program. However this does not allow the use of names from the target scope in
the `extend` to be used from within the enclosing definition scope. They would
not become available until the enclosing definition scope was closed and
complete.

In particular, we want this to work:

```carbon
interface Z {
  let X:! type;
  fn F() -> X;
}
class C {
  extend impl as Z where .X = () {
    // Names `X` directly.
    fn F() -> X;
  }

  // Names `X` and `F` directly.
  fn G() -> X { return F(); }
}

// Now `C` is complete, `C.X` and `C.F` are also available.
fn H() -> C.X {
  return C.F();
}
```
