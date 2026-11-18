# Allow unqualified name lookup

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Classes](#classes)
    -   [Interfaces](#interfaces)
    -   [Namespaces](#namespaces)
-   [Open question](#open-question)
    -   [Implicit instance binding to `me`](#implicit-instance-binding-to-me)
    -   [Out-of-line definitions for impls](#out-of-line-definitions-for-impls)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [No unqualified lookup when defining outside a scope](#no-unqualified-lookup-when-defining-outside-a-scope)

<!-- tocstop -->

## Abstract

Allow unqualified name lookup in multiple situations:

-   For classes and interfaces, whether inside the class scope or within an
    out-of-line function definition.
-   For namespaces, when the namespace is used in a declaration.

## Problem

[Member access](/docs/design/expressions/member_access.md) defines certain
member access behaviors. However, it doesn't cover what happens if an
unqualified name lookup occurs within a class, particularly for an out-of-line
member function definition, or other situations.

## Background

The [member access design](/docs/design/expressions/member_access.md) and
[information accumulation principle](/docs/project/principles/information_accumulation.md)
affect this.

This will also work similarly to
[unqualified lookup within C++](https://en.cppreference.com/w/cpp/language/unqualified_lookup).

## Proposal

Allow unqualified name lookup which will use the appropriate scope.

Implicit instance binding to `me` is not proposed; it is left as an
[open question](#implicit-instance-binding-to-me).

### Classes

This proposal updates [the class design](/docs/design/classes.md) to address
classes.

### Interfaces

```carbon
interface Vector {
  fn Scale[me: Self](v: f64) -> Self;
  // Default definition of `Invert` calls `Scale`.
  default fn Invert[me: Self]() -> Self;
}

// `Self` is valid here because it's doing unqualified name lookup into
// `Vector`.
default fn Vector.Invert[me: Self]() -> Self {
  // `Scale` is valid here because it does unqualified name lookup into
  // `Vector`, then an instance binding with `me`.
  return me.(Scale)(-1.0);
}
```

### Namespaces

More generally, this should also be true of other scopes used in declarations.
In particular, namespaces should also follow the same rule. However, since
[name lookup](/docs/design/name_lookup.md) has not been fleshed out, this won't
make much of an update to it.

An example for namespaces would be:

```carbon
namespace Foo;
var Foo.a: i32 = 0;

class Foo.B {}

// `B` and `a` are valid here because unqualified name lookup occurs within
// `Foo`.
fn Foo.C(B b) -> i32 {
  return a;
}
```

## Open question

### Implicit instance binding to `me`

In C++, unqualified name lookup can implicitly do instance binding to `this`. In
other words, `this->Member()` and `Member()` behave similarly inside a method
definition.

In Carbon, the current design hasn't fleshed out whether `me` would behave
similarly. Most design documentation assumes it will not, but it hasn't been
directly considered in a proposal, and
[implicit scoped function parameters](https://github.com/carbon-language/carbon-lang/issues/1974)
might offer a way to make it work in a language-consistent manner.

This proposal takes no stance on unqualified name lookup resolving `me`: it is
not intended to change behavior from previous proposals.

### Out-of-line definitions for impls

Issue [#2377](https://github.com/carbon-language/carbon-lang/issues/2377) asks
how unqualified lookup should work for `impl`. The
[generics design](/docs/design/generics/details.md) also doesn't appear to give
syntax for out-of-line definitions of other impls.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Performing unqualified name lookup for class members should be fairly
        unsurprising to readers, and should allow for more concise code when
        working within a namespace.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This behavior will be similar to how C++ works.

## Alternatives considered

### No unqualified lookup when defining outside a scope

We could decide not to support unqualified lookup when defining something that
is presented within the top-level scope of the file.

Note this has subtle implications. If `Foo.C` in the namespace example is
considered to be outside the `Foo` scope for this purpose, it means the function
would need to look like:

```
fn Foo.C(Foo.B b) -> i32 {
   return Foo.a;
}
```

It could also mean that, on a class, an inline declaration
`fn Foo() -> ClassMember` is valid, while an out-of-line definition
`fn Class.Foo() -> ClassMember` is not, requiring `Class.ClassMember`.

Advantages:

-   Explicit in access.
    -   For example, name lookup results could be mildly confusing if both
        `package.a` and `package.Foo.a` are defined but `package.Foo.a` is
        hidden in code while `package.a` is easy to find. It's likely that
        `package.Foo.a` would be considered unambiguous for unqualified name
        lookup.

Disadvantages:

-   Very verbose, and could prove un-ergonomic for developers.
