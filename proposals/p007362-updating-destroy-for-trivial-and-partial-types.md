# Define `Destroy` for `class` types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7362)

<!-- toc -->

## Table of contents

- [Define `Destroy` for `class` types](#define-destroy-for-class-types)
  - [Table of contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Problem](#problem)
  - [Background](#background)
  - [Proposal](#proposal)
  - [Details](#details)
    - [`Destructor` interface](#destructor-interface)
    - [`Destroy` interface](#destroy-interface)
    - [`TrivialDestroy` interface](#trivialdestroy-interface)
    - [`DynamicDestroy` interface](#dynamicdestroy-interface)
  - [Rationale](#rationale)
  - [Alternatives considered](#alternatives-considered)
  - [Future work](#future-work)

<!-- tocstop -->

## Abstract

Describe the relationship between `Destroy` and `class` types.

## Problem

Carbon has not specified how `Destroy` works for `class` types. This is a broad
problem because classes may customise how they are destroyed by implementing
`Destroy`.

## Background

* **[Issue #6124]** discusses how `Destroy` should interact with `class`
  types. The Leads decided that destructors are separate from how objects are
  destroyed.
* **[Issue #6161]** discusses whether `partial` types are destroyable. The
  Leads decided that `Destructor.Op` is valid for both `partial` and final
  types. Dynamic types are to be destroyed by the interface `DynamicDestroy`.
* **[Issue #6464]** clarifies that trivial destruction is a property that code
  can query. The Leads confirmed this is the case.

## Proposal

Add to the `Core` library:

```carbon
interface Destructor {
  private fn Op(ref self: partial Self);
}
```
```carbon
interface Destroy {
  private fn Op(ref self);
}
```
```carbon
interface TrivialDestroy {
  private fn Op(ref self);
}
```
```carbon
interface DynamicDestroy {
  private fn Op(ref self);
}
```

## Details

### `Destructor` interface

`Destructor` describes how an object performs cleanup and resource deallocation.
`Destructor.Op()` tears down a single object of the calss that implements the
interface. Subobjects and lifetime management are unaffected. Types only need
to implement `Destructor` when custom destruction logic is required.

All types are provided with an implicit, toolchain-synthesised implementation of
`Destructor` by default. Types that don't provide a user-defined implementation
have trivial destruction. A type that has trivial destruction may be comprised
of subobjects that do not have trivial destruction.

### `Destroy` interface

`Destroy` destroys complete objects---including subobjects---and ends their
lifetimes. Classes cannot customise `Destroy`: it is exclusively implemented by
the compiler. Types requiring control over how subobjects are destroyed must use
a raw storage type for subobjects.

Classes are allowed to opt out from being destroyable. The mechanism for
describing non-destroyable types is not yet determined.

`Destroy` behaves uniformly for all destroyable types.

### `TrivialDestroy` interface

A type automatically impls `TrivialDestroy` if none of its subobjects' impl
`Destructor`. `TrivialDestroy` can only be implemented by the compiler.

### `DynamicDestroy` interface

## Rationale

TODO: How does this proposal effectively advance Carbon's goals? Rather than
re-stating the full motivation, this should connect that motivation back to
Carbon's stated goals and principles. This may evolve during review. Use links
to appropriate sections of [`/docs/project/goals.md`](/docs/project/goals.md),
and/or to documents in [`/docs/project/principles`](/docs/project/principles).
For example:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

## Alternatives considered

TODO: What alternative solutions have you considered?

## Future work

* Add an opt-out from implementing `Destroy`.
* Unify with copy and move semantics.

<!-- Links -->

[Issue #6124]: https://github.com/carbon-language/carbon-lang/issues/6124
[Issue #6161]: https://github.com/carbon-language/carbon-lang/issues/6161
[Issue #6464]: https://github.com/carbon-language/carbon-lang/issues/6464
