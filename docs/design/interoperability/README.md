# Carbon &lt;-> C/C++ interoperability

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Overview](#overview)
- [Goals](#goals)
- [Philosophy of interoperability layer](#philosophy-of-interoperability-layer)
- [Interoperability syntax elements](#interoperability-syntax-elements)
- [Details](#details)
  - [Bridge code in Carbon files](#bridge-code-in-carbon-files)
  - [Name mapping](#name-mapping)
  - [Type mapping](#type-mapping)
    - [Primitive types](#primitive-types)
    - [User-defined types](#user-defined-types)
    - [Vocabulary types](#vocabulary-types)
  - [Enums](#enums)
  - [Templates and generics](#templates-and-generics)
  - [Functions and overload sets](#functions-and-overload-sets)
  - [Other syntax](#other-syntax)
  - [Migration examples](#migration-examples)
- [Acknowledgements](#acknowledgements)

<!-- tocstop -->

## Overview

It's critical that Carbon can both access C++ interfaces and export C++
interfaces. Supporting C++ interoperability is a
[key requirement for Carbon's goals](/docs/project/goals.md) and is expected to
influence the design of Carbon itself.

## Goals

The goals of Carbon's interoperability layer are heavily influenced by the
[language-level goals](/docs/project/goals.md). Notably, we prioritize
performance, and making any performance overhead visible and opt-in.

Goals:

- The majority of simple C/C++ interfaces should be usable from Carbon without
  any custom bridge code and without any runtime overhead.
- There should be support for most idiomatic usage of advanced C++ features:
  templates, overload sets, ADL.
- It should be possible to map types across languages.
  - Primitive types should have unsurprising mappings.
  - There should be transparent, automatic translation between C++ and Carbon
    non-owning vocabulary types, such as pointers and references, without
    runtime overhead.
  - It should be possible to expose other C++ vocabulary types with reasonable,
    but but potentially non-zero-cost, conversions available to map into Carbon
    vocabulary types.
- Mappings should be easy to maintain.
  - We should provide a syntax for transparently, automatically exposing a
    subset of Carbon types and interfaces to C++ code without custom bridge
    code, including instantiating templates.

Non-goals:

- We will not make Carbon -> C++ migrations as easy as C++ -> Carbon migrations.
  - Automatically exposing all Carbon code would impose language and API
    constraints; we want to limit these constraints to where users need it.
- We do not expect to support all C/C++ corner cases: the complexity of
  supporting any given construct must be balanced by the real world need for
  that support.
  - We may target C++17, and not keep adding interoperability support for later
    C++ features.
    - There may be interest in supporting some C++20 features, particularly
      modules. However, exhaustive support should not be assumed.
  - For example, we might not prioritize support for non-idiomatic code,
    interfaces, or patterns outside of those in widespread open source libraries
    and used by key contributors.
  - For example, we might not support low-level C ABIs outside of modern 64-bit
    ABIs: Linux, POSIX, and a small subset of Windows' calling conventions.
- We may choose not to use the existing ABI for the C++ language or standard
  library.
  - It might be reasonable to eventually support these with added runtime
    overhead.
- We may choose not to provide exact matches between Carbon and C++ vocabulary
  types.
- We may choose not to provide full support for unwinding exceptions across
  Carbon and C/C++ boundaries.

## Philosophy of interoperability layer

The design for interoperation between Carbon and C++ hinges on:

1. A focus on types, and simple overload sets built from those types.
2. A willingness to expose the idioms of C++ into Carbon code, and vice versa,
   when necessary to maximize performance.
3. The use of wrappers, generic programming, and templates to minimize or
   eliminate runtime overhead.

These things come together when looking at how custom data structures in C++ are
exposed into Carbon or vice versa. In both languages, it is reasonable and even
common to have customized low-level data structures such as associative
containers. Even today, there are numerous data structures for mapping from a
key to a value that might be "best": hash table, linked hash table, sorted
vector, and btree to name a few. Experience with C++ tells us that we should
also expect slow but meaningful evolution even in implementation strategies that
cause divergence.

The result is that it is often reasonable to directly expose a data structure
from C++ to Carbon without converting it to a "native" or "idiomatic" Carbon
data structure. For many data structures, code will reasonably support multiple
different implementations, even if there is an extremely good default. We can
expose C++ data structures as another implementation and then focus on wrapping
it to match whatever idioms Carbon expects of that kind of data structure.

The reverse is also true. C++ code will often not care, or can be enhanced to
not care, what specific data structure is used. Carbon data structures can be
exposed as yet another implementation in C++, and wrapped in C++ code to match
C++ idioms and be usable within templated contexts.

Another fundamental philosophy of interoperability between Carbon and C++, and
generally between any two languages, is that it requires expressing one language
as a subset of another. The approach proposed is to do this in two directions:
take specific, restricted C++ APIs and make them available using restricted
Carbon APIs, and also take specific, restricted Carbon APIs and make them
available as restricted C++ APIs. In both languages, the API restrictions on
exported interfaces and imported interfaces can be intersected. These
intersections define the interoperability layer and constrain the expressivity
of Carbon/C++ interoperability. Our goal is that these expressivity constraints
are wide enough to make the amount of bridge code sustainable and the overhead
of wrappers manageable.

## Interoperability syntax elements

> References: [Name mapping](name_mapping.md) and
> [user defined types](user_defined_types.md).

Most C++ code will be automatically exposed to Carbon. However, special
interoperability syntax elements will be required when exposing Carbon code to
C++.

Notable elements are:

- `$extern("Cpp")`: Indicates that Carbon code should be exposed for C++.
  Similarly, `$extern("Swift")` might be used to indicate exposure for Swift at
  some point in the future.
  - `namespace` and `name` parameters are provided to override default choices,
    particularly to assist migration of C++ APIs to Carbon.
  - A `parent` parameter will be provided to set the C++ parent class on an
    externalized Carbon struct. This is expected to be generally useful for
    interoperability.
  - Externs may be #included using `.6c.h` files.
- `import Cpp "path"`: Imports API calls from a C++ header file.

We use the name `Cpp` because `import` needs a valid identifier.

## Details

### Bridge code in Carbon files

> TODO: We should allow writing bridge C++ code in Carbon files to ease
> maintenance of compatibility layers. Syntax needs to be proposed.

### Name mapping

> References: [Name mapping](name_mapping.md).
>
> TODO: Add a reference for incomplete types when one is created.

C/C++ names are mapped into the `Cpp` Carbon package. C++ namespaces work the
same fundamental way as Carbon namespaces within the `Cpp` package name. Dotted
names are used when referencing these names from Carbon code. For example,
`std::exit` becomes `Cpp.std.exit`.

C++ incomplete types will be mirrored into Carbon's incomplete type behavior.
Users wanting to avoid differences in incomplete type behaviors should fully
define the C++ types using repeated imports.

Carbon names which are mapped into C++ will use a top-level namespace of
`Carbon` by default, with the package name and namespaces represented as
namespaces below that. For example, the `Widget` Carbon package with a namespace
`Foo` would become `::Carbon::Widget::Foo` in C++. This may be renamed for
backwards compatibility for C++ callers when migrating code, for example
`$extern("Cpp", namespace="widget")` for `::widget`.

### Type mapping

Carbon and C/C++ will have a number of types with direct mappings between the
languages.

The behavior of mapped types will not always be identical; they need only be
similar. For example, we expect Carbon's `UInt32` to map to C++'s `uint32_t`.
While behavior is mostly equivalent, where C++ would use modulo wrapping, Carbon
will instead have trapping behavior.

In some cases, there are multiple C/C++ types that map to a single Carbon type.
This is expected to generally be transparent to users, but may impose important
constraints around overload resolution and other C++ features that would "just
work" with 1:1 mappings.

#### Primitive types

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

#### User-defined types

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

#### Vocabulary types

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

### Enums

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

### Templates and generics

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

### Functions and overload sets

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

### Other syntax

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

### Migration examples

> TODO: Proposals pending to fill in; see linked PRs on
> [#108](https://github.com/carbon-language/carbon-lang/pull/108).

## Acknowledgements

The thought put into Swift's C/C++ interoperability plan for
[C](https://github.com/apple/swift/blob/master/docs/HowSwiftImportsCAPIs.md) and
[C++](https://github.com/apple/swift/blob/master/docs/CppInteroperabilityManifesto.md)
has helped shape our ideas.
