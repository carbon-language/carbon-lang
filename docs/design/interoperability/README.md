# Carbon ↔ C/C++ interoperability

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
- [Interoperability Syntax Elements](#interoperability-syntax-elements)
- [Details](#details)
  - [Name mapping](#name-mapping)
  - [Type mapping](#type-mapping)
  - [Functions and overload sets](#functions-and-overload-sets)
  - [Other syntax](#other-syntax)
  - [Migration examples](#migration-examples)

<!-- tocstop -->

## Overview

It's critical that Carbon can both access C++ interfaces and export C++
interfaces. Supporting C++ interoperability is a
[key requirement for Carbon's goals](/docs/project/goals.md) and is expected to
influence the design of Carbon itself.

## Goals

The goals of Carbon's interoperability layer are heavily influenced by the
language-level goals for Carbon and C++ at Google. Notably, we prioritize
performance, and making any performance overhead visible and opt-in.

Goals:

- The majority of simple C/C++ interfaces should be usable from Carbon without
  any custom bridge code and without any runtime overhead.
- There should be support for most idiomatic usage of advanced C++ features
  (templates, overload sets, ADL).
- It should be possible to map types across languages.
  - Primitive types should have unsurprising mappings.
  - There should be transparent, automatic translation between C++ and Carbon
    non-owning vocabulary types (e.g., pointers and references) without runtime
    overhead.
  - It should be possible to expose other C++ vocabulary types with reasonable
    (but potentially non-zero-cost) conversions available to map into Carbon
    vocabulary types.
- Mappings should be easy to maintain.
  - We should provide a syntax for transparently, automatically exposing a
    subset of Carbon types and interfaces to C++ code (including instantiating
    templates) without custom bridge code.

Non-goals:

- We will not make Carbon -> C++ migrations as easy as C++ -> Carbon migrations.
  - Doing automatic externs for Carbon code would impose language constraints
    that we want to limit to where users need it.
- We do not expect to support all C/C++ corner cases: the complexity of
  supporting any given construct must be balanced by the real world need for
  that support.
  - We may target C++17, and not keep adding interoperability support for later
    C++ features.
  - e.g., we might not prioritize support for non-idiomatic "modern" code,
    interfaces, or patterns outside of those in widespread open source libraries
    and used by key contributors.
  - e.g., we might not support low-level C ABIs outside of modern 64-bit ABIs
    (Linux, POSIX, and a small subset of Windows' calling conventions).
- We may choose not to use the existing C++ language or standard library ABI.
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
data structure. For many data structures, while there should always be an
extremely good default, code will reasonably support multiple different
implementations. We can expose C++ data structures as another implementation and
then focus on wrapping it to match whatever idioms Carbon expects of that kind
of data structure.

The reverse is also true. C++ code will often not care (or can be enhanced to
not care) what specific data structure is used. Carbon data structures can be
exposed as yet another implementation in C++, and wrapped in C++ code to match
C++ idioms and be usable within templated contexts.

Another fundamental philosophy of interoperability between Carbon and C++ (and
generally between any two languages) is that it requires expressing one language
as a subset of another. The approach proposed is to do this in two directions:
take specific, restricted C++ APIs and make them available using restricted
Carbon APIs, and also take specific, restricted Carbon APIs and make them
available as restricted C++ APIs. In both languages, the API restrictions on
exported interfaces and imported interfaces can be intersected. These
intersections define the interoperability layer and constrain the expressivity
of Carbon/C++ interoperability. Our goal is that these expressivity constraints
are wide enough to make the amount of bridge code sustainable and the overhead
of wrappers manageable.

## Interoperability Syntax Elements

This document uses currently unique Carbon syntax elements to indicate
interoperability requirements which have not received any discussion. These
should be discussed further to determine appropriate syntax.

The current text should be treated as placeholders (`$` is a deliberately bad
character choice).

Notable elements are:

- `$extern("Cpp")`: Indicates that Carbon code should be exposed for C++.
  Similarly, `$extern("Swift")` might be used to indicate exposure for Swift at
  some point in the future.
  - This should have `namespace` and `name` parameters, to allow for easy
    migration of C++ APIs to Carbon.
  - This should have a `parent` parameter, to allow for setting C++ parents on
    externalized Carbon structs.
- `import Cpp "<path>"`: Imports API calls from a C++-style #include path.

## Details

### Name mapping

For more details, see [name mapping](name_mapping.md).

### Type mapping

For more details, see [type mapping](type_mapping.md).

### Functions and overload sets

For more details, see
[functions and overload sets](functions_and_overload_sets.md).

### Other syntax

For more details, see [other syntax](other_syntax.md).

### Migration examples

For examples of how this design is intended to function, please see our
[migration examples](migration_examples.md).
