# Goals and philosophy

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Overview](#overview)
- [Goals](#goals)
- [Non-goals](#non-goals)
- [Philosophy of interoperability layer](#philosophy-of-interoperability-layer)

<!-- tocstop -->

## Overview

The goals of Carbon's interoperability layer are heavily influenced by the
[language-level goals](/docs/project/goals.md). Notably, we prioritize
performance, and making any performance overhead visible and opt-in.

## Goals

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
    but potentially non-zero-cost, conversions available to map into Carbon
    vocabulary types.
- Mappings should be easy to maintain.
  - We should provide a syntax for transparently, automatically exposing a
    subset of Carbon types and APIs to C++ code without custom bridge code,
    including instantiating templates.

## Non-goals

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
- Interoperability features should not be expected to work for arbitrary C++
  toolchains. While pre-compiled C++ libraries may be callable, the Carbon
  toolchain should be used to compile both C++ and Carbon in order to achieve
  full support.
  - Carbon must be able to compile C++ headers in order to translate names and
    types.
  - While arbitrary C++ code may be able to call into Carbon code that has been
    pre-compiled into a library, a more complex interaction like C++ code
    calling Carbon templates requires compilation of _both_ languages together.

## Philosophy of interoperability layer

The design for interoperation between Carbon and C++ hinges on:

1. A focus on types, and simple overload sets built from those types.
2. A willingness to expose the idioms of C++ into Carbon code, and the other way
   around, when necessary to maximize performance.
3. The use of wrappers, generic programming, and templates to minimize or
   eliminate runtime overhead.

These things come together when looking at how custom data structures in C++ are
exposed into Carbon, or Carbon into C++. In both languages, it is reasonable and
even common to have customized low-level data structures such as associative
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
