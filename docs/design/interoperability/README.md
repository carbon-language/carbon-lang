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
- [Interoperability Syntax Elements](#interoperability-syntax-elements)
- [Details](#details)
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
  - Doing automatic externs for Carbon code would impose language constraints
    that we want to limit to where users need it.
- We do not expect to support all C/C++ corner cases: the complexity of
  supporting any given construct must be balanced by the real world need for
  that support.
  - We may target C++17, and not keep adding interoperability support for later
    C++ features.
  - For example, we might not prioritize support for non-idiomatic "modern"
    code, interfaces, or patterns outside of those in widespread open source
    libraries and used by key contributors.
  - For example, we might not support low-level C ABIs outside of modern 64-bit
    ABIs: Linux, POSIX, and a small subset of Windows' calling conventions.
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

## Interoperability Syntax Elements

This document uses currently unique Carbon syntax elements to indicate
interoperability requirements which have not received any discussion. These
should be discussed further to determine appropriate syntax.

The current text should be treated as placeholders; `$` is a deliberately bad
character choice.

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

C/C++ names are mapped into the `Cpp` Carbon package. C++ namespaces work the
same fundamental way as Carbon namespaces within the `Cpp` package name. Dotted
names are used when referencing these names from Carbon code. For example,
`std::exit` becomes `Cpp.std.exit`.

C++ incomplete types will be mirrored into Carbon's incomplete type behavior.
Users wanting to avoid differences in incomplete type behaviors should fully
define the C++ types using repeated imports.

Carbon names which are mapped into C++ will use a top-level namespace of
`Carbon`, with the package name and namespaces represented as namespaces below
that. For example, the `Widget` Carbon package with a namespace `Foo` would
become `::Carbon::Widget::Foo` in C++.

For more details, see [name mapping](name_mapping.md).

### Type mapping

Carbon and C++, as well as the C subset of C++, will have a number of types with
direct mappings between the languages. The existence of these mappings allow
switching from one type to another across any interface boundary between the
languages. However, this only works across the interface boundary to avoid any
aliasing or other concerns. Transitioning in or out of Carbon code is what
provides special permission for these type aliases to be used.

Also note that the behavior of these types will not always be identical between
the languages. It is only the values that transparently map from one to the
other. Mapping operations is significantly different. For example, Carbon may
have `Float32` match the C++ `float` storage while making subtle changes to
arithmetic and/or comparison behaviors. We will prioritize reflecting the intent
of type choices.

Last but not least, in some cases, there are multiple C or C++ types that can
map to a single Carbon type. This is generally "fine" but may impose important
constraints around overload resolution and other C++ features that would
otherwise "just work" due to these mappings.

For more details, see [type mapping](type_mapping.md).

#### Primitive types

For more details, see [primitive types](primitive_types.md).

#### User-defined types

For more details, see [user-defined types](user_defined_types.md).

#### Vocabulary types

For more details, see [vocabulary types](vocabulary_types.md).

### Enums

For more details, see [enums](enums.md).

### Templates and generics

For more details, see [templates and generics](templates.md and generics).

### Functions and overload sets

Mapping non-overloaded functions between Carbon and C++ is trivial - if the
names are made available, they can be called. Because Carbon may use a different
calling convention, it would need to either emit custom C++ annotations to
coerce calls to use its calling convention directly, or emit wrapper calls that
map in source code between the two. Even the latter is likely to be easily
optimized away. The real difficulty is in mapping the types used on the function
call, which we described in detail above.

However, both Carbon and C++ support function overloading. This is a much more
complex surface to translate between languages. Carbon's overloading is designed
to be largely compatible with C++ so that this can be done reasonably well, but
it isn't expected to be precisely identical. Carbon formalizes the idea of
overload resolution into pattern matching. C++ already works in an extremely
similar way although without the formalization.

For more details, see
[functions and overload sets](functions_and_overload_sets.md).

### Other syntax

Beyond the above in-depth discussions, a few key syntax details to be aware of
are:

- C typedefs are generally mapped to Carbon aliases.
- C/C++ macros that are defined as constants will be imported as constants.
  Otherwise, macros will be unavailable in Carbon.

For more details, see [other syntax](other_syntax.md).

### Migration examples

Large-scale migrations need to be done piecemeal. The goal of migration examples
is to use interoperability in order to produce small migration steps which can
be performed independently to a large codebase, without breaking surrounding
code.

Examples:

- [Incremental migration of APIs](example_incremental.md)
- [Framework API migration](example_framework.md)
