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
  - [Using C++ templates from Carbon](#using-c-templates-from-carbon)
  - [Using Carbon templates from C++](#using-carbon-templates-from-c)
  - [Using Carbon generics from C++](#using-carbon-generics-from-c)
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
  - Doing automatic externs for Carbon code would impose language constraints
    that we want to limit to where users need it.
- We do not expect to support all C/C++ corner cases: the complexity of
  supporting any given construct must be balanced by the real world need for
  that support.
  - We may target C++17, and not keep adding interoperability support for later
    C++ features.
    - There may be interest in supporting some C++20 features, particularly
      modules. However, exhaustive support should not be assumed.
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

> References: [Name mapping](name_mapping.md) and
> [user defined types](user_defined_types.md).

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
  - This should have a `parent` parameter, to allow for setting C++ parent
    classes on externalized Carbon structs.
- `import Cpp "<path>"`: Imports API calls from a C++-style #include path.

We use the name `Cpp` because `import` needs a valid identifier.

## Details

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
`Carbon`, with the package name and namespaces represented as namespaces below
that. For example, the `Widget` Carbon package with a namespace `Foo` would
become `::Carbon::Widget::Foo` in C++.

### Type mapping

Carbon and C/C++ will have a number of types with direct mappings between the
languages.

The behavior of mapped types will not always be identical; they need only be
similar. For example, we expect Carbon's `Int32` to map to C++'s `int`. While
behavior is mostly equivalent, where C++ would overflow, Carbon will instead
have trapping behavior.

In some cases, there are multiple C/C++ types that map to a single Carbon type.
This is expected to generally be transparent to users, but may impose important
constraints around overload resolution and other C++ features that would "just
work" with 1:1 mappings.

#### Primitive types

> References: [Primitive types](primitive_types.md).

We'll have a simple mapping for C++ primitive types. Conversion methods will
exist for cross-platform 32-bit/64-bit compatibility.

#### User-defined types

> References: [User-defined types](user_defined_types.md).

All user-defined, non-template C/C++ class and struct types are directly
available within Carbon with the exact C++ layout.

For example, given a C/C++ `Circle` class:

```cc
class Circle {
 public:
  double GetArea();
 private:
  double radius_;
};
```

We expect this to behave as a similar Carbon class:

```carbon
package Cpp;

struct Circle {
  fn GetArea() -> Float64;
  private var Float64: radius_;
};
```

Carbon won't have inheritance in the same way that C++ provides inheritance. For
compatibility, we provide the option of explicitly setting a parent for the
externed version of a Carbon struct. This won't affect the Carbon
implementation, and has
[edge cases worth considering](user_defined_types.md#inheriting-from-c-types-with-carbon-structs)–however,
it may often assist in interoperability where C++ inheritance is required.

For example, to declare a Carbon struct `Circle` which, when observed from C++,
inherits from `Shape`:

```carbon
import Cpp "project/shape.h"

$extern("Cpp", parent="Cpp.Shape") struct Circle {
  fn GetArea() -> Float64 { ... };
}
```

#### Vocabulary types

> References: [Vocabulary types](vocabulary_types.md).

There are several cases of vocabulary types that are important to consider:

- Non-owning types passed by reference or pointer, such as `std::vector<T> &&`.
  - C++ references map to Carbon non-null pointers, or `T*`.
  - C++ pointers map to Carbon nullable pointers, or `T*?`.
- Non-owning types passed by value, such as `std::string_view` or `std::span`.
  - We copy these to Carbon types with similar semantics. These should have
    trivial construction costs.
- Owning types signaling a move of ownership, such as `std::unique_ptr` or
  `std::vector<T> &&`.
  - We will try to transfer ownership to a Carbon type where possible, but may
    need to copy to the Carbon type in complex cases.
- Owning types signaling a copy of data, such as `std::vector<T>`.
  - Copying overhead should be expected and normal, even for a Carbon type.

### Enums

> References: [Enums](enums.md).

C++ enums will generally translate nautrally to Carbon, whether using `enum` or
`enum class`. In the other direction, we expect Carbon enums to always use
`enum class`.

For example, here is a C++ and Carbon version of a `Direction` enum, both of
which will be equivalent in either language:

```cc
enum class Direction {
  East,
  West,
  North,
  South,
};
```

```carbon
$extern("Cpp") enum Direction {
  East = 0,
  West = 1,
  North = 2,
  South = 3,
}
```

### Templates and generics

### Using C++ templates from Carbon

> References: [Templates and generics](templates_and_generics.md).

Simple C++ class templates are directly made available as Carbon templates. For
example, ignoring allocators and their associated complexity, `std::vector<T>`
in C++ would be available as `Cpp.std.vector(T)` in Carbon. More complex C++
templates may need explicit bridge code.

### Using Carbon templates from C++

> References: [Templates and generics](templates_and_generics.md).

Carbon templates should be usable from C++: we will add extensions to Clang in
support of this.

### Using Carbon generics from C++

> References: [Templates and generics](templates_and_generics.md).

Carbon generics will require bridge code that hides the generic. This bridge
code may be written using a Carbon template, changing compatibility constraints
to match.

For example, given the Carbon code:

```carbon
fn GenericAPI[Foo:$ T](T*: x) { ... }

fn TemplateAPI[Foo:$$ T](T* x) { GenericAPI(x); }
```

We could have C++ code that uses the template wrapper to use the generic:

```cc
CppType y;
::Carbon::TemplateAPI(&y);
```

### Functions and overload sets

TODO: Re-examine this writing

> References: [Functions and overload sets](functions_and_overload_sets.md).

Non-overloaded functions may be trivially mapped between Carbon and C++. If the
names are made available, then they can be called. Carbon may use a different
calling convention, which may require wrappers or custom C++ annotations to
match, but either is likely to be minor and optimized away.

However, function overloading is supported in both languages, and presents a
much more complex surface to translate. Carbon's overloading is designed to be
largely compatible with C++ so that this can be done reasonably well, but it
isn't expected to be precisely identical. Carbon formalizes the idea of overload
resolution into pattern matching. C++ already works in an extremely similar way,
although without the formalization. We expect to be able to mirror most function
overloads between the two approaches.

### Other syntax

> References: [Other syntax](other_syntax.md).

Beyond the above in-depth discussions, a few key syntax details to be aware of
are:

- C typedefs are generally mapped to Carbon aliases.
- C/C++ macros that are defined as constants will be imported as constants.
  Otherwise, macros will be unavailable in Carbon.

### Migration examples

Large-scale migrations need to be done piecemeal. The goal of migration examples
is to use interoperability in order to produce small migration steps which can
be performed independently to a large codebase, without breaking surrounding
code.

Examples:

- [Incremental migration of APIs](example_incremental.md)
- [Framework API migration](example_framework.md)
