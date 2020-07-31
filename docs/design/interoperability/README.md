# Bidirectional interoperability with C/C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Interoperability syntax elements](#interoperability-syntax-elements)
-   [Bridge code in Carbon files](#bridge-code-in-carbon-files)
-   [Name mapping](#name-mapping)
-   [Type mapping](#type-mapping)
    -   [Primitive types](#primitive-types)
    -   [User-defined types](#user-defined-types)
    -   [Vocabulary types](#vocabulary-types)
-   [Enums](#enums)
-   [Templates and generics](#templates-and-generics)
    -   [Using C++ templates from Carbon](#using-c-templates-from-carbon)
    -   [Using Carbon templates from C++](#using-carbon-templates-from-c)
    -   [Using Carbon generics from C++](#using-carbon-generics-from-c)
-   [Functions and overload sets](#functions-and-overload-sets)
-   [Other syntax](#other-syntax)
-   [Migration examples](#migration-examples)

<!-- tocstop -->

## Overview

It's critical that Carbon can both access C++ APIs and export C++ APIs.
Supporting C++ interoperability is a
[key requirement for Carbon's goals](/docs/project/goals.md) and is expected to
influence the design of Carbon itself. The interoperability layer also has its
own [goals and philosophy](goals_and_philosophy.md) which guide the design of
individual features.

## Interoperability syntax elements

> References: [Name mapping](name_mapping.md).

An `import` will be sufficient for Carbon to call most C++ APIs, with no changes
to the C++ code. However, special interoperability syntax elements will be
required when exposing Carbon code to C++.

Notable elements are:

-   `$extern("Cpp")`: Indicates that Carbon code should be exposed for C++.
    Similarly, `$extern("Swift")` might be used to indicate exposure for Swift
    at some point in the future.
    -   `namespace` and `name` parameters are provided to override default
        choices, particularly to assist migration of C++ APIs to Carbon. For
        example, `$extern("Cpp", namespace="myproject")`.
    -   The Carbon toolchain will translate externed declarations to C++, and
        they will be available to C++ code through `.6c.h` header files.
-   `import Cpp "path"`: Imports APIs from a C++ header file.

We use the name `Cpp` because `import` needs a valid identifier.

## Bridge code in Carbon files

> TODO: We should allow writing bridge C++ code in Carbon files to ease
> maintenance of compatibility layers. Syntax needs to be proposed, and guard
> rails to prevent overuse/misuse should be considered.

## Name mapping

> References: [Name mapping](name_mapping.md).
>
> TODO: Add a reference for incomplete types when one is created.

C/C++ names are mapped into the `Cpp` Carbon package. C++ namespaces work the
same fundamental way as Carbon namespaces within the `Cpp` package name. Dotted
names are used when referencing these names from Carbon code. For example,
`std::exit` becomes `Cpp.std.exit`.

C++ incomplete types will be mirrored into Carbon's incomplete type behavior.
Users wanting to avoid differences in incomplete type behaviors should fully
define the C++ types using imports.

Carbon names which are mapped into C++ will use a top-level namespace of
`Carbon` by default, with the package name and namespaces represented as
namespaces below that. For example, the `Widget` Carbon package with a namespace
`Foo` would become `::Carbon::Widget::Foo` in C++. This may be renamed for
backwards compatibility for C++ callers when migrating code, for example
`$extern("Cpp", namespace="widget")` for `::widget`.

## Type mapping

Carbon and C/C++ will have a number of types with direct mappings between the
languages.

Where performance is critical, such as primitive types, mappings are required to
have identical memory layout between C++ and Carbon. This is necessary to
provide inteoperability calls without a conversion cost.

The behavior of mapped types will not always be identical; they need only be
similar. For example, we expect Carbon's `UInt32` to map to C++'s `uint32_t`.
While behavior is mostly equivalent, where C++ would use modulo wrapping, Carbon
will instead have trapping behavior.

In some cases, there are multiple C/C++ types that map to a single Carbon type.
This is expected to generally be transparent to users, but may impose important
constraints around overload resolution and other C++ features that would "just
work" with 1:1 mappings.

### Primitive types

> References: [Primitive types](primitive_types.md).

We'll have a simple mapping for C++ primitive types. Conversion methods will
exist for cross-platform 32-bit/64-bit compatibility.

### User-defined types

> TODO: Handling of user-defined types should be addressed.

### Vocabulary types

> References: [Vocabulary types](vocabulary_types.md).

There are several cases of vocabulary types that are important to consider:

-   Non-owning types passed by reference or pointer, such as
    `std::vector<T> &&`.
    -   C++ references map to Carbon non-null pointers, or `T*`.
    -   C++ pointers map to Carbon nullable pointers, or `T*?`.
-   Non-owning types passed by value, such as `std::string_view` or `std::span`.
    -   We copy these to Carbon types with similar semantics. These should have
        trivial construction costs.
-   Owning types signaling a move of ownership, such as `std::unique_ptr`.
    -   We will try to transfer ownership to a Carbon type where possible, but
        may need to copy to the Carbon type in complex cases.
-   Owning types signaling a copy of data, such as `std::vector<T>`.
    -   Copying overhead should be expected and normal, even for a Carbon type.

## Enums

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

## Templates and generics

### Using C++ templates from Carbon

> References: [Templates and generics](templates_and_generics.md).

C++ class templates are directly made available in Carbon. For example, ignoring
allocators and their associated complexity, `std::vector<T>` in C++ would be
available as `Cpp.std.vector(T)` in Carbon.

### Using Carbon templates from C++

> References: [Templates and generics](templates_and_generics.md).

Carbon templates should be usable from C++.

### Using Carbon generics from C++

> References: [Templates and generics](templates_and_generics.md).

Carbon generics will require bridge code that hides the generic. This bridge
code may be written using a Carbon template, changing compatibility constraints
to match.

For example, given the Carbon code:

```carbon
fn GenericAPI[Foo:$ T](T*: x) { ... }

fn TemplateAPI[Foo:$$ T](T*: x) { GenericAPI(x); }
```

We could have C++ code that uses the template wrapper to use the generic:

```cc
CppType y;
::Carbon::TemplateAPI(&y);
```

## Functions and overload sets

> References: [Functions and overload sets](functions_and_overload_sets.md).

Non-overloaded functions may be trivially mapped between Carbon and C++, so long
as their parameter and return types have suitable mappings. If the names are
made available, then they can be called.

However, function overloading is supported in both languages, and presents a
much more complex surface to translate. Carbon's overloading is designed to be
largely compatible with C++ so that this can be done reasonably well, but it
isn't expected to be precisely identical. Carbon formalizes the idea of overload
resolution into pattern matching. C++ already works in an extremely similar way,
although without the formalization. We expect to be able to mirror most function
overloads between the two approaches.

## Other syntax

> References: [Other syntax](other_syntax.md).

Beyond the above in-depth discussions, a few key syntax details to be aware of
are:

-   C typedefs are generally mapped to Carbon aliases.
-   C/C++ macros that are defined as constants will be imported as constants.
    Otherwise, macros will be unavailable in Carbon.

## Migration examples

Large-scale migrations need to be done piecemeal. The goal of migration examples
is to use interoperability in order to produce small migration steps which can
be performed independently to a large codebase, without breaking surrounding
code.

Examples:

-   [Incremental migration of APIs](example_incremental.md)
-   [Framework API migration](example_framework.md)
