# Bidirectional interoperability with C and C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Philosophy and goals](#philosophy-and-goals)
-   [Overview](#overview)
-   [C++ interoperability model: introduction and principles](#c-interoperability-model-introduction-and-principles)
    -   [The successor language mandate](#the-successor-language-mandate)
    -   [The C++ interop type](#the-c-interop-type)
-   [Importing C++ APIs into Carbon](#importing-c-apis-into-carbon)
    -   [Importing C++ libraries (header-based)](#importing-c-libraries-header-based)
    -   [TODO: Importing C++ code (inline)](#todo-importing-c-code-inline)
    -   [Accessing built-in C++ entities (file-less)](#accessing-built-in-c-entities-file-less)
    -   [The `Cpp` package](#the-cpp-package)
    -   [Importing C++ macros](#importing-c-macros)
-   [Calling C++ code from Carbon](#calling-c-code-from-carbon)
    -   [Function call syntax and semantics](#function-call-syntax-and-semantics)
    -   [TODO: Overload resolution](#todo-overload-resolution)
    -   [TODO: Constructors](#todo-constructors)
    -   [TODO: Struct literals](#todo-struct-literals)
-   [TODO: Accessing C++ classes, structs, and members](#todo-accessing-c-classes-structs-and-members)
-   [TODO: Accessing global variables](#todo-accessing-global-variables)
-   [Bi-directional type mapping: primitives and core types](#bi-directional-type-mapping-primitives-and-core-types)
    -   [TODO: Integers](#todo-integers)
    -   [Literals](#literals)
    -   [Character types](#character-types)
        -   [References](#references)
-   [TODO: Advanced type mapping: pointers, references, and `const`](#todo-advanced-type-mapping-pointers-references-and-const)
-   [TODO: Bi-directional type mapping: standard library types](#todo-bi-directional-type-mapping-standard-library-types)
-   [TODO: The operator interoperability model](#todo-the-operator-interoperability-model)

<!-- tocstop -->

## Philosophy and goals

The C++ interoperability layer of Carbon allows a subset of C++ APIs to be
accessed from Carbon code, and similarly a subset of Carbon APIs to be accessed
from C++ code. This requires expressing one language as a subset of the other.
Bridge code may be needed to map some APIs into the relevant subset, but the
constraints on expressivity should be loose enough to keep the amount of such
bridge code sustainable.

The [interoperability philosophy and goals](philosophy_and_goals.md) provide
more detail.

## Overview

Carbon's bidirectional interoperability with C++ is
[a cornerstone of its design](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code),
enabling a gradual transition from existing C++ codebases. The goal is not just
a foreign function interface (FFI), but a seamless, high-fidelity integration
that supports advanced C++ features, from templates to class hierarchies.

C++ APIs are imported into Carbon using an `import Cpp` directive, which makes
C++ declarations available within a dedicated `Cpp` package in Carbon. This
prevents name collisions and makes the origin of symbols explicit. Carbon code
can then call C++ functions, instantiate C++ classes, and use C++ types, while
respecting C++'s semantics, including its complex overload resolution rules and
preserving the nominal distinctions between C++ types like `long` and
`long long`, or `T*` and `T&`, which is critical for correct overload resolution
and template instantiation.

Similarly, Carbon APIs can be designed to be callable from C++. The
interoperability layer is designed to be zero-cost, avoiding unnecessary
allocations or copies when calling between the two languages.

## C++ interoperability model: introduction and principles

### The successor language mandate

The design of Carbon's C++ interoperability is governed by its foundational
goal: [to be a successor language](/README.md), not merely a language with a
foreign function interface (FFI). This mandate dictates a design that moves
beyond the C-style FFI adopted by most modern languages and instead provides
seamless, bidirectional interoperability. The objective is to support deep
integration with existing C++ code, encompassing its most complex features, from
inheritance to templates.

This goal has profound implications for the Carbon compiler and language
semantics. It requires that C++ is not treated as a foreign entity. Instead,
Carbon's semantic model must be _co-designed_ to understand, map, and interact
with C++'s semantic constructs—including templates, class hierarchies, and
complex overload resolution—with high fidelity. The interoperability layer must,
therefore, operate at the semantic analysis level, not just at the linking (ABI)
level. This document specifies the design of this semantic contract.

### The C++ interop type

A core mechanism in this design is the C++ interop type. This concept defines
the "trigger" that activates C++-specific semantic rules within the Carbon
compiler. Any operation involving a type that is designated as a C++ interop
type could invoke the specialized interoperability logic, such as C++ overload
resolution or operator overload resolution that involves both Carbon and C++
operator overloads.

A type is considered a C++ interop type if its definition involves an imported
C++ type in any of the following ways:

1.  A C++ imported type (for example, `Cpp.Widget`).
2.  A pointer to a C++ interop type (for example, `Cpp.Widget*`).
3.  A Carbon generic type parameterized with a C++ interop type (for example,
    `MyCarbonVector(Cpp.Widget)`).

More generally, a C++ interop type is any type for which Carbon's
[orphan rule](https://docs.carbon-lang.dev/docs/design/generics/details.html#orphan-rule)
would allow an impl to be provided by a library in `package Cpp`.

This "pervasive" model of C++-awareness is a fundamental design choice. The C++
semantics are not confined to a specific `unsafe` or `extern "C++"` block; they
affect any Carbon type that composes them. For example, when the Carbon compiler
instantiates a _Carbon_ generic type like `MyCarbonVector(Cpp.Widget)`, its type
system must be aware that the `Cpp.Widget` parameter carries C++-specific rules.
This mandates that Carbon's own generic system, struct layout logic, overload
resolution and operator lookup must query the type system for the presence of a
C++ interop type. If present, Carbon must consider C++ rules when operating over
C++ interop types. This design prioritizes the goal of a seamless and intuitive
user experience.

## Importing C++ APIs into Carbon

### Importing C++ libraries (header-based)

The primary mechanism for importing existing, user-defined C++ code is through
header file inclusion. Carbon must be able to parse and analyze C++ header files
to make their declarations available within Carbon.

**Syntax:** The syntax for this operation is `import Cpp library "header_name"`.
This syntax is used for both standard library headers and user-defined headers:

-   **Standard Library:**

    ```carbon
    import Cpp library "<cstdio>";
    ```

    This import makes entities like `putchar` available.

-   **C++ User-Defined Header:**
    ```carbon
    import Cpp library "circle.h";
    ```
    This import makes user-defined declarations and definitions available.

### TODO: Importing C++ code (inline)

### Accessing built-in C++ entities (file-less)

Some C++ entities, particularly built-in primitive types, are not defined in any
header file. They are "intrinsic" to the C++ language. These entities are
available in Carbon without an explicit `import` declaration.

### The `Cpp` package

A critical design choice for managing C++ imports is the mandatory use of a
containing package, `Cpp`. All imported C++ named entities (functions, types,
namespaces) are contained in the `Cpp` package.

-   **Functions:** `Cpp.putchar(...)`
-   **Classes/Types:** `Cpp.Circle`, `Cpp.Point`
-   **Constructors:** `Cpp.Circle.Circle()`

The `Cpp.` prefix makes the _origin_ of every symbol explicit and unambiguous.
It ensures that C++ entities cannot collide with Carbon code.

### Importing C++ macros

An object-like C/C++ macro that evaluates to a constant expression is imported
from C++ as a constant in Carbon.

See [Importing C/C++ macros](macros.md) for details.

## Calling C++ code from Carbon

### Function call syntax and semantics

Once imported, C++ functions are invoked using standard Carbon function call
syntax, prefixed with the `Cpp` name. The Carbon compiler is responsible for
mapping the Carbon arguments to the types expected by the C++ function's
signature.

This often requires explicit casting on the Carbon side, using the `as` keyword,
to satisfy the C++ function's parameter types.

**Example:** The following example imports `cstdio` and calls the C function
`putchar`. The Carbon `Core.Char` variable `n` must be cast first to `u8` and
then to `i32` to match the `int` parameter expected by `putchar`.

```carbon
import Cpp library "<cstdio>";

fn Run() {
  let hello: array(Core.Char, 6) = ('H', 'e', 'l', 'l', 'o', '!');
  for (n: Core.Char in hello) {
    // Carbon 'as' casting is used to match the C++ signature
    Cpp.putchar((n as u8) as i32);
  }
}
```

### TODO: Overload resolution

### TODO: Constructors

### TODO: Struct literals

## TODO: Accessing C++ classes, structs, and members

## TODO: Accessing global variables

## Bi-directional type mapping: primitives and core types

### TODO: Integers

### Literals

Carbon and C++ support bi-directional mapping for the types of integer and
floating-point literals. Because Carbon literals do not have suffixes, they
typically follow C++ rules for unsuffixed decimal literals, but can also be
explicitly cast. C++ literals map to corresponding Carbon types based on their
suffixes.

For details, see [literals](literals.md).

### Character types

Carbon's `char` type transparently maps to C++'s `char` type. Carbon's `char`
type is always unsigned. When compiling C++ using Carbon's toolchain, C++ `char`
is treated as `unsigned` by default (`-funsigned-char`). When interoperating
with a signed C++ `char` type (`-fno-unsigned-char`), Carbon will maintain
interoperability, though bits will be interpreted differently in each language.

#### References

-   Proposal
    [#6710: `char` redesign](https://github.com/carbon-language/carbon-lang/pull/6710)

## TODO: Advanced type mapping: pointers, references, and `const`

## TODO: Bi-directional type mapping: standard library types

## TODO: The operator interoperability model
