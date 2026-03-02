# C++ Interop: API importing and semantics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6358)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [The `Cpp` package and namespace mapping](#the-cpp-package-and-namespace-mapping)
    -   [`import Cpp library` directive](#import-cpp-library-directive)
    -   [C++ built-in types](#c-built-in-types)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

This proposal defines the concrete technical mechanisms for C++
interoperability. It specifies the precise syntax and semantics for importing
C++ APIs. This includes the `import Cpp library "..."` and implicitly importing
C++ built-in entities, and the establishment of the `Cpp` package as the
dedicated namespace for all imported entities.

## Problem

While Carbon has a stated goal of seamless C++ interoperability, and a
high-level direction has been agreed upon, there is currently no concrete,
specified mechanism for developers to actually import and use C++ APIs. This
proposal aims to address that by defining the specific syntax and semantics for
C++ interoperability.

## Background

One of Carbon's primary goals is to be a successor language. This strategy is
entirely dependent on seamless, bidirectional interoperability with C++ to
enable large-scale adoption and migration for existing C++ codebases.

This proposal provides the necessary details on how C++ APIs should be imported.

## Proposal

We propose to formalize the following specific design elements for C++
interoperability:

1.  **The `Cpp` Package:** All imported C++ entities, whether from built-ins or
    library headers (see below), will be nested within a dedicated `Cpp`
    package. This prevents name collisions with Carbon code and makes the
    language boundary explicit.

    ```carbon
    fn UseCppTypes() {
      // Access C++ types and functions by way of the Cpp package
      var circle: Cpp.Circle = Cpp.GenerateCircle();
      Cpp.PrintCircle(circle);
    }
    ```

2.  **Importing C++ Header-Defined APIs:** To import C++ APIs from a specific
    library header file (for example, `<vector>` or `"my_library.h"`), Carbon
    code will use the `import Cpp library "..."` directive.

    ```carbon
    import Cpp library "<vector>";
    import Cpp library "circle.h";
    ```

3.  **Importing C++ Built-in Entities:** To access fundamental C++ types (such
    as `int`, `bool`, etc.), no explicit importing is needed and writing
    `Cpp.int` and `Cpp.bool` would just work.

## Details

### The `Cpp` package and namespace mapping

All C++ declarations will be imported into the `Cpp` package. C++ namespaces
will be mapped to nested packages within `Cpp`. For example, a C++ function
`std::string::find` would be accessible in Carbon as `Cpp.std.string.find`. The
C++ global namespace will be mapped to the `Cpp` package itself. So a function
`MyGlobalFunction` in the C++ global namespace will be `Cpp.MyGlobalFunction` in
Carbon.

### `import Cpp library` directive

The `import Cpp library "..."` directive will instruct the Carbon compiler to
parse the specified C++ header file. The compiler will use the standard C++
include paths to locate the header. Additional paths can be provided through
compiler flags.

The Carbon compiler will leverage a C++ front-end, like Clang, to parse the
headers. This ensures a high degree of compatibility with existing C++ code.
Only the declarations from the header will be imported, not the definitions
(unless they are inline).

### C++ built-in types

A set of fundamental C++ types will be available within the `Cpp` package
without any explicit `import` directive. Mapping examples:

| C++ Type       | Carbon Type        |
| -------------- | ------------------ |
| `int`          | `Cpp.int`          |
| `unsigned int` | `Cpp.unsigned_int` |
| `double`       | `Cpp.double`       |
| `float`        | `Cpp.float`        |
| `bool`         | `Cpp.bool`         |
| `char`         | `Cpp.char`         |

This automatic availability of built-in types is designed to make basic
interoperability tasks as smooth as possible.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   **Explicitness and Clarity:** The `import Cpp library "..."` directives
        make all dependencies on C++ headers.
    -   **Preventing Name Collisions:** The `Cpp` package is a critical design
        element. It provides a clean, unambiguous namespace for all imported C++
        code.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This proposal defines a foundation for seamless C++ interoperability.

## Alternatives considered

-   **Alternative: Explicitly importing built-ins:** We considered making C++
    built-in types (like `int`) require some `import Cpp` directive like
    `import Cpp;`.
    -   **Reason for Rejection:** Since `Cpp` is a special package, it should be
        implicitly imported, similar to Carbon's prelude.
