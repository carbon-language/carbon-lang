# Importing C/C++ macros

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Namespace](#namespace)
    -   [Constant type](#constant-type)
    -   [Constant value](#constant-value)
    -   [Supported constant expressions](#supported-constant-expressions)
    -   [Empty macros](#empty-macros)
    -   [Implementation](#implementation)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

C/C++ object-like macros are frequently used in APIs of standard and low-level
C++ libraries to define constants (such as error codes in `<errno.h>`). To
support seamless interoperability, an object-like C/C++ macro that evaluates to
a constant expression is imported into Carbon as a constant.

For example:

**C++**:

```cpp
#define BUFFER_SIZE 4096
```

**Carbon**:

```carbon
// Cpp.BUFFER_SIZE is imported as a value of type i32 with a value of 4096.
let a: i32 = Cpp.BUFFER_SIZE;
```

## Details

### Namespace

Imported C++ macros are evaluated in the global `Cpp` namespace and are
accessible under that prefix (for example, `Cpp.BUFFER_SIZE`).

### Constant type

The type of the imported constant is deduced by Clang by evaluating the constant
expression, and then mapped to a Carbon type following the
[Carbon <-> C++ type mapping rules](/proposals/p005448-carbon-c-interop-primitive-types.md).

### Constant value

The value of the constant is deduced by evaluating the tokens of the macro's
replacement list as a C++ constant expression.

### Supported constant expressions

The replacement list in the object-like macro expanding to a constant expression
can contain:

-   **Operators**: arithmetic: `+`, `-`, `*`, `/`; bitwise: `|`, `&`, `^`, `<<`,
    `>>` ; logical: `||`, `&&`; comparison: `<`, `>`, `<=`, `>=`, `==`; casts
    etc, with arbitrary number of operands.

    For example:

    ```cpp
    #define ADDITION 1+2+3
    ```

    However, note that this macro behaves differently in Carbon when used inside
    an expression. [The following C++ program](https://godbolt.org/z/6ndzv764n)
    prints `7`, since the macro is expanded before the multiplication operation;
    `2 * 1 + 2 + 3` is evaluated as `(2 * 1) + 2 + 3`:

    ```cpp
    #include <iostream>

    #define ADDITION 1+2+3

    int main() {
    std::cout << (2 * ADDITION) << '\n';
    }
    ```

    While [the following Carbon program](https://godbolt.org/z/WxvrjYGn6) prints
    `12`, since `Cpp.ADDITION` is treated as a constant with value `6`:

    ```carbon
    import Core library "io";

    import Cpp inline "#define ADDITION 1+2+3";

    fn Run() {
    Core.Print(2 * Cpp.ADDITION);
    }
    ```

-   **Chained macros**: macros that expand to other macros which evaluate to
    constants.

    For example:

    ```cpp
    #define VALUE 123
    #define MY_VALUE VALUE
    ```

-   **Enum constants and `constexpr` variables**: if a macro's replacement list
    refers to a named constant, such as an enum constant or a `constexpr`
    variable, it is imported as an alias rather than as a literal value. This
    allows Carbon to preserve the specific type of the constant (such as `Color`
    in the example below). In the case of `constexpr` variables, importing as an
    alias also preserves addressability (that the constant is an lvalue), which
    would be lost if only the value were imported.

    For example:

    **C++**:

    ```cpp
    enum class Color { Red = 1, Green = 2 };
    #define GREEN_COLOR Color::Green

    constexpr int kValue = 123;
    #define VALUE kValue
    ```

    **Carbon**:

    ```carbon
    // Cpp.GREEN_COLOR is an alias to Cpp.Color.Green which has a type Cpp.Color.
    let b: Cpp.Color = Cpp.GREEN_COLOR;

    // Cpp.VALUE is an alias to kValue.
    let a: i32 = Cpp.VALUE;
    ```

Macros are evaluated in the global namespace (for example `Cpp.VALUE`).

> **Future work**: Evaluating in a child namespace (`Cpp.SomeNamespace.VALUE`)
> may also be possible.

### Empty macros

Macros without a replacement list are not imported into Carbon. They do not have
a Carbon equivalent.

```cpp
#define EMPTY
```

### Implementation

1.  **Name lookup**: When a C++ macro name is encountered in Carbon, it is
    looked up first, before other names. Following C++ rules, this ensures that
    the macro is found even if there is a non-macro entity (such as a variable)
    with the same name.
2.  **Macro import**: If an eligible macro is found, the compiler (effectively)
    attempts to generate a C++ helper declaration and imports it if that
    succeeds:

    ```cpp
    constexpr inline decltype(auto) __carbon_import_MY_MACRO = (MY_MACRO);
    ```

    This delegates parsing and type/value evaluation to Clang.

## Future work

Whether Carbon will support other macro forms is still to be determined:

-   [Object-like macros](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#object-like-macros-1),
    for example whose body names a variable or type
-   [Function-like macros](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#function-like-macros-1)
    that have parameters
-   [Predefined macros](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#predefined-macros-1)
    like `__FILE__` or `__LINE__`

## Alternatives considered

-   [Reusing Swift/C implementation](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#alternatives-considered)
-   [Manually scanning tokens](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#alternatives-considered)
-   [Importing macros that refer to enum constants and `constexpr` variables as constant values, instead of aliases in Carbon](/proposals/p006676-carbon-c-interop-importing-c-c-object-like-macros.md#alternatives-considered)

## References

-   Proposal
    [#6676: Carbon/C++ Interop: Importing C/C++ object-like macros](https://github.com/carbon-language/carbon-lang/pull/6676)
