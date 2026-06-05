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

When importing an object-like macro, the tokens of the macro's replacement list
are evaluated as a C++ constant expression in the global C++ namespace, and the
resulting constant value is imported into the `Cpp` Carbon namespace. Its type
is mapped to a Carbon type following the
[Carbon <-> C++ type mapping rules](/proposals/p005448-carbon-c-interop-primitive-types.md),
and its expression category is determined by the C++ value category: lvalues are
imported as references, and rvalues are imported as values.

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
// Cpp.GREEN_COLOR is equal to Cpp.Color.Green, and has type Cpp.Color.
let b: Cpp.Color = Cpp.GREEN_COLOR;

// Cpp.VALUE is an alias to kValue.
let a: i32 = Cpp.VALUE;
```

Note that this means that an imported macro can behave differently in Carbon
when used inside an expression.
[The following C++ program](https://godbolt.org/z/6ndzv764n) prints `7`, since
the macro is expanded before the multiplication operation; `2 * 1 + 2 + 3` is
evaluated as `(2 * 1) + 2 + 3`:

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

> **Future work**: It may be possible to evaluate the macro definition in a
> child namespace, rather than the global C++ namespace.

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
-   Proposal
    [#7308: Clarify support for imported object-like macros](https://github.com/carbon-language/carbon-lang/pull/7308)
