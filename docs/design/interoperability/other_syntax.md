# Other syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Methods](#methods)
  - [C varargs](#c-varargs)
  - [C Inline functions](#c-inline-functions)
- [Function pointers and functors](#function-pointers-and-functors)
- [Typedefs](#typedefs)
- [Macros](#macros)

<!-- tocstop -->

## Methods

### C varargs

C varargs can be called from Carbon with compatible types.

For example, given the C function:

```cc
int printf(const char* format, ...);
```

It can be called from Carbon:

```carbon
import Cpp "<stdio.h>"

C.printf("%d\n", 2);
```

### C Inline functions

C inline functions will be treated
[as Swift does](https://github.com/apple/swift/blob/master/docs/HowSwiftImportsCAPIs.md#inline-functions),
because the caller is responsible for emitting a definition of the function:

> Therefore, the Swift compiler uses Clang’s CodeGen library to emit LLVM IR for
> the C inline function. LLVM IR for C inline functions and LLVM IR for Swift
> code is put into one LLVM module, allowing all LLVM optimizations (like
> inlining) to work transparently across language boundaries.

## Function pointers and functors

> TODO: A design for C function pointers, member function pointers, and functors
> should be added. This is an acknowledged gap in the current design.

## Typedefs

C typedefs are generally mapped to Carbon aliases, except for a few common C
patterns that are recognized by the C importer and are handled in a special way.

For example, given the C code:

```cc
// Regular typedef.
typedef int Money;

// A special case typedef pattern matched by the C importer.
typedef struct {
  int x;
  int y;
} Point;
```

This will generate similar Carbon code:

```carbon
alias Money = Int64;

struct Point {
  var Int32: x;
  var Int32: y;
}
```

## Macros

C/C++ macros that are defined as constants will be imported as constants.
Otherwise, macros will be unavailable in Carbon.

For example, given the C code:

```cc
#define BUFFER_SIZE 4096
#define bswap_16(x) _byteswap_ushort(x)
```

We will provide equivalent Carbon code:

```carbon
// $const syntax is still under discussion, thus the '$'.
$const Int64: BUFFER_SIZE = 4096;
// bswap_16 is lost because it's not a constant.
```
