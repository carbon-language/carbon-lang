# Other syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C varargs](#c-varargs)
- [C/C++ Inline functions](#cc-inline-functions)
- [Function pointers and functors](#function-pointers-and-functors)
- [Typedefs](#typedefs)
- [Macros](#macros)
- [Alternatives](#alternatives)
  - [Don't support varargs](#dont-support-varargs)

<!-- tocstop -->

## C varargs

C varargs can be called from Carbon with compatible types.

For example, given the C function:

```cc
int printf(const char* format, ...);
```

It can be called from Carbon:

```carbon
import Cpp "<stdio.h>"

Cpp.printf("%d\n", 2);
```

## C/C++ Inline functions

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

C/C++ macros that are compile-time constant will be imported as constants.
Otherwise, macros will be unavailable in Carbon.

For example, given the C code:

```cc
#define BUFFER_SIZE 4096
#define bswap_16(x) _byteswap_ushort(x)

#define LONGLONG(x) x##LL
#define INT64_MAX LONGLONG(0x7FFFFFFFFFFFFFFF)
```

We will provide equivalent Carbon code:

```carbon
// :$$ is the current "const" syntax for Carbon
var Int64:$$ BUFFER_SIZE = 4096;
// bswap_16 is lost because it's not a constant.

// Even though there are macro indirections, this is still constant.
$const Int64: INT64_MAX = 0x7FFFFFFFFFFFFFFF;
```

## Alternatives

### Don't support varargs

It's possible that [C varargs](#c-varargs) may not be common enough to be worth
supporting. This is a simple trade-off, weighing frequently of use versus
incremental complexity. At present, we expect that uses will be frequent enough
that support is valuable.
