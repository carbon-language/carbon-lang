# The Core.Slice type for contiguous memory views

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7536)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [C++](#c)
    -   [Rust](#rust)
    -   [Go](#go)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Usage Example](#usage-example)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Raw pointer and size](#raw-pointer-and-size)
    -   [Builtin slice syntax only](#builtin-slice-syntax-only)

<!-- tocstop -->

## Abstract

We propose to add `Core.Slice(T)` as a library type in the `prelude` library
of the `Core` package. `Core.Slice(T)` represents a non-owning, safe view
into a contiguous sequence of elements of type `T`. It is implemented as a
pointer to the start of the sequence and a size.

## Problem

In systems programming, we frequently need to write functions that operate on
contiguous sequences of data (like arrays or vectors) without taking ownership
and without copying the data.

In C++, this has historically been done by:

1.  Passing a reference to `std::vector<T>` or `std::array<T, N>`. This limits
    the function to only working with that specific container type (and for
    `std::array`, requires templating on the size `N`).
2.  Passing a raw pointer and a size. This is unsafe, as it relies on the caller
    passing the correct size, and does not carry bounds information natively.
3.  Templates, which can accept any container but lead to code bloat and require
    the function definition to be in the header.

C++20 introduced `std::span<T>` to solve this, but Carbon needs a native, safe,
and idiomatic solution from the start.

## Background

### C++

C++20 `std::span<T>` is a non-owning view over a contiguous sequence of objects.
It can have static extent (size known at compile time) or dynamic extent (size
known at runtime). We are primarily concerned with dynamic extent here.

### Rust

Rust uses slice references `&[T]` as a fundamental type. The slice `[T]` itself
is a dynamically sized type (DST), and `&[T]` is a "fat pointer" containing
both the pointer to the data and the length.

### Go

Go uses slices `[]T` as a primary data structure. Go slices are headers containing
a pointer to an underlying array, a length, and a capacity. Slices in Go can
be grown (which may reallocate), whereas Carbon's `Slice` is a view and cannot
be grown.

## Proposal

We propose to introduce the `Core.Slice(T)` type in the `Core` prelude, alongside
a `slice` keyword that maps to `Core.Slice` (such that `slice(T)` evaluates to
`Core.Slice(T)`, similar to `str` mapping to `Core.String`).

This follows the [All APIs are library APIs principle](/docs/project/principles/library_apis_only.md)
by defining the underlying type in the standard library while providing a
primitive keyword alias.

## Details

The `Core.Slice(T)` type is defined as:

```carbon
package Core library "prelude/types/slice";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators/index";
import library "prelude/types/uint";

private fn PointerOffset[T:! type](p: T*, offset: u64) -> T* = "pointer.offset";

class Slice(T:! type) {
  fn Make(ptr: T*, size: u64) -> Self {
    return {.ptr = ptr, .size = size};
  }

  fn Size(self) -> u64 { return self.size; }
  fn Ptr(self) -> T* { return self.ptr; }

  fn Range(self, start: u64, count: u64) -> Self {
    return Make(PointerOffset(self.ptr, start), count);
  }

  impl as Copy {
    fn Op(self) -> Self { return {.ptr = self.ptr, .size = self.size}; }
  }

  private var ptr: T*;
  private var size: u64;
}

impl forall [T:! type, Subscript: ImplicitAs(u64)]
    Slice(T) as IndirectIndexWith(Subscript)
    where .ElementType = T {
  fn Addr[me: Self](subscript: Subscript) -> T* {
    return PointerOffset(me.Ptr(), subscript as u64);
  }
}



impl forall [T:! type] Slice(T) as UnformedInit {}
```

### Usage Example

```carbon
import Core;

fn PrintSlice(s: slice(i32)) {
  // Access elements by way of indexing:
  var elem: i32 = s[0];

  // Sub-slicing using Range:
  var sub: slice(i32) = s.Range(1, 3);
}

fn Main() -> i32 {
  var data: array(i32, 5) = (1, 2, 3, 4, 5);
  var s: slice(i32) = slice(i32).Make(&data[0], 5);
  PrintSlice(s);
  return 0;
}
```

## Rationale

This proposal advances the following Carbon goals:

-   **Performance-critical software**: Slices are lightweight (typically two words: pointer + size) and passed by value, avoiding allocation and indirection.
-   **Practical safety and testing mechanisms**: Slices carry their size, allowing for runtime bounds checking (in debug modes) to prevent buffer overflows.
-   **Code that is easy to read, understand, and write**: A single type `Slice(T)` replaces various ad-hoc patterns for passing array views.

## Alternatives considered

### Raw pointer and size

We could rely on passing `(T*, u64)` pairs. This is rejected because it is unsafe (no bounds checking possible at the type level), verbose, and not idiomatic.

### Builtin slice syntax only

We could introduce `[T]` as a builtin type without a library representation. This is rejected because it violates the "All APIs are library APIs" principle and makes it harder to define methods or implement interfaces (like `Copy`) on the slice type itself.
