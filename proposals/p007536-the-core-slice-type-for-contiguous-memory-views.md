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
    -   [Carbon](#carbon)
    -   [C++](#c)
    -   [Rust](#rust)
    -   [Go](#go)
    -   [Swift](#swift)
    -   [Safe C++](#safe-c)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Representation](#representation)
    -   [Construction](#construction)
    -   [Size and data pointer](#size-and-data-pointer)
    -   [Element access](#element-access)
    -   [Sub-slicing](#sub-slicing)
    -   [Iteration](#iteration)
    -   [Copying and initialization](#copying-and-initialization)
    -   [Conversions](#conversions)
    -   [Usage Example](#usage-example)
-   [Future work](#future-work)
    -   [Safety](#safety)
    -   [Static bound](#static-bound)
    -   [C++ interop](#c-interop)
    -   [Syntax](#syntax)
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

1.  Passing a reference to
    [`std::vector<T>`](https://en.cppreference.com/w/cpp/container/vector) or
    [`std::array<T, N>`](https://en.cppreference.com/w/cpp/container/array).
    This limits the function to only working with that specific container type
    (and for `std::array`, requires templating on the size `N`).
2.  Passing a raw pointer and a size. This is unsafe, as it relies on the caller
    passing the correct size, and does not carry bounds information natively.
3.  Templates, which can accept any container but lead to code bloat and require
    the function definition to be in the header.

C++20 introduced
[`std::span<T>`](https://en.cppreference.com/w/cpp/container/span) to solve
this, but Carbon needs a native, safe, and idiomatic solution from the start.

## Background

### Carbon

[p004682](/proposals/p004682-the-core-array-type-for-direct-storage-immutably-sized-buffers.md)
introduced `Core.Array(T, N)` (spelled `array(T, N)`), an owning, direct-storage
buffer of a compile-time number of elements. `Core.Slice(T)` is the non-owning
counterpart: it refers to a contiguous sequence of elements stored elsewhere,
such as in an array, and its size is known only at runtime.

Carbon's `str` type (`Core.Str`, per leads issue
[#5969](https://github.com/carbon-language/carbon-lang/issues/5969)) is a
similar non-owning view, specialized for strings: it refers to a sequence of
`char` code units, and is represented as a pointer and a size. See
[p006177](/proposals/p006177-c-interop-mapping-std-string-view-to-core-str.md)
and [p006710](/proposals/p006710-char-redesign.md). `Core.Slice(T)` provides the
same kind of view for elements of any type `T`.

### C++

C++20 [`std::span<T>`](https://en.cppreference.com/w/cpp/container/span) is a
non-owning view over a contiguous sequence of objects. It can have static extent
(size known at compile time) or
[dynamic extent](https://en.cppreference.com/w/cpp/container/span/dynamic_extent)
(size known at runtime). We are primarily concerned with dynamic extent here.

### Rust

Rust uses slice references
[`&[T]`](https://doc.rust-lang.org/reference/types/slice.html) as a fundamental
type. The slice `[T]` itself is a
[dynamically sized type](https://doc.rust-lang.org/reference/dynamically-sized-types.html)
(DST), and `&[T]` is a "fat pointer" containing both the pointer to the data and
the length. For compile-time-sized views, Rust uses references to fixed-size
arrays, [`&[T; N]`](https://doc.rust-lang.org/reference/types/array.html), which
are thin pointers and distinct from `&[T]`.

### Go

Go uses slices [`[]T`](https://go.dev/ref/spec#Slice_types) as a primary data
structure. Go slices are headers containing a pointer to an underlying array, a
length, and a capacity. Slices in Go can be
[grown](https://go.dev/ref/spec#Appending_and_copying_slices) (which may
reallocate), whereas Carbon's `Slice` is a view and cannot be grown.

### Swift

Swift has long provided
[`UnsafeBufferPointer<T>`](https://developer.apple.com/documentation/swift/unsafebufferpointer),
a pointer and count that is only bounds-checked in debug builds and carries no
lifetime information.
[SE-0447](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0447-span-access-shared-contiguous-storage.md)
(Swift 6.2) added
[`Span<T>`](https://developer.apple.com/documentation/swift/span), a safe,
non-owning view of contiguous initialized memory. It is also represented as a
pointer and a count, but is bounds-checked on every access, and is a
non-escapable type, so it cannot outlive the container it borrows from.

### Safe C++

The [Safe C++ proposal](https://safecpp.org/draft.html#tuples-arrays-and-slices)
follows Rust: `[T; N]` is an array, and `[T; dyn]` is a dynamically sized slice
type that is used through a borrow, which is represented as a pointer and a
length.

## Proposal

We propose to introduce the `Core.Slice(T)` type in the `Core` prelude, alongside
a `slice` keyword that maps to `Core.Slice` (such that `slice(T)` evaluates to
`Core.Slice(T)`, similar to `str` mapping to `Core.Str`).

This follows the [All APIs are library APIs principle](/docs/project/principles/library_apis_only.md)
by defining the underlying type in the standard library while providing a
primitive keyword alias.

## Details

### Representation

`Core.Slice(T:! type)` is a class whose value is a pointer to the first element
of the viewed sequence and a count of elements. As with `std::span<T>`, the
bounds are represented as start and count rather than start and end. The fields
are private; the representation is not part of the interface, and may change
(for example, the size may become a word-sized integer, and the pointer may
become an array iterator). In particular, this proposal doesn't guarantee layout
compatibility with `std::span<T>`; see [C++ interop](#c-interop).

### Construction

A slice can be constructed from a pointer to the first element and an element
count:

```carbon
var s: slice(i32) = Core.Slice(i32).Make(p, n);
```

Nothing checks that `p` points to at least `n` valid elements, so this operation
is expected to become unsafe once Carbon has a way to mark operations as such.
Most slices are expected to come from [conversions](#conversions) instead.

### Size and data pointer

`s.Size()` returns the number of elements in the slice, and `s.Ptr()` returns a
pointer to the first element.

### Element access

A slice supports indexing with any subscript type that implicitly converts to
`u64`. `s[i]` refers to the `i`th element of the viewed sequence, and is a
reference expression even if `s` is a value, because the slice doesn't own its
elements. This is provided by implementing `IndirectIndexWith`.

### Sub-slicing

`s.Range(start, count)` returns a slice viewing `count` elements of `s`,
beginning with element `start`.

### Iteration

When `T` is copyable, a slice can be iterated with a `for` loop, which visits
each element in order:

```carbon
for (x: i32 in s) {
  ...
}
```

### Copying and initialization

A slice is cheaply copyable; a copy views the same elements as the original.
Slices also support unformed initialization.

### Conversions

An array type `[T; N]` (written as `array(T, N)` in the prelude) can be implicitly converted to a `Slice(T)` (written as `slice(T)`). The resulting slice points to the first element of the array and has a size of `N`.

This conversion is valid if the elements of the array are compatible with the elements of the slice.

This allows passing an array directly to a function taking a slice:

```carbon
fn PrintSlice(s: slice(i32));

fn Main() -> i32 {
  var data: array(i32, 5) = (1, 2, 3, 4, 5);
  // Implicit conversion from Array to Slice:
  PrintSlice(data);
  return 0;
}
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
  var s: slice(i32) = data;
  PrintSlice(s);
  return 0;
}
```

## Future work

### Safety

This type should eventually support
[spatial](/docs/design/safety/terminology.md#spatial-safety) and [temporal](/docs/design/safety/terminology.md#temporal-safety) [safety](/docs/design/safety/README.md#memory-safety-model)
in some form, similarly applicable to (but not yet specified for)
[`str`](/proposals/p006177-c-interop-mapping-std-string-view-to-core-str.md) -
but the specifics of that safety
(per Carbon's [safety strategy](/docs/project/principles/safety_strategy.md) as
updated in [p005914](/proposals/p005914-updating-carbon-s-safety-strategy.md))
are left out of scope for this proposal, though it's part of the motivation for
the specific direction of this proposal (without safety in mind, perhaps the
pointer+length tuple would be less problematic).

### Static bound

A type similar to `Slice`, but [supporting a compile-time bound](https://docs.google.com/document/d/1hdYyCLmzEOj9gDulm7Eo1SVNc0pY7zbMvFmEzenMhYE/edit?usp=sharing)
should be provided, but may be a [distinct/unrelated type](/proposals/p004682-the-core-array-type-for-direct-storage-immutably-sized-buffers.md#absence-of-syntax-should-make-clear-defaults)
from this one (more like Rust's `&[T; N]` versus `&[T]`, and unlike C++'s
[`std::span`](https://en.cppreference.com/cpp/container/span) that uses a template parameter to
represent either the specific static bound, or the absence of a static bound -
which seems like it might complicate the interface more than providing some
generality over the two scenarios; see also the [`[T; N]` discussion in p004682](/proposals/p004682-the-core-array-type-for-direct-storage-immutably-sized-buffers.md#t-n-builtin-syntax)).

### C++ interop

We expect C++'s `std::span<T>` (with dynamic extent) to map to `slice(T)`,
similar to how
[p006177](/proposals/p006177-c-interop-mapping-std-string-view-to-core-str.md)
maps `std::string_view` to `str`. A zero-cost mapping would require the two
types to have identical representations, which depends on the member order used
by each C++ standard library implementation, and on the size field matching
`size_t` (which is 32 bits on 32-bit targets, whereas `Slice` currently uses
`u64`). The specifics of this mapping, including any constraints it places on
the representation of `Slice`, are left to a future proposal.

A `std::span<T, N>` with static extent would more naturally map to the
[statically bounded type](#static-bound) described above.

### Syntax

[p004682](/proposals/p004682-the-core-array-type-for-direct-storage-immutably-sized-buffers.md#proposal) and
['Pointers Arrays, and Slices'](https://docs.google.com/document/d/1hdYyCLmzEOj9gDulm7Eo1SVNc0pY7zbMvFmEzenMhYE/edit?tab=t.0#heading=h.fahgww8db6f0) refer to some possible builtin syntax (such as `[T]` or `[T]*`) for referring to specific Spans/Slices, but that isn't covered here and is left for future proposals.

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
