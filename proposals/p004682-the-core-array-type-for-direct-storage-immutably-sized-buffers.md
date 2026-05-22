# The Core.Array type for direct-storage immutably-sized buffers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4682)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Rust](#rust)
    -   [Swift](#swift)
    -   [Safe C++](#safe-c)
    -   [Goals](#goals)
        -   [Privileging the most common type names](#privileging-the-most-common-type-names)
        -   [Absence of syntax should make clear defaults](#absence-of-syntax-should-make-clear-defaults)
        -   [Avoiding confusion with other languages](#avoiding-confusion-with-other-languages)
        -   [Avoiding confusion with other domains](#avoiding-confusion-with-other-domains)
    -   [No predeclared identifiers](#no-predeclared-identifiers)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Future work](#future-work)
    -   [Namespacing the `Core` package](#namespacing-the-core-package)
-   [Alternatives considered](#alternatives-considered)
    -   [`[T; N]` builtin syntax](#t-n-builtin-syntax)
    -   [`array [T; N]` builtin syntax](#array-t-n-builtin-syntax)
    -   [Just the `Core.Array(T, N)` library type](#just-the-corearrayt-n-library-type)
    -   [Implicitly importing `Core.Array(T, N)` to the file scope](#implicitly-importing-corearrayt-n-to-the-file-scope)

<!-- tocstop -->

## Abstract

We propose to add `Core.Array(T, N)` as a library type in the `prelude` library
of the `Core` package. Since arrays are a very frequent type, we propose to
privilege use of this type by providing a builtin keyword `array(T, N)` that
resolves to the `Core.Array(T, N)` type.

## Problem

Carbon's current syntax for a fixed-size, direct storage array (hereafter called
"array") is the provisional `[T; N]` and there is no syntax yet for a
mutably-sized indirect storage buffer (hereafter called "heap-buffer").

Arrays and heap-buffers are some of the most commonly used types, after
fundamental types. The syntax, whatever it is, will be incredibly frequent in
Carbon source code.

We explore and propose a new syntax for arrays that addresses design issues with
the provisional syntax that allows for writing each of the following in clear
ways: slice, compile-time sized slice, array, and pointer to array. And that
leaves clear room for a sibling indirect-storage type.

## Background

We have developed a matrix for enumerating and describing the vocabulary of
owning array and buffer types. Direct refers to an in-place storage buffer, as
with arrays. Indirect refers to heap allocation, where the type itself holds
storage of a pointer to the buffer, as with heap-buffers.

Here we are discussing the location of storage (direct vs indirect) as a way to
categorize types. Indirect-storage types may, for small payloads, store state
directly in its fields (such as with the Small String Optimization), but this is
an optimization for specific payloads and the category of the type remains an
indirect-storage type.

To provide familiarity, here is the table for the C++ language as a baseline:

| Owning type              | Runtime Sized          | Compile-time Sized          |
| ------------------------ | ---------------------- | --------------------------- |
| Direct, Immutable Size   | -                      | `T[N]` / `std::array<T, N>` |
| Indirect, Immutable Size | `std::unique_ptr<T[]>` | `std::unique_ptr<T[N]>`     |
| Indirect, Mutable Size   | `std::vector<T>`       | -                           |

### Rust

The Rust vocabulary is as follows:

| Owning type              | Runtime Sized | Compile-time Sized |
| ------------------------ | ------------- | ------------------ |
| Direct, Immutable Size   | -             | `[T; N]`           |
| Indirect, Immutable Size | `Box<[T]>`    | `Box<[T; N]>`      |
| Indirect, Mutable Size   | `Vec<T>`      | -                  |

There are a few things of note when comparing to C++:

-   The Rust `Box` and `Vec` types are part of `std` but are imported into the
    current scope automatically, so they do not need any prefix.
-   The `[T]` type represents a fixed-runtime-size buffer. The type itself is
    not instantiable since its size is not known at compile time. `Box` is
    specialized for the type to store a runtime size in its own type.
-   The array type syntax matches the Carbon provisional syntax.
-   The heap-buffer type name matches the C++ `vector` type, but it is
    privileged with a shorter name. The `Vec` type name is at most the same
    length as an array type name (for the same `T`).

### Swift

The Swift vocabulary is significantly smaller, to support automatic refcounting:

| Owning type              | Runtime Sized      | Compile-time Sized |
| ------------------------ | ------------------ | ------------------ |
| Direct, Immutable Size   | `InlineArray<T>`   | -                  |
| Indirect, Immutable Size | -                  | -                  |
| Indirect, Mutable Size   | `Array<T>` / `[T]` | -                  |

Because there was historically no direct storage option, only one name was
needed, and "Array" was used to refer to a heap-buffer.

On
[Feb 5 2025](https://forums.swift.org/t/accepted-with-modifications-se-0453-inlinearray-formerly-vector-a-fixed-size-array/77678),
a proposal was accepted to
[introduce `InlineArray<T>` for the direct storage immutably sized array type](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0453-vector.md).
Because "Array" is already taken, the original proposal called this new type
"Vector" in reference to mathematical vectors. The choice of name was
[heavily discussed](https://forums.swift.org/t/second-review-se-0453-vector-a-fixed-size-array/76412)
however, due to the confusion with C++'s `std::vector` and Rust's
`std::vec::Vec`. It was
[provisionally renamed to `Slab`](https://github.com/swiftlang/swift/pull/76438)
but settled on `InlineArray`.

### Safe C++

The [Safe C++ proposal](https://safecpp.org/draft.html#tuples-arrays-and-slices)
introduces array syntax very similar to Rust:

| Owning type              | Runtime Sized         | Compile-time Sized  |
| ------------------------ | --------------------- | ------------------- |
| Direct, Immutable Size   | -                     | `[T; N]`            |
| Indirect, Immutable Size | `std2::box<[T; dyn]>` | `std2::box<[T; N]>` |
| Indirect, Mutable Size   | `std2::vector<T>`     | -                   |

There are a few things of note:

-   While Rust omits a size to indicate the size is known only at runtime, Safe
    C++ uses a `dyn` keyword indicate the same.
-   The heap-buffer type name is unchanged from C++, sticking with `vector`.

### Goals

It will help to establish some goals in order to weigh alternatives against.
These goals are based on the
[open discussion from 2024-12-05](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?usp=sharing&resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA#heading=h.h0tg34pzq5yz),
where we discussed the
[Pointers, Arrays, Slices](https://docs.google.com/document/d/1hdYyCLmzEOj9gDulm7Eo1SVNc0pY7zbMvFmEzenMhYE/edit?usp=sharing)
document.

The goals here are largely informed by and trying to achieve the top-level goal
of
["Code that is easy to read, understand, and write"](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
We define some more specific targets here as relate to the specifics of the
array syntax.

#### Privileging the most common type names

-   "Explicitness must be balanced against conciseness, as verbosity and
    ceremony add cognitive overhead for the reader, while explicitness reduces
    the amount of outside context the reader must have or assume."

The more common it will be for a type to be used, the shorter we would like the
name to be. This follows from the presumption that we weigh conciseness as
increasingly valuable for types that will appear more frequently in Carbon code.

We expect the ordering of frequency in Carbon code to be:

-   fundamental types ≈ tuples >> heap-buffers > arrays >> everything else[^1].

Where fundamental types are: machine-sized integers (8 bit, 16 bit, etc.),
machine-sized floating points, and pointers including slices[^2]. Function
parameters/arguments are an example of tuples.

From this, we derive that we want:

-   Fundamental types and tuples to have the most concise names.
    -   We can lean on special syntax or keywords as needed to make them concise
        but descriptive.
-   Heap-buffers to have a concise name, even more so than arrays.
    -   We could use special syntax or keywords if needed to achieve
        conciseness.
-   Arrays to have a concise name, but they do not need to be comparably concise
    to fundamental types and tuples.
    -   We should try to avoid special syntax.
-   Everything else should be written as idiomatic types with descriptive names.

[^1]:
    "[chandlerc] Prioritize: slices first, then [resizable storage], then compile-time
    sized storage, then everything else is vastly less common. Between those three,
    the difference in frequency between the first two is the biggest." from [open discussion on 2024-12-05](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0)

[^2]:
    Slices are included with fundamental types for simplicity, since they will
    take the place of many pointers in C++, giving them similar frequency to
    pointers, and can be logically thought of as a bounded pointer.

#### Absence of syntax should make clear defaults

One way to write arrays and compile-time-sized slices is like we see in Rust:
`[T; N]` and `&[T; N]`. This suggests a relationship where array is like slice,
and the default form. But they are very different types, rather than a
modification of a single type, and this can be confusing[^3] for developers
learning the language.

[^3]: https://fire.asta.lgbt/notes/a1iay7r3e7or0a59 (content-warning: swearing)

We want to avoid the situation where
[absence of syntax](https://www.youtube.com/watch?v=-Hb-9TUyjoo), such as a
missing pointer indicator, changes the entire meaning of the remaining syntax or
is otherwise confusing.

#### Avoiding confusion with other languages

The most general meaning of "array" is a range of consecutive values in memory.

However in many languages it is used, either in formally or informally, to refer
to a direct-storage, immutably-sized memory range:

-   C,
    [colloquial](https://en.wikibooks.org/wiki/C_Programming/Arrays_and_strings)
-   C++, colloquial (from C) and
    [`std::array`](https://en.cppreference.com/w/cpp/container/array)
-   Go, [colloquial](https://go.dev/tour/moretypes/6)
-   Rust, [colloquial](https://doc.rust-lang.org/std/primitive.array.html)[^4]

In particular, this is the usage in the languages which Carbon will most
frequently interoperate, and/or from which code will be migrated to Carbon and
thus comments and variable names would use these terms in this way.

[^4]:
    Maybe this is more formal than colloquial, but the name is not part of the
    typename/syntax.

Languages which require shared ownership _don't have direct-storage arrays_, so
the same term gets used for indirect storage:

-   Swift, [`Array`](https://developer.apple.com/documentation/swift/array)
    -   As noted earlier, Swift is in the processes of adding a direct-storage
        array called
        [`InlineArray`](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0453-vector.md).
        Backwards compatibility prevents the use of `Array` for this.
-   Javascript,
    [`Array`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array)
-   Java and Kotlin,
    [`ArrayList`](https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html)

And some languages use array to refer to both direct and indirect storage types.

-   Dlang has direct-storage arrays
    [colloqually](https://dlang.org/spec/arrays.html) and the indirect-storage
    [`Array`](https://dlang.org/phobos/std_container_array.html) type.
-   Pascal uses the presence or absence of a size to determine if
    [`Array`](https://www.freepascal.org/docs-html/ref/refsu14.html) uses direct
    (immutably-sized) or indirect (mutably-sized) storage.

In sum, languages which have direct-storage immutably-sized arrays use the term
"array" to refer to those, and most then use a separate name for the
indirect-storage type.

#### Avoiding confusion with other domains

The term "vector" in mathematics refers to a fixed-size set of numbers. This
leads to confusion with the C++ type `std::vector` since it holds a
mutably-sized set of values. Developers coming from other domains must learn a
new and contradictory term of art. The Rust language chose naming that derives
from C++, with `std::vec::Vec`.

These type names conflict with names in mathematical and graphics libraries,
which want to use vector in its mathematical sense. In Rust this leads to `Vec`
for a mutably-sized array, and
[`Vec3`](https://docs.rs/bevy/latest/bevy/prelude/struct.Vec3.html),
[`Vec4`](https://docs.rs/bevy/latest/bevy/prelude/struct.Vec4.html), and so on
for fixed-size mathematical vectors. While not fatal, this does create ambiguity
that must be overcome by developers.

### No predeclared identifiers

Recently the proposal
[p4864: No predeclared identifiers, Core is a keyword](https://docs.carbon-lang.dev/proposals/p4864.html)
clarified a direction for the Carbon language, wherein there will not be
implicit imports from the `Core` library. Anything accessible directly in the
language, rather than through a package name, is done so through a builtin
keyword. This ensures that raw identifier syntax is always available for those
same names in Carbon code.

## Proposal

The
[All APIs are library APIs principle](/docs/project/principles/library_apis_only.md)
states:

> In Carbon, every public function is declared in some Carbon API file.

As such, we propose a `Core` library type for a direct-storage immutably-sized
array, and then a builtin shorthand for referring to that library type.

In line with other languages surveyed above, given the presence of a
direct-storage immutably-sized array in Carbon, we will reserve the unqualified
name "array" for this type. In full, its name is `Core.Array(T, N)`, where `T`
is the type of elements in the array, and `N` is the number of elements. Notably
this leaves room for supporting multi-dimensional arrays by adding further
optional size parameters, either in the `Array` type or in a similar sibling
type.

Here is a provisional vocabulary table to compare with other languages:

| Owning type              | Runtime Sized | Compile-time Sized                 |
| ------------------------ | ------------- | ---------------------------------- |
| Direct, Immutable Size   | -             | `array(T, N)` / `Core.Array(T, N)` |
| Indirect, Immutable Size | ?             | `Core.Box(Array(T, N))`            |
| Indirect, Mutable Size   | `Core.Buf(T)` | -                                  |

Carbon does not have proposed names for heap-allocated storage, so we use some
placeholders here, in order to show where `Array` fits into the picture:

-   `Box(T)` for a heap-allocated `T` value.
-   `Buf(T)` for a heap-buffer of `T` values.

An indirect, immutably-sized buffer does not have a clearly expressible syntax
at the moment. `Box([T])` is the closest fit with the current provisional syntax
for slices. But `[T]` is a sized pointer, which would make this type a
heap-allocated sized pointer, rather than a heap-allocated fixed-size array.
This is in contrast with Rust where `&[T]` is a slice, and `[T]` is a fixed-size
buffer; so it then follows that `Box<[T]>` is a heap-allocated fixed-size
buffer.

Because arrays will be very common in Carbon code, we want to privilege their
usage. There are at least two ways in which we can do so. The first is to
include them in the `prelude` library of the `Core` package. This ensures they
are available in every Carbon file as `Core.Array(T, N)`. The second is by
making the type available through a shorthand without being qualfiied by the
`Core` package name. We propose the `array(T, N)` builtin keyword as that
shorthand.

## Rationale

As this proposal is addressing the question of introducing a new `prelude`
library type in `Core`, it is mostly focused on the goal
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)

This proposal aims to make code easy to understand by using a name that is
consistent across systems programming languages, and avoiding names that have
conflicting meaning. It also uses a standard type syntax, with a type in the
`Core` package, making the type and its documentation maximally discoverable
without requiring special-casing.

We introduced some more specific sub-goals above:

1. Privileging the most common type names

This proposal privileges `Core.Array` as it will appear frequently in code, by
placing it in the `prelude` library. This avoids the need for developers to
`import` another `Core` library in order to access the type.

Since array types are expected to be very frequent, we also propose an `array`
builtin keyword as a shorthand. This is in line with the
[No predeclared identifiers](https://docs.carbon-lang.dev/proposals/p4864.html)
proposal, and uses a lowercase spelling to mark the word as a builtin. The
spelling of `array(T, N)` will resolve to the library type `Core.Array(T, N)`.

In this proposal, we avoid introducing additional syntax (such as with `[T; N]`
or `(1, 2)`) because the frequency of use of arrays will be lower than that of
fundamental types and tuples.

2. Absence of syntax should make clear defaults

We introduce a type name, with a keyword that has a clear relationship to the
generic type name, rather than making arrays look more like slices but without
being a pointer. This is maent to avoid the confusion raised when removing
syntax changes the meaning significantly, and especially in ways that differ
from defaults/options for a single language concept.

3. Avoiding confusion with other languages

We propose using the `Array` type name, and `array` shorthand, in line with how
other languages use the same term. When a direct-storage array type is part of
the language, it's consistently referred to as an "array" without
qualifications.

Most importantly, the name is consistent with the meaning in C++ and its
standard library (`std::array<T, N>`) as well as with Rust, the languages which
we expect Carbon code to interact with the most.

4. Avoiding confusion with other domains

The name `Vector` is a possible choice for a fixed-length set of values, due to
its mathematical meaning, as was originally proposed for the direct-storage
immutably-sized array type in Swift. However any use of the name `Vector` in a
core systems programming language construct is fraught. Either the name is to be
incorrectly confused with a mathematical vector or with a C++ `std::vector`. We
avoid the confusion by avoiding this name.

## Future work

### Namespacing the `Core` package

At this time, the `Core` package remains small, but there will come a time where
the names within need to be split into smaller namespaces. Then the name
`Core.Array`, among others, will become longer and the act of previleging the
name through the `array` keyword will become more pronounced and helpful. At
this time, we don't propose to put `Array` into a namespace in `Core` as there's
no such existing structure to point to yet.

## Alternatives considered

### `[T; N]` builtin syntax

This is the current syntax used by the toolchain, however it had the following
problems raised:

-   It's very similar to the syntax for slices, which is `[T]`, but very
    different in nature, being storage instead of a reference to storage.
-   Given `[T]` is a slice, `[T; N]` would better suit a compile-time-sized
    slice.

The syntax for a slice may also be changed, we discussed
[adding a pointer annotation](https://docs.google.com/document/d/1hdYyCLmzEOj9gDulm7Eo1SVNc0pY7zbMvFmEzenMhYE/edit?tab=t.0#heading=h.fahgww8db6f0)
to it, such as `[T]*` and `[T; N]*`. Some downsides remained:

-   The `[T; N]*` syntax would be a fixed-size slice, rather than a pointer to
    an array. This leaves no room for writing a pointer to an array, which can
    indicate a different intent, that it always includes the full memory range
    of the array. Without this distinction, we can't model both
    `std::span<T, N>` and `std::array<T, N>*` in code migrated from C++ to
    Carbon and would need to collapse these to a single type.
-   Removing the pointer annotation would change the meaning of the type
    expression more then we'd like, since it would change from a slice into an
    array, rather than pointer-to-an-array into an array.

### `array [T; N]` builtin syntax

This introduces a keyword as a modifier of a fixed-size slice, rather than a
builtin forwarding type. While arrays will be very common, it's not clear that
they rise to the level of requiring breaking the languages naming rules (using a
lowercase name) in order to provide a shorthand. And the shorthand is longer in
the end than the `Array(T, N)` being proposed here. So this uses a larger
weirdness budget for privileging the type while achieving less conciseness.

This has a similar issue as with `[T; N]` but in the reverse. Removing the
`array` modifier keyword changes the meaning of the type expression in ways that
are larger than a default/modifier relationship. Fixed-size slices are not the
more-default array.

The use of a lowercase keyword also costs us by preventing users from using the
word `array` in variables, a name which is quite common.

### Just the `Core.Array(T, N)` library type

Providing just the library type is possible, but arrays will be one of the most
common types in Carbon code, as described earlier. Privileging them with a
shorthand that avoids `Core.` will help make Carbon code significantly more
concise, due to the frequency, without hurting understandability. This makes it
worth the tradeoff of putting a name into the file scope (by way of a builtin
type).

### Implicitly importing `Core.Array(T, N)` to the file scope

We considered a direction where some subset of names in the `Core` packages's
`prelude` library are imported automatically to the file scope. This would be
similar to the
[Rust `std::prelude` module](https://doc.rust-lang.org/stable/std/prelude/index.html),
which names aliases that are pulled into the global scope.

Importing names poses challenges for migrating code to Carbon. Names imported
from a library may allow shadowing in other scopes, which would create ambiguity
about the meaning of these names. And names imported from a library would not be
avoidable using the raw identifier syntax: `Array` and `r#Array` would both
refer to the same imported `Core.Array` type. This would present a challenge for
code migrated to Carbon which uses the name `Array` in its own types or
variables. Whereas with a builtin keyword, `array` and `r#array` refer to
different things: The first is the keyword, and the second is a name that the
developer can use freely for other purposes.

The
[No predeclared identifiers, Core is a keyword](https://docs.carbon-lang.dev/proposals/p4864.html)
proposal discusses in more detail why this approach was not taken.
