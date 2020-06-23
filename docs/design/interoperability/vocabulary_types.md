# Vocabulary types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Non-owning value types](#non-owning-value-types)
- [Non-owning references and pointers](#non-owning-references-and-pointers)
  - [Slice special-casing](#slice-special-casing)
  - [Mapping similar built-in types](#mapping-similar-built-in-types)
- [Ownership transfer types](#ownership-transfer-types)
  - [Alternative: Bind tightly to particular C++ libraries](#alternative-bind-tightly-to-particular-c-libraries)
- [Copying vocabulary types](#copying-vocabulary-types)

<!-- tocstop -->

There are several cases of vocabulary types that are important to consider and
offer different degrees of flexibility in support:

- Non-owning types passed by value, such as `std::string_view` or `std::span`.
- Non-owning types passed by reference or pointer, such as `std::vector<T> &`.
- Owning types signaling a move of ownership, such as `std::unique_ptr` or
  `std::vector<T> &&`.
- Owning types signaling a copy of data, such as `std::vector<T>`.

Each of these lends itself to different strategies of interoperation between
Carbon and C++.

## Non-owning value types

These are some of the most important types to have direct support for in Carbon
because they are complex and opaque C++ types that are expected to come at zero
runtime cost and refer back to data that will remain managed by C/C++ code. As
such, Carbon prioritizes direct mappings for the most important of these types
at no cost.

The primary idiom from C++ that Carbon attempts to directly support are types
which represent contiguous data in memory. The dominant case here is
`std::span<T>`, but there are a wide variety of similar vocabulary types. Here,
Carbon directly maps these types to a slice type: `T[]`. These types in Carbon
have the same core semantics and can be trivially built from any
pointer-and-size formulation.

There is an important special case: `std::string_view` and similar views into
C++ string data. It is an open question in Carbon whether there is a dedicated
`StringSlice` type or instead it simply uses a direct slice of the underlying
`CodeUnit` or `Char` type. Whatever is the canonical vocabulary used to convey a
slice of a Carbon `String` should also be used as the idiomatic mapping for
`std::string_view`.

Other non-owning value types will get automatic mappings as a use case is
understood to be sufficiently important. We expect the vast majority of
performance critical mappings to end up devolving to slices.

## Non-owning references and pointers

Non-owning references and pointers are perhaps the simplest non-owning types in
C and C++ and have the same critical performance requirements as value types.
Mapping them into Carbon is simple because they have limited and well-known
semantics. By default, these are both mapped to pointers in Carbon:

- References map to non-null pointers, or `T*`.
- C++ pointers map to nullable pointers, or `T*?`.

For example, given a C++ API:

```
Resource* LoadResource(const Location& name);
ResourceEntry& SelectResource(const Selector& sel);
```

We would expect a Carbon call to look like:

```
var Cpp.Location: loc = ...;
This maps the C++ * to a nullable pointer.
var Cpp.Resource*?: res = Cpp.LoadResource(&loc);

var Cpp.Selector: sel = ...;
// This maps the C++ & to a non-nullable pointer.
var Cpp.ResourceEntry*: entry = Cpp.SelectResource(sel);
```

However, there are interesting special cases where it will be advantageous to
promote these types to higher-level types in Carbon to make the interface
boundary cleaner. The currently planned special cases are listed here, and more
can be added as we discover both a compelling need and an effective strategy.

### Slice special-casing

Where possible to convert a reference or pointer to a slice, Carbon will do so
automatically. This should cover common patterns such as
`const std::vector<T> &` -> `T[]` and `const std::vector<T> *` -> `T[]?`.
Specific types that should provide this conversion:

- `const std::vector<T>`
- `std::array<T, N>`
  - This loses some info, and may instead build a compile-time-length slice.

### Mapping similar built-in types

When it is not possible to convert a non-owning reference or pointer to a C++
data structure or vocabulary type into a suitable Carbon type, the actual C++
type will be used. However, its API may not match Carbon idioms or patterns, and
may not integrate with generic Carbon code written against those idioms, or vice
versa.

For sufficiently widely used C++ types, Carbon will provide non-owning wrappers
(preferably using generics) that map between the relevant idioms. This will be a
Carbon wrapper to map from a C++ data type like `std::map` into a Carbon
idiomatic interface, or a C++ wrapper to map from a Carbon data type to the C++
idiomatic interface.

The result is that Carbon data structures and vocabulary types should be no more
foreign in C++ code than Boost or other framework libraries that carefully
adhere to C++ idioms, and similarly C++ types in Carbon code.

## Ownership transfer types

Another special case that is important to optimize for is when ownership of data
is being transferred between C++ and Carbon. This can be tricky to recognize due
to reasonable use of pass-by-value when doing ownership transfer in C++, but
Carbon should recognize as many idioms as possible.

The most fundamental case to handle is `std::unique_ptr`, which fortunately is
easily recognized. It can only signify a transfer of ownership. Here Carbon
should completely translate this transfer of ownership from the C++ heap to the
Carbon heap, including to a heap-owned pointer in Carbon. This in turn requires
the C++ heap to be implemented as part of the Carbon heap, allowing allocations
in one to be deallocated in the other and vice versa. TODO(chandlerc): need to
spell out how the heap works in Carbon to pin down the details here.

The next case is `std::vector<T> &&`. This should get translated to a transfer
of ownership with a Carbon `Array(T)`. These types may not have the same layout,
but it should be easy to migrate data from one to the other, even in the
presence of a small-size-optimized `Array(T)` by copying the data if necessary.
NB: that means that when transferring ownership into or out of Carbon, there is
the possibility of an extra copy. This does not precisely match the contract of
`std::vector` but should be a documented requirement for using Carbon's
automatic type mapping as it is expected to be correct in the overwhelming
majority of cases.

Other vocabulary types similar to `std::vector<T>`, or more generally where the
allocation can be transferred or a small copy be performed, should get similar
automatic mapping with Carbon.

Vocabulary types with significantly more complex data structures are unlikely to
be efficiently convertible and should remain C++ types for efficient access.
Some Carbon types can provide explicit conversion routines when useful, which
may be significantly more expensive, such as re-allocating. A good example here
is `boost::unordered_map<Key, Value>`. We likely do not want to constrain a
Carbon language-provided `Map(Key, Value)` type to match a particular C++
library's layout and representation to allow trivial translation from C++ to
Carbon. However, this also likely isn't necessary. As we outline in the
philosophy above, neither C++ nor Carbon aim to reduce the prevalence of custom
C++ data structures. And we can still provide explicit, but potentially
expensive, conversions when ownership transfer is necessary, rather than using
the non-owning wrappers described previously.

### Alternative: Bind tightly to particular C++ libraries

As an alternative, Carbon's language-provided types could precisely match the
internal representation and implementation of particular C++ libraries. In this
scenario, Carbon's `Map(Key, Value)` would need to precisely match
`boost::unordered_map<Key, Value>`.

Note that this doesn't affect the ability of a Carbon program to use
`boost::unordered_map` independently; it only affects the ability of a Carbon
program to use `boost::unordered_map` in place of `Map`.

Pros:

- Users can convert a popular C++ type to an idiomatic Carbon type without
  conversion overhead.

Cons:

- Carbon could not break cross-language type compatibility without
  unpredictability affecting performance of applications that rely on
  compatibility.
- Carbon would be restricted when trying to evolve the API unless we could get
  the particular library, such as Boost, to change their implementation in a
  matching manner.
  - Carbon and C++ should be expected to have slightly different performance
    nuances: a performance improvement for Carbon might be a slowdown for C++.
- If the particular library, such as Boost, changed their implementation, it
  would either break compiles or corrupt Carbon programs.
  - Users would need to bind to specific releases of the library. We would
    become responsible for matching all development.

## Copying vocabulary types

When a vocabulary type crosses between C++ and Carbon and a copy is a valid
option, an extremely good interoperability story can be provided. Here, we can
in almost all cases completely convert common data structures and vocabulary
types between the languages. The data is being copied anyways and so any
necessary changes to the representation and layout are unlikely to be an
unacceptable overhead.

This strategy should be available for essentially all containers and copiable
vocabulary types in the C++ STL, Abseil, and any other sufficiently widely used
libraries.
