# C++ Interop: Mapping pointer types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6357)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Non-nullable pointers to object types](#non-nullable-pointers-to-object-types)
        -   [Inconsistent type sugar](#inconsistent-type-sugar)
    -   [Non-nullable pointers to void](#non-nullable-pointers-to-void)
    -   [Nullable pointers and `Core.Optional`](#nullable-pointers-and-coreoptional)
    -   [`nullptr_t`](#nullptr_t)
        -   [`nullptr`](#nullptr)
    -   [Indexing and pointer arithmetic](#indexing-and-pointer-arithmetic)
-   [Rationale](#rationale)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)
    -   [Map all pointers to `T*`](#map-all-pointers-to-t)
    -   [Map `void*` to `()*`](#map-void-to-)
    -   [Map `void*` to `u8*`](#map-void-to-u8)
    -   [Map `void f()` to `fn Cpp.f() -> Cpp.void`](#map-void-f-to-fn-cppf---cppvoid)
    -   [Use the name `Core.CppCompat.Void` instead of `Core.CppCompat.VoidBase`](#use-the-name-corecppcompatvoid-instead-of-corecppcompatvoidbase)
    -   [Map `Cpp.void` to `()` instead of `Core.CppCompat.VoidBase`](#map-cppvoid-to--instead-of-corecppcompatvoidbase)
    -   [Do not provide `Cpp.void` at all](#do-not-provide-cppvoid-at-all)

<!-- tocstop -->

## Abstract

This proposal defines direct, zero-overhead mappings from C++ object pointer
types and `std::nullptr_t` to corresponding Carbon types.

## Problem

In order to support interoperability between C++ and Carbon, we need to map
types in each language to the other language. Currently we do not have a defined
mapping for pointer types, which are an important concept in both languages.
However, a direct mapping is not appropriate, as pointer types have different
semantics in Carbon versus in C++.

## Background

Pointer types in Carbon and in C++ have different semantics. C++ pointers can be
null, can be indexed if they point into an array, treat any non-array object as
pointing to an array of a single element for indexing purposes, and can point to
a position one past the last element of an array. Carbon pointers allow none of
these things, and always point to an object.

There are three kinds of pointer type in C++:

-   A _pointer to object type_ is a pointer whose pointee type is an object
    type, such as a scalar type or a class type.
-   A _pointer to void_ is a pointer type whose pointee type is (possibly
    `const` and/or `volatile`) `void`.
    -   Collectively, pointer to object types and pointer to void types are
        called _object pointer types_, despite `void` not being an object type.
-   A _pointer to function type_, or equivalently _function pointer type_, is a
    pointer whose pointee type is a function type.

C++ pointers to object types support arithmetic operations:

-   Addition and subtraction are supported, treating pointers as an affine space
    over the integers. Formally, arithmetic is only permitted between pointers
    that point to elements of the same array object, but in practice these
    operations are used substantially more broadly.
-   Pointers can be compared relationally, if they point to subobjects of the
    same object. The exact rules are a little more subtle than this, but
    formally, comparisons across distinct complete objects have unspecified
    results and most comparisons within a complete object have specified
    results. In practice, pointer comparisons on modern architectures are
    treated as a total order, except that the status of comparisons of
    bitwise-equal pointers where one points past the end of one complete object
    and the other points to the start of a distinct complete object are
    nebulous.
-   Pointers support array indexing notation `p[i]` -- and, to the surprise of
    many and the delight of few, `i[p]` -- which is mostly equivalent to
    `*(p + i)`.

In addition, the type of `nullptr`, known commonly by its standard library alias
`std::nullptr_t`, is important to modern C++ as a mechanism for forming null
pointer values. This type is not a pointer type, but implicitly converts to
every pointer type, forming a null pointer value of that type.

C++ also has pointers to members, `T C::*`, which are a distinct type from
pointer types in C++.

## Proposal

This proposal defines a mapping from C++ object pointer types and
`std::nullptr_t` into Carbon types, introducing new Carbon types as necessary to
provide the mapping:

| C++ type         | Carbon type                                     | Notes                                       |
| ---------------- | ----------------------------------------------- | ------------------------------------------- |
| `T*`             | `Core.Optional(T*)`                             | null pointers map to `None`                 |
| `T* _Nonnull`    | `T*`                                            | non-nullable pointers don't need `Optional` |
| `const T*`       | `Core.Optional(const T*)`                       |                                             |
| `void*`          | `Core.Optional(Core.CppCompat.VoidBase*)`       | `void` maps to `()` in some other contexts  |
| `void* _Nonnull` | `Core.CppCompat.VoidBase*`                      |                                             |
| `const void*`    | `Core.Optional(const Core.CppCompat.VoidBase*)` |                                             |
| `nullptr_t`      | `Core.CppCompat.NullptrT`                       |                                             |

Function pointer types and pointer to member types are out of scope for this
propopsal.

## Details

### Non-nullable pointers to object types

C++ doesn't have non-nullable pointer types, but C++ implementations do, in
various forms:

| C++ type                                  | Supported by | Behavior if null |
| ----------------------------------------- | ------------ | ---------------- |
| `T* _Nonnull`                             | Clang        | Erroneous        |
| `T* __attribute__((nonnull))`             | Clang        | Undefined        |
| `void f(T*) __attribute__((nonnull))`     | GCC, Clang   | Undefined        |
| `T* f() __attribute__((returns_nonnull))` | GCC, Clang   | Undefined        |
| `void f(_Notnull_ T*)`                    | MSVC         | Defined          |
| `_Ret_notnull_ T* f()`                    | MSVC         | Defined          |

We will support and encourage use of Clang's `_Nonnull` annotation, as it is the
only form that applies to a general pointer type rather than to a function
parameter or return type. Clang's `T* _Nonnull` maps to Carbon's `T*`.

The `__attribute__((nonnull))` and `__attribute__((returns_nonnull))` forms will
also be mapped into non-nullable Carbon pointers, as these attributes are
widespread in existing code. A pointer type that is the type of a function
parameter or the return type of a function that is annotated with these
attributes maps to Carbon's `T*`.

The MSVC attributes are intended for use by static analysis tools only, not as
compiler inputs, so are not suitable for our uses, and are mentioned here only
for completeness.

#### Inconsistent type sugar

The `_Nonnull` annotation is treated as _type sugar_, not as producing a
different type. This has some significant consequences for its use:

-   Type sugar is easily lost through incautious type navigation. Any operation
    that maps to a canonical type will lose the annotation. This is largely just
    a constraint that the toolchain implementation be careful, except...
-   ... type sugar is lost across template instantiations. There is work in
    Clang to preserve type sugar across template instantiations, but
    [at this time it is not complete](https://godbolt.org/z/hP35776zj). This
    means that in templates, the distinction between non-nullable pointers and
    nullable pointers should be expected to be lost.
-   The annotation can differ between redeclarations of the same entity.
    Inconsistent annotations may lead to a type being treated as either nullable
    or non-nullable. Clang will warn on this in some cases, but the fragility of
    type sugar means that such warnings should not be expected to be entirely
    robust.

The consequences in C++ for losing or incorrectly determining nullability are
mostly not too severe -- loss of best-effort diagnostics, and the compiler
treating the program as having defined behavior when the rules say its behavior
is undefined. But if it causes a type to map to a different type in Carbon, that
is potentially a larger issue, as it may affect whether the Carbon program
compiles.

For now we will use these type sugar anontations to inform our type mapping, but
will revisit this decision if they are too problematic in practice.

### Non-nullable pointers to void

`void` broadly has two different meanings in C++:

-   A complete type with an empty representation, a size and alignment of 1, and
    a single unique value. This is the meaning that is used by `void f()`,
    `static_cast<void>(expr)`, and arguably by the special case `int f(void)`.
-   An incomplete type that is a supertype of all other types (sometimes known
    as a top type ⊤, or a universal type). This is the meaning that is used by
    `void*`, as well as related ideas like `dynamic_cast<void*>`. While C++
    models this form of `void` as incomplete, it would be more accurate to
    consider it to be abstract.

There have been long-lived attempts to replace the second meaning in C++ with
the first, but so far they have not succeeded. If they do succeed, our approach
will need to shift to accommodate that change.

We map the first kind of `void` to Carbon's `()` type. In particular, a function
that returns `void` in C++ returns `()` in Carbon. While these types don't have
the same representation in general, they do have the same representation as a
return type, with both types corresponding to nothing being returned.

For the second kind of `void`, we introduce a new compatibility type:

```carbon
abstract class Core.CppCompat.VoidBase {}
```

Notionally, we think of every Carbon type, including types imported from C++, as
inheriting from this type, but practically we support a conversion from any
pointer type to a pointer to `VoidBase`, and we support a conversion from a
value of any type to a value of type `VoidBase`. Other inheritance-based
language properties should decide whether to treat `VoidBase` as a supertype of
all other types by considering how C++ treats `void` in similar contexts.

This new type is our primary mapping for `void`; the mapping of `void` return
types is treated as a special case that applies only in return position. Because
`void` in C++ can only appear in very limited positions, this means that
`VoidBase` is used as the mapping for `void` in only the following situations:

-   As the pointee type of a pointer type.
-   As a type template argument.
-   As a type of a typedef or type alias.

This seems like the best balance: for example, given

```c++
typedef void OpaqueObject;
void Call(OpaqueObject* handle);
```

... it seems useful to map `Cpp.OpaqueObject*` to `VoidBase*`, not to `()*`. And
given:

```c++
template<typename T> T ReturnAT() { return T(); }
```

... the call `Cpp.ReturnAT(VoidBase)` will still have a return type of `()`
rather than an abstract return type, because the type appeared in return
position after instantiation.

As a convenience shorthand, the name `Cpp.void` is added to the `Cpp` package
and refers to `Core.CppCompat.VoidBase`. C++'s `void* _Nonnull` maps to
`Cpp.void*`, and similarly `const void* _Nonnull` maps to `const Cpp.void*`.

### Nullable pointers and `Core.Optional`

Nullable object pointers, including nullable pointers to `void`, map to
`Core.Optional(T*)`. Null pointer values map to the optional's "none" value.
This places a constraint on the implementation of `Core.Optional(T*)` that it
has the same ABI as a C++ pointer, including that the "none" state is
represented with a C++ null pointer representation.

While we do not yet have an approved design for the `Core.Optional` type, it is
already in use in the design of `for` statements. Adding a design for
`Core.Optional` is out of scope for this proposal.

### `nullptr_t`

The C++ null pointer type, `decltype(nullptr)`, is exposed in the standard
library as `std::nullptr_t`. This type is a scalar type built into the C++
language, and as such, we need a custom mapping for it -- our mappings for other
categories of C++ types don't cover it.

C++'s `nullptr_t` has the same representation as `void*`, but its representation
comprises only padding bits. This doesn't correspond to any existing type in
Carbon, so we introduce a new compatibility type to model it:

```carbon
class Core.CppCompat.NullptrT {
  adapt MaybeUnformed(VoidBase*);
}
```

The C++ type maps to the above type, so after `import Cpp library "<cstddef>";`,
it can also be named as `Cpp.std.nullptr_t`.

#### `nullptr`

The name `Cpp.nullptr` is added to the `Cpp` package, and refers to the (empty)
constant value of type `Core.Cpp.NullptrT`.

### Indexing and pointer arithmetic

This proposal provides no support for indexing or pointer arithmetic on Carbon
pointers mapped from C++ pointers. Carbon pointers do not support indexing, and
the result of mapping a C++ pointer into Carbon would be a pointer that does not
support indexing. In the other direction, no mechanism prevents a Carbon pointer
from being passed into C++ code and then indexed, even though that is not an
operation that would be possible in pure, safe Carbon code.

Eventually we will need to provide some way to take a C++ pointer and perform
the equivalent of indexing into it in Carbon code. However, this mechanism will
require additional Carbon language features to be designed before it can be
specified. In particular, we will need a safety story for bounds safety, and a
type for representing an indexable location in some way, such as an array
iterator or array cursor type. As a result, we leave support for indexing and
pointer arithmetic as future work.

## Rationale

Goals:

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   C++ types are mapped to Carbon types with the same representation. This
        avoids the need for any deep copies on the interop boundary.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   The convenience names `Cpp.void` and `Cpp.nullptr` make it easier to
        write code that interoperates with C++ void pointers and C++ nullable
        pointers.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   C++ null pointers cannot "leak" into Carbon non-nullable pointer types,
        except by violating a "nonnull" annotation in the C++ code. If
        `_Nonnull` is violated, the result is not undefined behavior unless the
        pointer is used in a context that would result in undefined behavior in
        C++, such as a load or store through the pointer.
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
    -   We do not assume that a null pointer has an all-zero-bits
        representation. This is not true in practice on some modern
        heterogeneous compute architectures.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This proposal significantly enhances interoperability with C++ code, by
        providing a mapping for very common vocabulary types in C++.

Principles:

-   [All APIs are library APIs](/docs/project/principles/library_apis_only.md)
    -   The types added to support C++ are all ordinary class types provided in
        the Carbon prelude.
-   [Low context-sensitivity](/docs/project/principles/low_context_sensitivity.md)
    -   The Carbon `Cpp.void` type has only one meaning, rather than having a
        context-sensitive meaning. The C++ `void` type still has two different
        meanings, and therefore two different mappings into Carbon, but that
        problem is outside our domain.
-   [Namespace cleanliness](/docs/project/principles/namespace_cleanliness.md)
    -   The names we are adding to the `Cpp` package, `void` and `nullptr`, are
        C++ keywords, and so do not conflict with any C++ identifier. While
        Clang does provide an extension to define entities with these names, for
        example `int __identifier(void);`, interoperting with such code is not a
        priority.

## Future work

-   Add a design for `Core.Optional`, as we now have multiple language proposals
    that depend upon it.
-   Design interop for function pointers.
-   Design interop for pointers to members.
-   Design an approach for providing the functionality that C++ exposes as
    pointer indexing and pointer arithmetic.

## Alternatives considered

### Map all pointers to `T*`

We could avoid wrapping nullable pointers with `Core.Optional`. However, doing
so opens a large hole in Carbon's story for pointers, wherein pointers are not
nullable.

### Map `void*` to `()*`

We could map `void` to `()` in all contexts, and map C++'s `void*` to Carbon's
`()*` -- or rather, to `Core.Optional(()*)`. However, in order to support
passing arbitrary Carbon pointers to C++ `void*` parameters, we would need to
allow `T*` to implicitly convert to `void*` in Carbon, which means we would need
to allow `T*` to implicitly convert to `()*`. Therefore the C++ modeling of
`void` as a supertype of all other types leaks out into pure Carbon code. This
seems undesirable.

### Map `void*` to `u8*`

We could map `void*` to `u8*` or to a pointer to some other byte-like type, to
reflect that it represents a pointer to storage. This would result in an N:1
mapping from C++ types to Carbon types, because both `void*` and `uint8_t*`
would map to the same Carbon type. The same would happen if we picked any other
Carbon type that has a corresponding C++ type that is not `void`.

It's strongly desirable that our mapping between C++ and Carbon types fully
round trips, because otherwise passing types between the two languages, such as
in metaprogramming or by way of template argument deduction, would be lossy. For
example, if both a `vector<void*>` and a `vector<uint8_t*>` map to the same
Carbon type `buf(u8*)`, then passing an object of that type from C++ into Carbon
and then back into C++ must result in a type that mismatches at least one of the
original types.

It's possible that we could accept some N:1 mappings, but given how common
`void*` is on C and C++ API boundaries, the risk of problems seems particularly
significant in this case.

### Map `void f()` to `fn Cpp.f() -> Cpp.void`

We could use the custom `Cpp.void` type even as a function return type, removing
the non-uniformity of mapping it to `()` in function returns and to `Cpp.void`
elsewhere. However, `Cpp.void` is an abstract type, so there should not exist
initializing expressions of this type.

We could address that by instead mapping `void` to `partial Cpp.void`, or
aliasing `Cpp.void` to `partial Core.CppCompat.VoidBase`, but either way that
means that `void*` maps to a type whose pointee doesn't have the abstract /
incomplete behavior that we desire.

### Use the name `Core.CppCompat.Void` instead of `Core.CppCompat.VoidBase`

We could use a simpler name for the compatibillity type. However, given that
there are two different meanings of `void` in C++, having some extra clarity
about which meaning is intended seems useful.

### Map `Cpp.void` to `()` instead of `Core.CppCompat.VoidBase`

We could pick the other meaning of `void` as the meaning of `Cpp.void`. However,
the `()` meaning is only really interesting as a function return type, and there
is no reason to reach for `Cpp.void` if that meaning is desired. So mapping
`Cpp.void` to `VoidBase` is more useful.

Also, we want the mappings between C++ and Carbon types to be bidirectional, to
the extent that's possible. Mapping Carbon's `()` to C++'s `void` type would
mean that we can't consistently map all Carbon tuple types to the same family of
C++ tuple types, such as `std::tuple`.

### Do not provide `Cpp.void` at all

We could ask developers to write out the name of `VoidBase` when needed. But
it's long and cumbersome, and we expect most other C++ types with a
corresponding keyword to be provided in the `Cpp` package, so providing it is
both useful and improves language consistency.
