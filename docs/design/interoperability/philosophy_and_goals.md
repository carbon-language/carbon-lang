# Interoperability philosophy and goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
    -   [Other interoperability layers](#other-interoperability-layers)
-   [Philosophy](#philosophy)
-   [Language goal influences](#language-goal-influences)
    -   [Performance-critical software](#performance-critical-software)
    -   [Software and language evolution](#software-and-language-evolution)
    -   [Code that is easy to read, understand, and write](#code-that-is-easy-to-read-understand-and-write)
    -   [Practical safety guarantees and testing mechanisms](#practical-safety-guarantees-and-testing-mechanisms)
    -   [Fast and scalable development](#fast-and-scalable-development)
    -   [Modern OS platforms, hardware architectures, and environments](#modern-os-platforms-hardware-architectures-and-environments)
    -   [Interoperability with and migration from existing C++ code](#interoperability-with-and-migration-from-existing-c-code)
-   [Goals](#goals)
    -   [Support mixing Carbon and C++ toolchains](#support-mixing-carbon-and-c-toolchains)
    -   [Compatibility with the C++ memory model](#compatibility-with-the-c-memory-model)
    -   [Minimize bridge code](#minimize-bridge-code)
    -   [Unsurprising mappings between C++ and Carbon types](#unsurprising-mappings-between-c-and-carbon-types)
    -   [Allow C++ bridge code in Carbon files](#allow-c-bridge-code-in-carbon-files)
    -   [Carbon inheritance from C++ types](#carbon-inheritance-from-c-types)
    -   [Support use of advanced C++ features](#support-use-of-advanced-c-features)
    -   [Support basic C interoperability](#support-basic-c-interoperability)
-   [Non-goals](#non-goals)
    -   [Full parity between a Carbon-only toolchain and mixing C++/Carbon toolchains](#full-parity-between-a-carbon-only-toolchain-and-mixing-ccarbon-toolchains)
    -   [Never require bridge code](#never-require-bridge-code)
    -   [Convert all C++ types to Carbon types](#convert-all-c-types-to-carbon-types)
    -   [Support for C++ exceptions without bridge code](#support-for-c-exceptions-without-bridge-code)
    -   [Cross-language metaprogramming](#cross-language-metaprogramming)
    -   [Offer equivalent support for languages other than C++](#offer-equivalent-support-for-languages-other-than-c)
-   [Open questions to be resolved later](#open-questions-to-be-resolved-later)
    -   [Carbon type inheritance from non-pure interface C++ types](#carbon-type-inheritance-from-non-pure-interface-c-types)
    -   [CRTP support](#crtp-support)
    -   [Object lifetimes](#object-lifetimes)
-   [References](#references)

<!-- tocstop -->

## Background

Interoperability with and migration from C++ are a
[language goal](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).
However, performance and evolution are
[_higher_ priorities](/docs/project/goals.md#language-goals-and-priorities).
This interaction of priorities is important to understanding Carbon's
interoperability goals and trade-offs.

### Other interoperability layers

Other language interoperability layers that may offer useful examples are:

-   [Java/Kotlin](https://kotlinlang.org/docs/reference/java-to-kotlin-interop.html)
    should be a comparable interoperability story. The languages are different,
    but share an underlying runtime. This may be closest to the model we desire
    for Carbon.

-   [JavaScript/TypeScript](https://www.typescriptlang.org/docs/handbook/migrating-from-javascript.html)
    is similar to C/C++, where one language is essentially a subset of the
    other, allowing high interoperability. This is an interesting reference
    point, but we are looking at a different approach with a clearer boundary.

-   [C++/Java](https://en.wikipedia.org/wiki/Java_Native_Interface) is an
    example of requiring specialized code for the bridge layer, making
    interoperability more burden on developers. The burden of the approach may
    be considered to correspond to the difference in language memory models and
    other language design choices. Regardless, the result can be considered
    higher maintenance for developers than we want for Carbon.

-   [C++/Go](https://golang.org/cmd/cgo/) is similar to C++/Java. However, Go
    notably allows C++ bridge code to exist in the .go files, which can ease
    maintenance of the bridge layer, and is desirable for Carbon.

## Philosophy

The C++ interoperability layer of Carbon allows a subset of C++ APIs to be
accessed from Carbon code, and similarly a subset of Carbon APIs to be accessed
from C++ code. This requires expressing one language as a subset of the other.
Bridge code may be needed to map some APIs into the relevant subset, but the
constraints on expressivity should be loose enough to keep the amount of such
bridge code sustainable.

The design for interoperability between Carbon and C++ hinges on:

1.  The ability to interoperate with a wide variety of code, such as
    classes/structs and templates, not just free functions.
2.  A willingness to expose the idioms of C++ into Carbon code, and the other
    way around, when necessary to maximize performance of the interoperability
    layer.
3.  The use of wrappers and generic programming, including templates, to
    minimize or eliminate runtime overhead.

These things come together when looking at how custom data structures in C++ are
exposed into Carbon, and the other way around. In both languages, it is
reasonable and even common to have customized low-level data structures, such as
associative containers. For example, there are numerous data structures for
mapping from a key to a value that might be best for a particular use case,
including hash tables, linked hash tables, sorted vectors, and btrees. Even for
a given data structure, there may be slow but meaningful evolution in
implementations strategies.

The result is that it will often be reasonable to directly expose a C++ data
structure to Carbon without converting it to a "native" or "idiomatic" Carbon
data structure. Although interfaces may differ, a trivial adapter wrapper should
be sufficient. Many Carbon data structures should also be able to support
multiple implementations with C++ data structures being one such implementation,
allowing for idiomatic use of C++ hidden behind Carbon.

The reverse is also true. C++ code will often not care, or can be refactored to
not care, what specific data structure is used. Carbon data structures can be
exposed as yet another implementation in C++, and wrapped to match C++ idioms
and even templates.

For example, a C++ class template like `std::vector<T>` should be usable without
wrapper code or runtime overhead, and passing a Carbon type as `T`. The
resulting type should be equally usable from either C++ or Carbon code. It
should also be easy to wrap `std::vector<T>` with a Carbon interface for
transparent use in idiomatic Carbon code.

## Language goal influences

### Performance-critical software

Interoperability with C++ will be frequently used in Carbon, whether it's C++
developers trying out Carbon, incrementally migrating a large C++ codebase, or
continuing to use a C++ library long-term. In all cases, it must be possible to
write interoperable code with zero overhead; copies must not be required.

### Software and language evolution

Interoperability will require the addition of features to Carbon which exist
primarily to support interoperability use cases. However, these features must
not unduly impinge the overall evolution of Carbon. In particular, only a subset
of Carbon features will support interoperability with C++. To do otherwise would
restrict Carbon's feature set.

### Code that is easy to read, understand, and write

Interoperability-related Carbon code will likely be more difficult to read than
other, more idiomatic Carbon code. This is okay: aiming to make Carbon code
readable doesn't mean that it needs to _all_ be trivial to read. At the same
time, the extra costs that interoperability exerts on Carbon developers should
be minimized.

### Practical safety guarantees and testing mechanisms

Safety is important to maintain around interoperability code, and mitigations
should be provided where possible. However, safety guarantees will be focused on
native Carbon code. C++ code will not benefit from the same set of safety
mechanisms that Carbon offers, so Carbon code calling into C++ will accept
higher safety risks.

### Fast and scalable development

The interoperability layer will likely have tooling limitations similar to C++.
For example, Carbon aims to compile quickly. However, C++ interoperability
hinges on compiling C++ code, which is relatively slow. Carbon libraries that
use interoperability will see bottlenecks from C++ compile time. Improving C++
is outside the scope of Carbon.

### Modern OS platforms, hardware architectures, and environments

Interoperability will apply to the intersection of environments supported by
both Carbon and C++. Pragmatically, Carbon will likely be the limiting factor
here.

### Interoperability with and migration from existing C++ code

Carbon's language goal for interoperability will focus on C++17 compatibility.
The language design must be mindful of the prioritization; trade-offs harming
other goals may still be made so long as they offer greater benefits for
interoperability and Carbon as a whole.

Although the below interoperability-specific goals will focus on
interoperability, it's also important to consider how migration would be
affected. If interoperability requires complex work, particularly to avoid
performance impacts, it could impair the ability to incrementally migrate C++
codebases to Carbon.

## Goals

### Support mixing Carbon and C++ toolchains

The Carbon toolchain will support compiling C++ code. It will contain a
customized C++ compiler that enables some more advanced interoperability
features, such as calling Carbon templates from C++.

Mixing toolchains will also be supported in both directions:

-   C++ libraries compiled by a non-Carbon toolchain will be usable from Carbon,
    so long as they are ABI-compatible with Carbon's C++ toolchain.

-   The Carbon toolchain will support, as an option, generating a C++ header and
    object file from a Carbon library, with an ABI that's suitable for use with
    non-Carbon toolchains.

Mixing toolchains restricts functionality to what's feasible with the C++ ABI.
For example, developers should expect that Carbon templates will be callable
from C++ when using the Carbon toolchain, and will not be available when mixing
toolchains because it would require a substantially different and more complex
interoperability implementation. This degraded interoperability should still be
sufficient for most developers, albeit with the potential of more bridge code.

Any C++ interoperability code that works when mixing toolchains must work when
using the native Carbon toolchain. The mixed toolchain support must not have
semantic divergence. The converse is not true, and the native Carbon toolchain
may have additional language support and optimizations.

### Compatibility with the C++ memory model

It must be straightforward for any Carbon interoperability code to be compatible
with the C++ memory model. This does not mean that Carbon must exclusively use
the C++ memory model, only that it must be supported.

### Minimize bridge code

The majority of simple C++ functions and types should be usable from Carbon
without any custom bridge code and without any runtime overhead. That is, Carbon
code should be able to call most C++ code without any code changes to add
support for interoperability, even if that code was built with a non-Carbon
toolchain. This includes instantiating Carbon templates or generics using C++
types.

In the other direction, Carbon may need some minimal markup to expose functions
and types to C++. This should help avoid requiring Carbon to generate
C++-compatible endpoints unconditionally, which could have compile and linking
overheads that may in many cases be unnecessary. Also, it should help produce
errors that indicate when a function or type may require additional changes to
make compatible with C++.

Carbon's priority developers should be able to easily reuse the mature ecosystem
of C++ libraries provided by third-parties. A third-party library's language
choice should not be a barrier to Carbon adoption.

Even for first-party libraries, migration of C++ codebases to Carbon will often
be incremental due to human costs of executing and verifying source migrations.
Minimizing the amount of bridge code required should be expected to simplify
such migrations.

### Unsurprising mappings between C++ and Carbon types

Carbon will provide unsurprising mappings for common types.

**Primitive types** will have mappings with zero overhead conversions. They are
frequently used, making it important that interoperability code be able to use
them seamlessly.

The storage and representation will need to be equivalent in both languages. For
example, if a C++ `__int64` maps to Carbon's `Int64`, the memory layout of both
types must be identical.

Semantics need to be similar, but edge-case behaviors don't need to be
identical, allowing Carbon flexibility to evolve. For example, where C++ would
have modulo wrapping on integers, Carbon could instead have trapping behavior on
the default-mapped primitive types.

Carbon may have versions of these types with no C++ mapping, such as `Int256`.

**Non-owning vocabulary types**, such as pointers and references, will have
transparent, automatic translation between C++ and Carbon non-owning vocabulary
types with zero overhead.

**Other vocabulary types** will typically have reasonable, but potentially
non-zero overhead, conversions available to map into Carbon vocabulary types.
Code using these may choose whether to pay the overhead to convert. They may
also use the C++ type directly from Carbon code, and the other way around.

**Incomplete types** must have a mapping with similar semantics, similar to
primitive types.

### Allow C++ bridge code in Carbon files

Carbon files should support inline bridge code written in C++. Where bridge code
is necessary, this will allow for maintenance of it directly alongside the code
that uses it.

### Carbon inheritance from C++ types

Carbon will support inheritance from C++ types for interoperability, although
the syntax constructs may look different from C++ inheritance. This is
considered necessary to address cases where a C++ library API expects users to
inherit from a given C++ type.

This might be restricted to pure interface types; see
[the open question](#carbon-type-inheritance-from-non-pure-interface-c-types).

### Support use of advanced C++ features

There should be support for most idiomatic usage of advanced C++ features. A few
examples are templates, overload sets,
[attributes](https://en.cppreference.com/w/cpp/language/attributes) and
[ADL](https://en.wikipedia.org/wiki/Argument-dependent_name_lookup).

Although these features can be considered "advanced", their use is widespread
throughout C++ code, including STL. Support for such features is key to
supporting migration from C++ features.

### Support basic C interoperability

C interoperability support must be sufficient for Carbon code to call popular
APIs that are written in C. The ability of C to call Carbon will be more
restricted, limited to where it echoes C++ interoperability support. Basic C
interoperability will include functions, primitive types, and structs that only
contain member variables.

Features where interoperability will rely on more advanced C++-specific
features, such as templates, inheritance, and class functions, need not be
supported for C. These would require a C-specific interoperability model that
will not be included.

## Non-goals

### Full parity between a Carbon-only toolchain and mixing C++/Carbon toolchains

Making mixed C++/Carbon toolchain support equivalent to Carbon-only toolchain
support affects all interoperability features. Mixed toolchains will have
degraded support because full parity would be too expensive.

The feature of calling Carbon templates from C++ code is key when analyzing this
option. Template instantiation during compilation is pervasive in C++.

With a Carbon toolchain compiling both Carbon and C++ code, the C++ compiler
_can_ be modified to handle Carbon templates differently. Carbon templates can
be handled by exposing the Carbon compiler's AST to the C++ compiler directly,
as a compiler extension. While this approach is still complex and may not always
work, it should offer substantial value and ability to migrate C++ code to
Carbon without requiring parallel maintenance of implementations in C++.

With a mixed toolchain, the C++ compiler _cannot_ be modified to handle Carbon
templates differently. The only way to support template instantiation would be
by having Carbon templates converted into equivalent C++ templates in C++
headers; in other words, template support would require source-to-source
translation. Supporting Carbon to C++ code translations would be a complex and
high cost feature to achieve full parity for mixed toolchains. Requiring bridge
code for mixed toolchains is the likely solution to avoid this cost.

Note that this issue differs when considering interoperability for Carbon code
instantiating C++ templates. The C++ templates must be in C++ headers for
re-use, which in turn must compile with the Carbon toolchain to re-use the built
C++ code, regardless of whether a separate C++ toolchain is in use. This may
also be considered a constraint on mixed toolchain interoperability, but it's
simpler to address and less likely to burden developers.

To summarize, developers should expect that while _most_ features will work
equivalently for mixed toolchains, there will never be full parity.

### Never require bridge code

Corner cases of C++ will not receive equal support to common cases: the
complexity of supporting any given construct must be balanced by the real world
need for that support. For example:

-   Long-term, we expect interoperability will target all of C++, including new
    features as they are added, standardized, implemented, and adopted across
    the industry. The priority of _individual_ features will reflect how widely
    they are used in practice and how any gap impacts users trying to adopt
    Carbon. Exhaustive, high-quality support of the long-tail or corner cases of
    C++ features should not be assumed.

-   Support will be focused on idiomatic code, interfaces, and patterns used in
    widespread open source libraries or by other key constituencies. C++ code
    will have edge cases where the benefits of limiting Carbon's maintenance
    costs by avoiding complex interoperability outweighs the value of avoiding
    bridge code.

-   Support for low-level C ABIs may be focused on modern 64-bit ABIs, including
    Linux, POSIX, and a small subset of Windows' calling conventions.

### Convert all C++ types to Carbon types

Non-zero overhead conversions should only be _supported_, never _required_, in
order to offer reliable, unsurprising performance behaviors. This does not mean
that conversions will _always_ be supported, as support is a cost-benefit
decision for specific type mappings. For example, consider conversions between
`std::vector<T>` and an equivalent, idiomatic Carbon type:

-   Making conversions zero-overhead would require the Carbon type to mirror the
    memory layout and implementation semantics of `std::vector<T>`. However,
    doing so would constrain the evolution of the Carbon type to match C++.
    Although some constraints are accepted for most primitive types, it would
    pose a major burden on Carbon's evolution to constrain Carbon's types to
    match C++ vocabulary type implementations.

-   These conversions may not always be present, but `std::vector<T>` is a
    frequently used type. As a result, it can be expected that there will be
    functions supporting a copy-based conversion to the idiomatic Carbon type.

-   An interface which can hide the difference between whether `std::vector<T>`
    or the equivalent, idiomatic Carbon type is in use may also be offered for
    common types.

-   It will still be normal to handle C++ types in Carbon code without
    conversions. Developers should be given the choice of when to convert.

### Support for C++ exceptions without bridge code

Carbon may not provide seamless interoperability support for C++ exceptions. For
example, translating C++ exceptions to or from Carbon errors might require
annotations or bridge code, and those translations may have some performance
overhead or lose information. Furthermore, if Carbon code calls a C++ function
without suitable annotations or bridging, and that function exits with an
exception, the program might terminate.

### Cross-language metaprogramming

Carbon's metaprogramming design will be more restrictive than C++'s preprocessor
macros. Although interoperability should handle simple cases, such as
`#define STDIN_FILENO 0`, complex metaprogramming libraries may require a deep
ability to understand code rewrites. It should be reasonable to have these
instead rewritten to use Carbon's metaprogramming model.

### Offer equivalent support for languages other than C++

Long-term, it should be anticipated that Carbon will add interoperability with
non-C++ languages. However, interoperability discussions will be focused on C++
in order to support the
[language goal](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).
Although we should work to consider extensibility when building interoperability
facilities, C++ should be expected to have more robust support.

Many languages do offer interoperability layers with C. Carbon's
[C interoperability](#support-basic-c-interoperability) will likely offer a
degree of multi-language interoperability using C as an intermediary.

## Open questions to be resolved later

### Carbon type inheritance from non-pure interface C++ types

Some C++ APIs will expect that consumers use classes that inherit from a type
provided by the API. It's desirable to have Carbon support, in some way,
inheritance from API types in order to use these APIs.

It may be sufficient to require the parent type be a pure interface, and that
APIs with either use bridge code or switch implementations. That will be
determined later.

### CRTP support

Although
[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) is a
common technique in C++, interoperability support may require substantial work.
Libraries based on use of CRTP may require bridge code or a rewrite for Carbon
interoperability.

More analysis should be done on the cost-benefit of supporting CRTP before
making a support decision.

### Object lifetimes

Carbon may have a different object lifetime design than C++. For example, Carbon
may choose different rules for determining the lifetime of temporaries. This
could affect idiomatic use of C++ APIs, turning code that would be safe in C++
into unsafe Carbon code, requiring developers to learn new coding patterns.

More analysis should be done on object lifetimes and potential Carbon designs
for it before deciding how to treat object lifetimes in the scope of
interoperability.

## References

-   Proposal
    [#175: C++ interoperability goals](https://github.com/carbon-language/carbon-lang/pull/175)
