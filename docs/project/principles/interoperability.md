# Principle: C++ Interoperability

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principle

Our goal is not merely for Carbon to be a good language in general, but for it
to be specifically a good choice as the target of large-scale tool-assisted
migration from C++. A large migration can never be atomic, so high-fidelity and
high-performance interoperability between C++ and Carbon will have to be part of
the migration strategy. It must be possible to rewrite a C++ library as Carbon
without having to rewrite its dependencies or its clients, and a mixed
C++/Carbon program must have adequate performance.

Note, however, that the interoperability doesn't necessarily have to be
convenient. Once we have passed the threshold that incremental migration is
possible, we have a choice of how much we reduce migration costs and how
convenient we make the interoperability. To give examples from other languages:
we want C++/Carbon interoperability to have fewer rough edges than C++/Java or
C++/Rust, but it doesn't necessarily have to be as seamless as Java/Kotlin or
JavaScript/TypeScript.

Interoperability may sometimes be in tension with other goals, like performance,
safety, and ergonomics. We're willing to sacrifice some level of
interoperability convenience, even if C++ and Carbon code at the language
boundary might sometimes have to be awkward and non-idiomatic, if it's important
for achieving those goals.

## Application of these principles

These are non-exhaustive examples of design decisions where we are and are not
willing to compromise on C++/Carbon interoperability for the sake of other goals
like performance, safety, and Carbon ergonomics.

- Concrete C++ types should be usable from Carbon, and vice versa, at least on
  an opt-in basis. Possibly not all C++ types need be usable from Carbon, only
  ones tagged in some special way, but in any case crossing the language
  boundary should not require copying data to adhere to a different memory
  layout. At an implementation level, this might mean that a Carbon ABI has
  multiple memory layout strategies, one of which is C++'s, just as a C++
  compiler today supports multiple calling conventions for the sake of
  interoperability with C and other languages.

  - C++ classes with private member variables and virtual functions should work
    in Carbon.
  - C++ protocol buffers should work in Carbon.
  - C++ vocabulary types like `std::string`, `std::vector`, `std::tuple`, and
    `absl::flat_hash_map` should work in Carbon.

- Many C++ libraries have an API where a client is expected to supply an object
  that inherits from a specific abstract base class, so it should be possible
  for a Carbon struct to implement a C++ interface.

  - This doesn't necessarily mean that Carbon must have inheritance of
    implementation as a first-class feature. We might decide that this applies
    only to pure C++ interface classes, and that C++ "interfaces" with
    implementation must be rewritten for the Carbon migration.
  - Whether Carbon needs to have "classes" in the C++ sense remains to be seen:
    we need to make it possible for concrete Carbon types to be usable from C++,
    and it remains to be seen whether that can be done if Carbon doesn't have
    classes.

- Carbon doesn't need to have exactly the same set of primitive integer types as
  C++, so long as it's possible to know which Carbon type(s) a C++ integer type
  maps to, given a target platform.

- Carbon doesn't have to support C++ exceptions, still less throwing exceptions
  across the C++/Carbon boundary. Carbon could, for example, take the same
  approach that
  [Swift does](https://github.com/apple/swift/blob/master/docs/CppInteroperabilityManifesto.md#baseline-functionality-import-functions-as-non-throwing-terminate-on-uncaught-c-exceptions):
  if C++ code throws an exception that propagates into a Carbon stack frame, we
  terminate the program.

- Carbon's design for object lifetime and aliasing will be made for the sake of
  performance and safety, even if that means conflicting with C++ rules. For
  example, Carbon need not adopt C++ rules about lifetimes of temporaries and
  could adopt stricter rules, even though that would require more care in using
  some C++ APIs from Carbon.

  - As an example: a C++ program might pass the return value of `absl::StrCat`
    to a function whose parameter is `std::string_view`, which relies on C++
    extending the lifetime of a temporary to a full expression. If Carbon has
    different lifetime rules, then it will be harder to use such a C++ function
    safely from Carbon.

- It should be possible to use Carbon's parameterized types with concrete C++
  types. For example, there should be some way of putting C++ objects in Carbon
  containers like vectors or hash maps.

  - Since Carbon container types are likely to use generics, this implies that
    there should be some way of instantiating Carbon generics with C++ types.
    This doesn't necessarily mean that using a C++ type with a Carbon generic
    has to be as convenient as using a Carbon type with a Carbon generic: we
    might imagine requiring some kind of wrapper or declaration, so long as it
    doesn't impose too much boilerplate or performance overhead.

- Since C++ libraries and vocabulary types are frequently written in terms of
  templates, it should be possible to instantiate C++ templates with Carbon
  types.

  - We expect that complicated C++ metaprogramming libraries will need to be
    rewritten by hand, since the goal is to support the 95% case, but we
    shouldn't require a full rewrite for clients of template code.

  - Whether Carbon needs to support C++ libraries that are based on CRTP is an
    open question. It's a common technique, but supporting it would require
    tight intertwining between C++ and Carbon template instantiation.

- C++ type traits should work correctly on Carbon types.

- Carbon code should be able to use C++ APIs that are defined in terms of
  incomplete types.
  - However, Carbon code will not necessarily be able to define incomplete types
    with exactly the same look and feel as in C++. Carbon might satisfy these
    use cases with a different language feature.
