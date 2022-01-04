# Principle: One static open extension mechanism

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)

<!-- tocstop -->

## Background

In C++, a single function may be overloaded with definitions in multiple files.
The [ADL](https://en.wikipedia.org/wiki/Argument-dependent_name_lookup) name
lookup rule even allows an unqualified call to resolve to functions defined in
different namespaces. These rules are used to define extension points with
static dispatch for operator overloading and functions like
[`swap`](https://www.cplusplus.com/reference/algorithm/swap/).

Nothing in C++ restricts the signatures of function overloads. This means that
if overloading is used as an extension point to define an operation for a
variety of types, there is no way to type check generic code that tries to
invoke that operation over those types.

## Principle

Carbon uses [interfaces](/docs/design/generics/overview.md) for static open
extension. Each type may define its own implementation of each interface.
Generic code can be written that works with any type implementing the interface.
That code can be type checked independent of which type the generic code is
instantiated with by using the fact that the interface specifies the signatures
of the calls.

To keep the language simple, this is the only static open extension mechanism in
Carbon. This means that function overloading is limited in Carbon to only
signatures defined together in the same library. It also means that to
interoperate with C++, the operators and `swap` need to have corresponding
interfaces on the Carbon side.

The main advantage of interfaces as an open extension mechanism over open
overloading is allowing generics to be type checked separately. In addition,
they are less [context sensitive](low_context_sensitivity.md). Generics are
[coherent](/docs/design/generics/terminology.md#coherence), while open function
overloading can resolve names differently depending on what is imported. Closed
overloading in Carbon also simplifies what gets exported to C++ from Carbon.

Interfaces provide an way to group functions together, and express the
constraint that all of the functions in the group are implemented. Consider a
random-access iterator, which has a number of methods. If a C++ template
function only accesses some of those methods which happens to match the subset
defined for a type, the code will work temporarily but fail later when the code
is changed to use a different subset.

Another approach to operator overloading is to use methods with a specific name.
In C++ these start with the
[`operator` keyword](https://en.cppreference.com/w/cpp/language/operators).
[Python uses method with names starting and ending with double underscores](https://docs.python.org/3/reference/datamodel.html#special-method-names).
Interfaces are more flexible about where implementations may be defined. For
example, with special method names, `+` on a `Vector(T)` class could only be
defined as part of the `Vector(T)` definition. With interfaces, additionally `+`
for `Vector(MyType)` could be implemented with `MyType`.

This helps achieve the Carbon Goal of
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
