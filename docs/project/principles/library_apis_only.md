# Principle: All APIs are library APIs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of this principle](#applications-of-this-principle)
-   [Exceptions](#exceptions)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Background

Every major modern programming language comes with a standard library, which
consists of APIs that are not part of the core language, but instead are written
in the language (although their implementations may not be). However, different
languages draw the boundary between language and library in different places.
For example, Go's `map` type is built into the core language, whereas the C++
equivalent, `std::unordered_map`, is part of the standard library. In Swift,
even fundamental types like integers and pointers are part of the standard
library; there are no truly "built in" types.

These decisions can have important consequences for the design of the language.
For example, many important features of C++, such as move semantics, variadics,
and coroutines, were motivated largely by their anticipated uses in a small set
of standard library types. In a language with a different design philosophy,
those types could have been built into the core language. This would probably
have substantially simplified the language, and made those types available
faster. However, that would have come at the cost of less flexibility for users
outside the common case.

## Principle

In Carbon, every public function is declared in some Carbon `api` file, and
every public `interface`, `impl`, and first-class type is defined in some Carbon
`api` file. In some cases, the bodies of public functions will not be defined as
Carbon code, or will be defined as hybrid Carbon code using intrinsics that
aren't available to ordinary Carbon code. However, we will try to minimize those
situations.

Thus, even "built-in" APIs can be used like user-defined APIs, by importing the
appropriate library and using qualified names from that library, relying on the
ordinary semantic rules for Carbon APIs.

## Applications of this principle

We expect Carbon to have a special "prelude" library that is implicitly imported
by all Carbon source files, and there might be a special name lookup rule to
allow the names in the prelude to be used unqualified. However, in accordance
with this principle, they will remain available to ordinary qualified name
lookup as well.

According to the resolutions of
[#543](https://github.com/carbon-language/carbon-lang/issues/543) and
[#750](https://github.com/carbon-language/carbon-lang/issues/750), Carbon will
have a substantial number of type keywords, such as `i32`, `f64`, and `bool`.
However, these keywords will all be aliases for ordinary type names, such as
`Carbon.Int(32)`, `Carbon.Float(64)`, and `Carbon.Bool`. Furthermore, all
arithmetic and logical operators will be overloadable, so that those types can
be defined as class types. The member function bodies for these types will be
probably not be implemented in Carbon, but this principle applies only to
function declarations, not function definitions.

Similarly, a pointer type such as `Foo*` will be an alias for some library class
type, for example `Carbon.Ptr(Foo)`. As a result, Carbon will support
overloading pointer operations like `->` and unary `*`.

All Carbon operations that use function-style syntax, such as `sizeof()` and
`decltype()` in C++, will be standard library functions. As above, in some cases
we may choose to alias those functions with keywords, and the function bodies
may not be defined in Carbon.

## Exceptions

This principle applies to types only if they are _first-class_, meaning that
they can be the types of run-time variables, function parameters, and return
values. Carbon's type system will probably also include some types whose usage
is more restricted, and this principle will not apply to them. Most importantly,
function types might not be first-class types, in which case they need not be
library types.

The logic for translating a literal expression to a value of the appropriate
type is arguably part of that type's public API, but will not be part of that
type's class definition.

Tuple types will probably not fully conform to this principle, because doing so
would be circular: there is no way to name a tuple type that doesn't rely on
tuple syntax, and no way to define a class body for a tuple type that doesn't
contain tuple patterns. However, we will strive to ensure that it is possible to
define a parameterized class type within Carbon that supports all the same
operations as built-in tuple types.

## Alternatives considered

-   [Built-in primitive types](/proposals/p1280.md#built-in-primitive-types)
