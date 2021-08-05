# Basic classes: use cases, struct literals, struct types, and future work

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/561)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Earlier proposal](#earlier-proposal)
    -   [Interfaces implemented for anonymous data classes](#interfaces-implemented-for-anonymous-data-classes)
    -   [Access control](#access-control)
    -   [Introducer for structural data class types](#introducer-for-structural-data-class-types)
    -   [Terminology](#terminology)

<!-- tocstop -->

## Problem

We need to say how you define new types in Carbon. This proposal is specifically
about [record types](<https://en.wikipedia.org/wiki/Record_(computer_science)>).
The proposal is not intended to be a complete story for record types, but enough
to get agreement on direction. It primarily focuses on:

-   use cases including: data classes, encapsulated types with virtual and
    non-virtual methods and optional single inheritance, interfaces as base
    classes that support multiple inheritance, and mixins for code reuse;
-   anonymous structural data types for record literals used to initialize class
    values and ad-hoc parameter and return types with named components; and
-   future work, including the provisional syntax in use for features that have
    not been decided.

## Background

This is a replacement for earlier proposal
[#98](https://github.com/carbon-language/carbon-lang/pull/98).

## Proposal

This proposal adds an initial design for record types called "classes",
including _structural data classes_ called _struct types_ as well as struct
literals. The design is replacing the skeletal design for what were called
"struct" types with a [new document on classes](/docs/design/classes.md).

## Rationale based on Carbon's goals

This particular proposal is focusing on
[the Carbon goal](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
that code is "easy to read, understand, and write." Future proposals will
address other aspects of the class type design such as performance.

## Alternatives considered

### Earlier proposal

There was an earlier proposal
[#98](https://github.com/carbon-language/carbon-lang/pull/98), that made a
number of different choices, including:

-   Tuples were given named components instead of having separate struct
    literals.
-   No form of multiple inheritance was proposed.
-   Operators were define using methods like C++ instead of by implementing
    interfaces like Rust.
-   Constructors had a special form like C++ instead of being regular functions
    like Rust.
-   Tuples and classes were both considered example of record types.
-   Members of classes could be individually left uninitialized.
-   Coverage of nominal types, inheritance, etc. were considered in much more
    detail.

### Interfaces implemented for anonymous data classes

Whether we would support implementing interfaces for specific anonymous data
classes was
[discussed on Discord](https://discord.com/channels/655572317891461132/709488742942900284/867471671089561643).
[The conclusion](https://discord.com/channels/655572317891461132/709488742942900284/867516894029938710)
was "yes", reasoning that we would support that for the same reason as a number
of other cases such as tuple and pointer types.
[A specific use case](https://discord.com/channels/655572317891461132/709488742942900284/867517209026756630)
would be implementing interface

```
interface ConstructWidgetFrom { fn Construct(Self) -> Widget; }
```

for type `{.kind: WidgetKind, .size: Int}`.

### Access control

[Issue #665](https://github.com/carbon-language/carbon-lang/issues/665) decided
that by default members of a class would be publicly accessible. There were a
few reasons:

-   The readability of public members is the most important, since we expect
    most readers to be concerned with the public API of a type.
-   The members that are most commonly private are the data fields, which have
    relatively less complicated definitions that suffer less from the extra
    annotation.

Additionally, there is precedent for this approach in modern object-oriented
languages such as
[Kotlin](https://kotlinlang.org/docs/visibility-modifiers.html) and
[Python](https://docs.python.org/3/tutorial/classes.html), both of which are
well regarded for their usability.

It further decided that members would be given more restricted access using a
local annotation on the declaration itself rather than a block or region
approach such as used in C++. This is primarily motivated by a desire to reduce
context sensitivity, following
[the principle](/docs/project/principles/low_context_sensitivity.md) introduced
in [#646](https://github.com/carbon-language/carbon-lang/pull/646). It helps
readers to more easily determine the accessibility of a member in large classes,
say when they have jumped to a specific definition in their IDE.

### Introducer for structural data class types

[Issue #653](https://github.com/carbon-language/carbon-lang/issues/653)
discussed whether structural data class types should have an introducer to
distinguish them from structural data class literals. Ultimately we decided no
introducer was needed:

-   Outside of `{}`, types could be distinguished from literal values by the
    presence of a `:` after the first field name.
-   This creates a sort of consistency: introducers are frequently used when
    introducing new names, as in `fn`, `var`, `interface`, and so on. Struct
    type declarations don't introduce new names so they don't require an
    introducer.
-   It avoids having a different introducer for things that are still treated as
    classes for many purposes. This means we won't have to frequently say
    "struct or class" in documentation.
-   We do want to use these type expressions in contexts that will benefit from
    being more concise, such as inside function and variable declarations.

This does cause an issue that `{}` is both an empty struct literal and empty
struct type. However, we've already accepted that complexity with tuples, so
this choice is more consistent. If we find that we need an introducer for tuple
types to distinguish the empty tuple from its type, we expect to find that we
have the same problem with empty struct literals, and the other way around. We
are explicitly to choosing to accept the risk that this won't work out in order
to have a more concise syntax in case it does.

### Terminology

Do literals have "class" type or are they some other kind of type?
[Issue #651](https://github.com/carbon-language/carbon-lang/issues/651) decided
that all of these types were different kinds of classes:

-   Literals like `{.a = 2}` are "structural data class literals" or "struct
    literals" for short. Here "structural" means that two types are considered
    equal if their fields match. They are not "nominal" since they don't have a
    name to use for type equality.
-   The types of those literals like `{.a: i64}` would be "structural data
    classes" or "struct types" for short.
-   There would also be "nominal data classes" that are declared with a syntax
    more similar to other nominal classes.

We preferred to refer to all of these as class types, rather than have to
frequently refer to "struct or class types", adding additional words to name
more specific subsets, like "data classes". In contrast, tuple types are not
considered classes, but classes and tuples together form _product types_.

The term "class" was chosen over "struct" since they generally support
object-oriented features like encapsulation, inheritance, and dynamic dispatch,
and how C++ programmers generally refer to their record types. These were
considered more significant than the C++ distinction that classes default to
private access control. Since we plan to use a different syntax in Carbon to
specify access restrictions, the different default seemed straightforward to
teach.
