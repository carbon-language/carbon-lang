# Property naming in C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/720)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
    -   [Style](#style)
    -   [Properties](#properties)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Disallow "setters"](#disallow-setters)
    -   [Disallow mutable accessors](#disallow-mutable-accessors)
    -   [Looser restrictions on property method performance](#looser-restrictions-on-property-method-performance)
    -   [Require trailing `_` only for property data members](#require-trailing-_-only-for-property-data-members)

<!-- tocstop -->

## Problem

This proposal aims to address two problems:

-   Our current C++ style rules are prone to name conflicts, particularly
    between types and accessors.
-   There is preliminary interest in Carbon providing "properties" as a language
    feature. It would be useful to be able to experiment with the style and API
    effects of such a feature in our C++ code.

## Background

### Style

Our C++ style rules currently require all functions (including accessors) and
all types to have `CamelCase` names. However, this creates a nontrivial risk of
name collisions. To take two examples:

-   The base class
    [`Carbon::Pattern`](https://github.com/carbon-language/carbon-lang/blob/ebd6c7afa91a1a02961b72d619fba630d8fbfbff/executable_semantics/ast/pattern.h#L25)
    defines a `Kind` enum, which identifies the concrete type of a `Pattern`
    object, as part of its support for
    [LLVM-style RTTI](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html). It
    also needs to provide an accessor that exposes the `Kind` of a `Pattern`
    object, so that client code can `switch` on it rather than using virtual
    dispatch. The most natural name for that accessor would be `Kind`, but C++
    doesn't allow us to use the same name for both a type and a function in the
    same scope. Instead, the accessor is called `Tag`, but that's an entirely
    ad-hoc solution: there happened to be a synonymous term available, so we
    arbitrarily chose one for the type and the other for the function.
-   The class
    [`Carbon::ExpressionPattern`](https://github.com/carbon-language/carbon-lang/blob/ebd6c7afa91a1a02961b72d619fba630d8fbfbff/executable_semantics/ast/pattern.h#L181)
    adapts an `Expression` to behave as a `Pattern`. The underlying `Expression`
    is exposed by an accessor, which is named `Expression`. That function name
    shadows the type name within the class, so the type must instead be referred
    to by its fully-qualified name `Carbon::Expression`.

As these examples illustrate, there are occasional situations where the best
name for an accessor is the same word (or words) as the name of its return type,
because in context, that type fully describes what the accessor returns. Since
our style rules require those words to be turned into identifiers in the same
way (`CamelCase` rather than `snake_case`), this results in shadowing, or an
outright name collision.

### Properties

Some programming languages allow classes to define
[property members](<https://en.wikipedia.org/wiki/Property_(programming)>),
which are accessed using data member syntax, but allow the type to implement
those accesses using procedural code. Properties can thus offer the clarity and
convenience of public data members, while retaining many of the advantages of
private data with "getters" and "setters".

However, the fact that properties can be implemented using arbitrary procedural
code means that they are capable of behaving in ways that would be very
surprising to a user who thinks of them as ordinary data members, such as
performing large amounts of computation, blocking on I/O, or causing observable
side effects when the property is being read. Usage of properties therefore
involves a style tradeoff between expressive power and user surprise.

Properties are a popular feature of several of Carbon's peer languages, so we
are likely to eventually explore the possibility of adding them to Carbon. When
we do, it may be useful to have some experience with the style tradeoffs that
properties create.

## Proposal

I propose to allow C++ classes in the Carbon project to provide methods that are
named like properties, so long as they behave like properties. I also propose to
require data members to have a trailing `_`. See the changes to
`cpp_style_guide.md` for further specifics.

## Rationale based on Carbon's goals

Avoiding the name conflicts will incrementally advance our "Language tools and
ecosystem" goals by making those tools somewhat easier to implement.

Early experimentation with properties in C++ could help us design Carbon
properties to better address several of Carbon's goals, especially "code that is
easy to read, understand, and write".

## Alternatives considered

### Disallow "setters"

We could narrow the style rule to disallow `set_` methods.

Advantages:

-   This minimizes the likely syntactic divergence from Carbon properties,
    because reading a Carbon property will probably look very much like an
    accessor call, minus the `()`, but assigning to a Carbon property will
    probably look like an assignment, not a `set_` method call.

Disadvantages:

-   This would lead to inconsistencies where a `foo_bar()` accessor is paired
    with a `SetFooBar()` setter.
-   This would prevent us from gaining experience that could shed light on the
    design of mutable properties in Carbon.

### Disallow mutable accessors

We could narrow the style rule to disallow property accessors that provide
non-const access to the property.

Advantages:

-   It would be a more narrow feature to allow in our C++ code.
-   Carbon doesn't yet have a property design and we don't know the extent or
    mechanisms it might use for mutable access.
-   In C++, mutable accessors directly expose the underlying storage, removing
    most abstraction opportunities.

Disadvantages:

-   This would prevent us from resolving name collisions in cases that involve
    in-place mutation, and lead to inconsistencies where a `foo_bar()` const
    accessor is paired with a `FooBar()` (or perhaps `MutableFooBar()`)
    non-const accessor.
-   This would prevent us from gaining experience that could shed light on the
    design of mutable properties in Carbon.

### Looser restrictions on property method performance

We could allow methods to use property-like naming, even if their performance is
substantially worse than the cost of accessing a data member, so long as the
overall performance of the code calling the method is likely to be comparable.

For example, this could permit property methods to return strings by value, even
though that requires linear time in the length of the string, because the
subsequent usage of the string will very likely be linear time as well, in which
case there is no overall asymptotic slowdown.

Advantages:

-   It would allow us to define methods that behave like "computed properties"
    for types like strings that are not fixed-size, but that are typically
    treated as single values rather than collections.

Disadvantages:

-   It would be a substantially more subjective rule, since it involves
    guesswork about what usages are likely.
-   It would carry a greater risk of user surprise.

### Require trailing `_` only for property data members

It's necessary to have trailing `_`s (or some equivalent convention) on data
members that provide storage for properties, because otherwise their names would
collide with the names of member accessors. However, we could continue to forbid
a trailing `_` on all other data members.

Advantages:

-   This would be a more minimal change to our existing style, and in particular
    it would avoid putting any existing code out of compliance with the style
    guide.

Disadvantages:

-   It might be harder to predict and internalize which data members have a
    trailing `_`.
-   The trailing `_` can provide readability benefits even for non-property
    members, by explicitly marking member accesses in a way that distinguishes
    them from local variable accesses.
