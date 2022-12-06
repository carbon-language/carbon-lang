# Generic details 11: operator overloading

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1144)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Weak impls instead of adapters for reverse implementations](#weak-impls-instead-of-adapters-for-reverse-implementations)
    -   [Default impls instead of adapters for reverse implementations](#default-impls-instead-of-adapters-for-reverse-implementations)
    -   [Allow an impl declaration with `like` to match one without](#allow-an-impl-declaration-with-like-to-match-one-without)
    -   [Where are the impl definitions from `like` generated?](#where-are-the-impl-definitions-from-like-generated)
    -   [Support marking interfaces or their members as `external`](#support-marking-interfaces-or-their-members-as-external)

<!-- tocstop -->

## Problem

C++ supports
[operator overloading](https://en.wikipedia.org/wiki/Operator_overloading), and
we would like Carbon to as well. This proposal is about the general problem, not
the specifics application to any particular operator.

This proposal does not attempt to define a mechanism by which we can ensure that
`a < b` has the same value as `b > a`.

## Background

The generics feature is the
[single static open extension mechanism](/docs/project/principles/static_open_extension.md)
in Carbon, and so will be what we use operator overloading. We have already
started specifying the ability to extend or customize the behavior of operators
by implementing interfaces, as in these proposals:

-   [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
-   [#845: as expressions](https://github.com/carbon-language/carbon-lang/pull/845)
-   [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911)
-   [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

Proposal
[#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
specified
[using interfaces for overloading the comparison operators](p0702.md#overloading),
but did not pin down specifically what those interfaces are.

## Proposal

This proposal adds an
["Operator overloading" section](/docs/design/generics/details.md#operator-overloading)
to the [detailed design of generics](/docs/design/generics/details.md).

## Rationale based on Carbon's goals

This proposal advances Carbon's goals:

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
    by making common constructs more concise, and allowing the syntax to more
    closely mirror notation used math or the application domain.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution),
    since this allows standard types to implement operators using the same
    mechanisms that user types do, this allows changes between what is built-in
    versus provided in a library without user-visible impact.

## Alternatives considered

The current proposal requires the user to define a reverse implementation, and
recommends using an adapter to do that more conveniently. We also considered
approaches that would provide the reverse implementation more automatically.

### Weak impls instead of adapters for reverse implementations

We proposed
[weak impls](https://github.com/carbon-language/carbon-lang/pull/1027) as a way
of defining blanket impls for the reverse impl that did not introduce
[cycles](/docs/design/generics/details.md#acyclic-rule). We rejected that
approach due to giving the reverse implementation the wrong priority. This meant
that there were many situations where `a < b` and `b > a` would give different
answers.

### Default impls instead of adapters for reverse implementations

We then proposed
[default impls](https://github.com/carbon-language/carbon-lang/pull/1034) as a
way to define reverse implementations. These were rejected because they had a
lot of overlap with blanket impls, making it difficult to describe when to use
one over the other, and because they introduced a lot of complexity without
fully solving the priority problem. Most of the complexity was from the criteria
for determining whether the default implementation would be used. As
[noted](/docs/design/generics/details.md#binary-operators), the current proposal
still has some priority issues, but this way the relevant impls are visible in
the source which will hopefully make it clearer why it happens.

The capability provided by default impls -- the ability to conveniently give
implementations of other interfaces -- may prove useful enough that we would
reconsider this decision in the future.

### Allow an impl declaration with `like` to match one without

We considered allowing an impl declared with `like` to match the equivalent
impls without `like`. The main concern was there would not be a canonical form
without `like`, particularly of how the newly introduced parameter would be
written. We thought we might say that the `like` declaration, since it omits a
spelling of the parameter, is allowed to match any spelling of the parameter.
However, there would still be a question of whether to use a deduced parameter,
as in `[T:! ImplicitAs(i64)] Vector(T)` or not as in
`Vector(T:! ImplicitAs(i64))`. We also considered the canonical form of
`Vector(_:! ImplicitAs(i64))` without naming the parameter. In the end, we
decided to start with a restrictive approach with the knowledge that we could
change once we gained experience.

The main use case for allowing declarations in a different form, which may
motivate changes in the future, is to prioritize the different implementations
generated by the `like` shortcut separately in `match_first` blocks.

This was discussed in
[open discussion on 2022-03-24](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#).

### Where are the impl definitions from `like` generated?

We considered whether the additional impl definitions would be generated with
the first declaration of an impl using `like` or with its definition. We
ultimately decided on the former approach for two reasons:

-   The generated impl definitions are parameterized even if the explicit
    definition is not, and parameterized impl definitions may need to be in the
    API file to allow separate compilation.
-   This will make the code doing implicit conversions visible to callers,
    allowing it to be inlined, matching how the caller does implicit conversions
    for method calls.

This was discussed in
[the #generics channel on Discord](https://discord.com/channels/655572317891461132/941071822756143115/962059164014739526).

### Support marking interfaces or their members as `external`

We
[discussed on 2022-03-28](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.sk06n1ggoa3o)
the idea that operator interfaces might be marked `external`. This would either
mean that types would only be able to implement them using `external impl` or
that even if they were implemented internally, they would not add the names of
the interface members to the type. Alternatively, individual members might be
marked `external` to indicate that their names are not added to implementing
types, which might also be useful for making changes to an interface in a
compatible way.

We were not sure if this feature was needed, so we left this as future work.
