# Generics details 9: forward declarations

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1084)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [No `default` keyword on interface members](#no-default-keyword-on-interface-members)
    -   [Declaring an implementation of an incomplete interface](#declaring-an-implementation-of-an-incomplete-interface)
    -   [Allow definition of private interfaces in separate impl file](#allow-definition-of-private-interfaces-in-separate-impl-file)
    -   [No implementations for incomplete types](#no-implementations-for-incomplete-types)
    -   [No forward declaration of named constraints](#no-forward-declaration-of-named-constraints)
    -   [Repeating `private` in both declaration and definition](#repeating-private-in-both-declaration-and-definition)
    -   [Allow function bodies using incomplete interfaces](#allow-function-bodies-using-incomplete-interfaces)
    -   [Don't require parameter names to match](#dont-require-parameter-names-to-match)
    -   [Allow deduced parameters to vary](#allow-deduced-parameters-to-vary)

<!-- tocstop -->

## Problem

Developers want to organize their code for readability and convenience. For
example, they may want to present the public API of their type in a concise way.
That includes the ability to say a type implements an interface without
repeating the full contents of that interface.

The Carbon compiler can give better diagnostics if it can assume every
identifier it encounters refers to some earlier declaration in the file.
However, sometimes multiple entities will reference each other in a cycle so no
one entity can be defined first.

## Background

We have decided to tackle these problems in a manner similar to C++ by
supporting forward declarations:

-   [issue #472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472)
-   [proposal #875: Principle: information accumulation](https://github.com/carbon-language/carbon-lang/pull/875).

Use of the `default` keyword in `interface` definitions to allow defaulted
members to be defined out-of-line was originally proposed in
[withdrawn proposal #1034](https://github.com/carbon-language/carbon-lang/pull/1034).

This proposal implements the decisions in
[issue #1132: How do we match forward declarations with their definitions?](https://github.com/carbon-language/carbon-lang/issues/1132)
as they apply to generic interfaces, implementations, and so on.

## Proposal

This proposal makes changes to these sections of the
[generics details design document](/docs/design/generics/details.md):

-   [Forward declarations and cyclic references](/docs/design/generics/details.md#forward-declarations-and-cyclic-references)
    section added
-   [Interface members with definitions](/docs/design/generics/details.md#interface-members-with-definitions)
    section added to

## Rationale based on Carbon's goals

Forward declarations are intended to advance these goals:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem),
    by making Carbon easier to interpret by tooling in a single top-down pass.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
    by allowing developers to separate declaration from definition when
    organizing the presentation of their code, and imposing constraints that
    allow readers to interpret the code with less skipping around.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    from potential build performance improvements that come from allowing an
    `impl` to be defined in the `impl` file instead of the `api` file.

The rationale behind using forward declarations are covered in more detail in:

-   [issue #472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472)
-   [proposal #875: Principle: information accumulation](https://github.com/carbon-language/carbon-lang/pull/875).

## Alternatives considered

### No `default` keyword on interface members

Without the `default` keyword, default definitions would always have to be
inline. We discussed this in
[the #syntax channel on Discord](https://discord.com/channels/655572317891461132/709488742942900284/941408009689641010)
which eventually led to the
[question-for-leads issue #1082: Use `default` keyword in interface defaults?](https://github.com/carbon-language/carbon-lang/issues/1082).

The conclusion was that we did want to support forward declarations of default
interface members. To make it so that users would have a single place to look to
see whether the member had a definition even when it might be out of line, we
decided to use a `default` keyword as a prefix of the declaration. We considered
putting the keyword at the end of the declaration, but we decided it was more
readable if it wasn't next to the return type. It was also more consistent with
`final`, an alternative to `default`, which also now supports forward
declaration.

### Declaring an implementation of an incomplete interface

We did not have any use cases for forward declaring an impl of an incomplete
interface, and so we took the conservative position of forbidding that. We could
add this feature in the future if use cases were found, but clearly we can't
have impl definitions until the interface is defined.

### Allow definition of private interfaces in separate impl file

This proposal requires the definition of an interface to be in the same file as
any declaration of it. We
[anticipate](https://github.com/carbon-language/carbon-lang/pull/1084#discussion_r824214281)
the possibility that we will find a use case for declaring a private interface
in an API file that is defined in the corresponding impl file. An example where
this may arise is if the constraint is only used when defining private members
of an exported class. We would be willing to change if we see demand for this in
the future.

### No implementations for incomplete types

For simplicity, generally Carbon entities should either be "incomplete" or
"defined" and never "partially defined". However, the set of interfaces
implemented for a type is by necessity only ever partially known by the nature
of being the
[one static open extension mechanism](https://github.com/carbon-language/carbon-lang/pull/998)
in Carbon. As a result, we felt there was more leeway for implementing
interfaces for incomplete types. This happens incidentally when implementing the
interface inline in the scope of a class definition. We also wanted to allow it
in the case where there was only a forward declaration of the type in an API
file.

### No forward declaration of named constraints

We considered omitting the ability to forward declare named constraints, but we
discovered that ability made declaring interfaces with cyclic dependencies
easier and cleaner. Without this feature,
[the graph example of cyclic references](/docs/design/generics/details.md#example-of-declaring-interfaces-with-cyclic-references)
looked like this instead:

```
// Forward declaration of interface
interface EdgeInterface;

// Definition that only uses the declaration of
// `EdgeInterface`, not its definition.
interface NodeBootstrap {
  let EdgeType:! EdgeInterface;
  fn Edges[me: Self]() -> Vector(EdgeType);
}

// Now can define `EdgeInterface` in terms of
// `NodeBootstrap`.
interface EdgeInterface {
  let NodeType:! NodeBootstrap where .EdgeType == Self;
  fn Head[me: Self]() -> NodeType;
}

// Make `NodeInterface` a named constraint defined in
// terms of `NodeBootstrap`, adding in constraints that
// couldn't be written until `EdgeInterface` was defined.
constraint NodeInterface {
  extends NodeBootstrap where .EdgeType.NodeType == Self;
}
```

We did not like how the definition of `NodeInterface` was split into two pieces,
making it harder to understand what it contained.

This question was discussed in
[the #generics channel on Discord](https://discord.com/channels/655572317891461132/941071822756143115/951288264315265114).

### Repeating `private` in both declaration and definition

We considered repeating the access-control keyword `private` as a prefix of all
`impl` declarations and definitions. The
[current rule](/docs/design/generics/details.md#declaring-interfaces-and-named-constraints)
only marks the first declaration or definition, which is consistent with
[the policy of not repeating access-control keywords stated in an API file in an impl file](/docs/design/code_and_name_organization#exporting-entities-from-an-api-file).

This was discussed in
[the #syntax channel on Discord](https://discord.com/channels/655572317891461132/709488742942900284/951520959544823868),
but this decision should be considered provisional since it was not considered
deeply. We would be open to revisiting this decision in the future, once we had
some experience with it.

### Allow function bodies using incomplete interfaces

We
[considered](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.oqmpxtubjmkm)
allowing a function definition to use an incomplete interface. One concern was
whether the criteria for when the function body depended on something in the
interface's definition would be too subtle for developers to reason about. We
eventually concluded that, unless using a monomorphization compilation strategy,
efficient code generation for a generic function would need to use the
interface's definition. For example, an interface that represented a single
function call might use a function pointer instead of a witness table. This same
argument led to the requirement that the interface's definition be visible at
call sites as well.

### Don't require parameter names to match

We decided to diverge from C++ in requiring parameter names to match between
declarations for a few reasons:

-   wanting to avoid the confusion that we've experienced when they don't match,
    noting that common C++ lint tools ask to make them match;
-   wanting reflection to return a single parameter name for a parameter; and
-   wanting the parameter names to be consistent with the single docstring we
    expect to associate with a function.

This was discussed in
[open discussion on 2022-03-14](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.oqmpxtubjmkm)
and
[question-for-leads issue #1132](https://github.com/carbon-language/carbon-lang/issues/1132).

### Allow deduced parameters to vary

We decided to apply
[the same matching requirements for other parameter names](#dont-require-parameter-names-to-match)
to deduced parameters for consistency. We may in the future allow some rewrites
between equivalent expressions, such as between `Vector(T:! Type)` and
`[T:! Type] Vector(T)`, but for now we are starting with the more restrictive
rule. This was discussed in
[open discussion on 2022-03-24](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.w4zgqvarhnbn)
and in
[#syntax channel on Discord](https://discord.com/channels/655572317891461132/709488742942900284/953798170750615622).
