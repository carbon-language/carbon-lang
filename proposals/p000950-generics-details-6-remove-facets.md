# Generics details 6: remove facets

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/950)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Status quo: name lookup only in the type-of-type](#status-quo-name-lookup-only-in-the-type-of-type)
    -   [Status quo: facet types](#status-quo-facet-types)
    -   [Name lookup in type and type-of-type](#name-lookup-in-type-and-type-of-type)
    -   [No generic 'let'](#no-generic-let)
    -   [Generic 'let' erases connection to original type](#generic-let-erases-connection-to-original-type)

<!-- tocstop -->

## Problem

There were some concerns about facet types leaking out of generic code in return
types. Some initial fixes for this were done in
[PR #900](https://github.com/carbon-language/carbon-lang/pull/900), but there
remain concerns, for example when associated types are involved.

In particular, given an interface method with return type using an associated
type, as in:

```
interface Deref {
  let Result:! Type;
  fn DoDeref[me: Self]() -> Result;
}

class IntHandle {
  impl Deref {
    let Result:! Type = i32;
    fn DoDeref[me: Self]() -> Result { ... }
  }
}
```

Since `Result` has type `Type`, we had the problem that `IntHandle.DoDeref`
would have to return `i32 as Type`, instead of the desired `i32`.

We also think we can simplify the model by eliminating the facet type concept
and syntax.

## Proposal

This proposal removes facet types, introduces archetypes in their place,
clarifies how associated types work outside of a generic function, and specifies
how a generic `let` statement in a function body works.

## Details

The details of this proposal are in the changes to these generics design
documents:

-   [Overview](/docs/design/generics/overview.md)
-   [Goals](/docs/design/generics/goals.md)
-   [Terminology](/docs/design/generics/terminology.md)
-   [Details](/docs/design/generics/details.md)

## Rationale based on Carbon's goals

This proposal
[adds a goal](/docs/design/generics/goals.md#path-from-regular-functions) about
minimizing the differences between a regular function and one with a generic
parameter from the perspective of the caller. This is to support the
[software and language evolution](/docs/project/goals.md#software-and-language-evolution)
goal by reducing the amount of changes needed to source code when generalizing a
function.

This change is also a simplification to the programming model, removing a
concept that developers have to learn and reducing the number of types a program
deals with. This is in support of the
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal.

## Alternatives considered

This design was the conclusion of a number of discussions and proposals:

-   [#typesystem discussion in Discord on Nov 2](https://discord.com/channels/655572317891461132/708431657849585705/905248525028323368)
-   Nov 3, 2021 document by `zygoloid` titled
    [Member lookup in generic and non-generic contexts](https://docs.google.com/document/d/1-vw39x5YARpUZ0uD2xmKepLEKG7_u122CUJ67hNz3hk/edit#)
-   Nov 4, 2021 document by `josh11b` and `mconst` titled
    [Carbon: facets exposed from generics](https://docs.google.com/document/d/1C1eIzd6JY0ooE1rDjW1vx7e3i7sgGugCA9bPMRhwWM0/edit#)
-   [Open discussion on 2021-11-08](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.ec285oam2okw)
-   [Open discussion 2021-11-11](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.8vuatm82d1mk)

The main alternatives we evaluated are summarized below.

### Status quo: name lookup only in the type-of-type

The main problem with the status quo is that we would look up names in the
type-of-type whether or not the type value was known. This meant that there was
no way to have a type variable that didn't alter the API of the type it held,
even when the value of that variable was known at compile time. It also created
problems putting a particular facet type into an associated type, as was desired
when computing the `CommonType` of two facets of the same type, since the
type-of-type of the associated type would overwrite the facet. This last concern
was raised in this
[#typesystem discussion in Discord on Nov 2](https://discord.com/channels/655572317891461132/708431657849585705/905248525028323368).

### Status quo: facet types

Facet types as a user-visible type expression `MyType as MyConstraint` have a
few downsides:

-   Facet types introduce extra types for the user to concern themselves with.
-   Facet types introduce extra types for instantiation purposes.
-   We don't want the possibility of facet types leaking outside of generic
    code.

Archetypes have replaced using facet types to type check a function with a
generic parameter. Generic `let` and adapter types address trying to repeatedly
access methods from an interface implemented externally.

We considered making the other changes in this proposal with a different way of
forming a facet type than `MyType as MyConstraint`
[in this document](https://docs.google.com/document/d/1C1eIzd6JY0ooE1rDjW1vx7e3i7sgGugCA9bPMRhwWM0/edit#).
Ultimately we all agreed that we did not want to instantiate templates called
with a generic type with a facet type, and the removal of facet types simplifies
the design.

### Name lookup in type and type-of-type

[`zygoloid` proposed](https://docs.google.com/document/d/1-vw39x5YARpUZ0uD2xmKepLEKG7_u122CUJ67hNz3hk/edit#)
a solution of looking up names in both the type and the type-of-type, resolving
conflicts in the same way `&` would between two type-of-types. This has the nice
property of using the type information when it is available, but still allowing
the type-of-type to affect name lookup. This made the code inside and outside a
generic consistent as long as there were no name conflicts. The downside is that
this still involved types potentially changing when you added a generic
parameterization, even though the changes were smaller, and didn't have the
simplification advantages of removing facet types entirely.

We did like the idea from
[that proposal](https://docs.google.com/document/d/1-vw39x5YARpUZ0uD2xmKepLEKG7_u122CUJ67hNz3hk/edit#)
that we would perform look up in the type when it was a constant, since that
addressed the main problem we were trying to address.

### No generic 'let'

Generic `let` has two main use cases: manual monomorphizing a generic and making
repeated calls to members of an interface implemented externally more
convenient. The former may be performed by using qualified member names, and the
latter by an adapter type, so this feature is not strictly needed.

However, `chandlerc` believes including generic `let` is more consistent, for
example with the use of generic `let` to declare associated types in interfaces
and implementations. It make replicating the change in type behavior of calling
a generic function much more straightforward, without having to introduce a
function call or rewriting all of the member accesses to add qualification.

Removing generic `let` is a simplification we would consider in the future.

### Generic 'let' erases connection to original type

We considered the idea that a generic `let` would fully erase type identity,
[on discord as "option A"](https://discord.com/channels/655572317891461132/708431657849585705/908834806551445554).
This didn't have any clear advantages other than making a generic `let` in a
function body be more similar to a generic function parameter. Erasing the type
identity would have made generic `let` much harder to use, though, without any
clear ways to get values of the new type. The option we chose is more similar to
generic `let` used to declare an associated type.
