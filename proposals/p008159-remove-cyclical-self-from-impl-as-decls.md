# Remove cyclical `.Self` from `impl as` decls

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8159)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Make `.Self` less powerful inside `impl as`](#make-self-less-powerful-inside-impl-as)
    -   [Make the toolchain implementation closer to logic programming](#make-the-toolchain-implementation-closer-to-logic-programming)

<!-- tocstop -->

## Abstract

The `.Self` in an `impl` declaration is cyclic. It can expose a dependency on
the `impl` declaration before it exists which can cause code to not behave
according to expectations. We propose to require cycles that depend on the
`impl` appear in the definition, where the cyclic dependency can be satisfied.
To that end, we remove the name `.Self` in an `impl` declaration.

## Problem

Within an `impl` declaration, `.Self` can appear in the facet type being
implemented. It refers symbolically to the self facet of the `impl` declaration,
and gives access to the interface being implemented through the self facet. This
is a cyclical dependency, where the facet type can depend on the `impl`
declaration. Within the facet type, we can employ various tricks to allow that
cycle to function, but it can also be exposed externally through `eval fn`, or
deduction in other generic `impls` declarations, where we are unable to fully
honour the cyclical dependency. This allows the user to construct code that
seems like it should work, accessing associated constants of the interface whose
values can not become available at least until the `impl` declaration is
complete.

For example, given this code:

```carbon
interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {}
impl forall [T: Z where .Z1 = {}] T as Y {}

class C(T: Y) {}
impl () as Z where .Z1 = {} and .Z2 = C(.Self) {}
```

The specific `C(.Self)` refers to `C(() as Z)` and to the left of it the user
has already written `.Z1 = {}`. So we'd expect the `impl` for `T as Y` to be
matched and used. However facets encode only witnesses, not the value of
associated constants, as those are to be looked up through the witness. But the
witness for `() as Z` does not exist while building the facet type on the
right-hand-side of `impl () as ...`. So this makes a cycle where the `impl () as
Z` declaration depends on its own declaration to be completed in order to be
checked.

## Background

-   [Proposal 5168](/proposals/p005168-forward-impl-declaration-of-an-incomplete-interface.md)
    introduced `where` assignment declarations inside the body of the `impl`
    definition. This provides a different place to write self-referential values
    for associated constants that is not problematic.
-   This was discussed in open discussion on
    [2026-11-12](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.uvyw2c4zrz9e#heading=h.8iwpeulxc8hd).

## Proposal

We propose to not introduce the name `.Self` for the `where` immediately on the
right-hand-side of an `impl as` declaration.

Generally `where` introduces the name `.Self` which is a facet symbolically
representing the type that will satisfy the extended constraints found on the
left-hand-side of the `where` operator. In an `impl as` declaration, this
`.Self` contains a witness which can not be used to access associated constants
yet. So we propose to eliminate the name entirely in this context.

The use of `where X = Y` assignments inside the `impl` definition can be used
for any associated constants that depend on the `impl`'s self facet. Inside the
definition, it is referred to as `Self`, which is the actual self facet. The
witness provided by the `impl` declaration is accessible through `Self`, both
inside the `impl` definition and externally by anything it refers to, making
cyclic dependencies feasible to satisfy.

This does reduce expressiveness of forward declarations of an `impl`. It removes
the ability to forward declare an `impl` with the value of an associated
constant that depends on the self facet in a way that requires knowing it will
implement the interface that the `impl` is targeting. We don't have any
reasonable examples where this would be necessary to break a cycle.

## Rationale

This advances Carbon's goals of making an efficient toolchain implementation. In
order to satisfy the cycle described in the [Problem](#problem), the toolchain
would need to track additional state, or introduce deferred checking. These make
the language more complex to implement a toolchain for, and make the toolchain
slower. In general, we want to offload complexity around cycles to the user by
allowing them to order their code to break cycles, as given by the
[information accumulation principle](/docs/project/principles/information_accumulation.md).

Here we still have a cycle, but we can ensure the user code provides the
dependency, the `impl` declaration, before it are used in the `impl` definition.

## Alternatives considered

### Make `.Self` less powerful inside `impl as`

We could explicitly give `.Self` access to the declaration (the existence of the
witness) but not to the definition (associated entities). That matches the
failure we find in the example from the [Problem](#problem) section. This makes
`.Self` behave differently in different contexts though, which we worry would be
confusing for users. `.Self` would be enough to get access to associated
entities when used in facet types anywhere other than an `impl` declaration. And
even worse, within an `impl` declaration it might still give access to a
_different_ and incorrect `impl` definition if there is overlap between them.

### Make the toolchain implementation closer to logic programming

Building on a logic programming model, we could establish "facts" as we
encounter them in a single pass. Then check against those facts in a second
pass. This would make it possible to check facts that are obtained through a
cyclic dependency. However we want to perform semantic checking in a single pass
over the parse tree for performance reasons. We note that the Polonius borrow
checker was originally written by way of a fact-based datalog model but it was
too slow for production use.
