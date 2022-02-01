# Generics: Set associated constants using `where` constraints

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1013)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Status quo](#status-quo)
    -   [`with` and `,` instead of `where` and `and`](#with-and--instead-of-where-and-and)
-   [Future work](#future-work)

<!-- tocstop -->

## Problem

There are a variety of contexts that currently use the keyword `let`:

-   declaring associated constants or types in an interface,
-   defining associated constants or types in an implementation,
-   defining local constant in a function body, and
-   defining class constants.

In all but the implementation case, the semantics are generally similar to the
semantics of passing a value into a function, with some erasing of the specific
value passed and using the type to determine how the name can legally be used.
However,
[proposal #950](https://github.com/carbon-language/carbon-lang/pull/950) has
changed the `let` in an implementation to use the value specified, not its type,
creating an inconsistency with the other uses of `let`.

Furthermore, we have come to the realization that we still want to specify the
values of associated constants and types for an implementation even in an API
file where we only want to make a forward declaration. This makes that
information available to clients that only look at the API file, who need to
know those values for type checking, but otherwise don't need to see the full
definition of the implementation. This suggests that those assignments should be
declared outside of definition block's curly braces `{`...`}`.

Lastly, there is a bit of redundancy in Carbon since `where` clauses are also a
way of specifying the values of associated constants and types in other Carbon
contexts.

## Background

The `let` syntax for setting an associated type in an interface implementation
was originally decided in issue
[#739: Associated type syntax](https://github.com/carbon-language/carbon-lang/issues/739)
and implemented in proposal
[#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731).

Proposal
[#950: Generics details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
made two relevant changes:

-   The type part of a `let` in an `impl` block is no longer "load bearing": the
    only legal types are `auto` and whatever was in the corresponding interface.
    In particular, the `let` in an `impl` block does not erase.
-   There is now a defined meaning for a generic `let` statement in a function
    body that can erase depending on the type specified.

Combined with the `let` in an interface giving you an erased type, or archetype,
this has made the meaning of `let` in an `impl` block inconsistent with other
places using `let`.

## Proposal

The suggested change is to use a `where` clause as part of an `impl` declaration
to specify associated constants and types instead of `let` declarations inside
of the `impl` definition. In effect, it removes `let` declarations from `impl`
blocks in exchange for allowing an `impl` declaration to implement a constraint
expression instead of a simple interface or named constraint.

This proposal updates the following design docs on the generics feature to
reflect this change:

-   [docs/design/generics/overview.md](/docs/design/generics/overview.md)
-   [docs/design/generics/terminology.md](/docs/design/generics/terminology.md)
-   [docs/design/generics/details.md](/docs/design/generics/details.md)

## Rationale based on Carbon's goals

As a simplification, this proposal advances the goal of having Carbon
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
In particular, having a simple specification and be simple to implement.

This is an example of
[the "prefer providing only one way to do a given thing" principle](/docs/project/principles/one_way.md),
by switching to a single way of specifying associated constants and values.

## Alternatives considered

### Status quo

The main alternative considered was the status quo. We did have two concerns
with this proposal, however we felt that this behavior would not be surprising
to developers in practice.

**Concern:** Due to interface defaults, it is possible for copy-pasting the
type-of-type expression from an `impl` block in a `class` into a constraint in a
function signature to give a constraint that is weaker than what that impl block
actually delivers.

**Concern:** Because a specialization of an `impl` can change the values of
associated constants, a type might not actually satisfy a constraint that it
appears to implement when that constraint specifies the values of associated
constants. In this example:

```
interface Bar {
  let X:! Type;
}
class Foo(T:! Type) {
  impl as Bar where .X = T { ... }
}
```

it appears that `Foo(T)` satisfies the constraint that `Bar where .X = T`, but
there could be specializations that set `.X` to different values for some
specific values of `T`.

### `with` and `,` instead of `where` and `and`

Instead of matching the syntax used when specifying constraints, we could have
used a different syntax to highlight that this is assigning instead of
constraining. The suggestion that came up in discussion was using `with` instead
of `where` and a comma `,` instead of `and` to join multiple clauses.

We decided that it would not be good to have two syntaxes that were very similar
but different, and that there was some benefit to be able to copy-paste between
the constraint context and the implementation context.

## Future work

This proposal will allow us to support declaring that a type implements an
interface inside an API file separate from the definition of the `impl`, even
for internal `impl`s. However, that feature is waiting on resolution of
[#472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472)
and proposal
[#875: Principle: information accumulation](https://github.com/carbon-language/carbon-lang/pull/875).

If and when we do add support declaration of impls without definition, we will
need to answer the question: do you have to repeat `where` constraints from a
forward declaration of an impl when it is later defined?

```
class Vector(T:! Type) {
  impl as Container where .Element = T and .Iter = VectIter(T);
}

// Probably okay:
fn Vector(T:! Type).(Container.Begin)[me: Self]() ...

// Maybe okay:
class Vector(T:! Type) {
  // Not repeating constraints on .Element and .Iter above:
  impl as Container {
    fn Begin[me: Self]() ...
  }
}
```
