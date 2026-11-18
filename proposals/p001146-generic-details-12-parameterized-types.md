# Generic details 12: parameterized types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1146)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

Most aspects of generic parameterization are the same between functions and
types, but there are a few things specific to types. In particular:

-   the declaration of a type can contain a greater variety of members, and
-   types have identity which affects type comparisons, deduced parameters, and
    implied constraints.

We also want a
[generic specialization](/docs/design/generics/terminology.md#checked-generic-specialization)
story that works well for types, without giving up the ability to type check
users of a type without knowing which specializations apply.

## Background

C++ supports specialization, including partial specialization, for templated
types and functions.

## Proposal

This proposal adds a
["parameterized types" section](/docs/design/generics/details.md#parameterized-types)
to the [detailed design of generics](/docs/design/generics/details.md). Of note,
it proposes not to support specialization of types or functions since those use
cases can be handled by delegating to interfaces, which already support
specialization.

## Rationale based on Carbon's goals

Specialization is important for allowing code to be generic without sacrificing
[performance](/docs/project/goals.md#performance-critical-software). Since there
is already a way to support specialization use cases without adding direct
support for specializing types, this proposal follows the Carbon principle to
[prefer providing only one way to do a given thing](/docs/project/principles/one_way.md).
By avoiding another way of customizing behavior for specific types, it makes
interfaces the
[single static open extension mechanism](/docs/project/principles/static_open_extension.md).
This proposal maintains consistency between generic parameterization of types
and functions, in support of
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).

## Alternatives considered

We considered supporting specialization for types directly. To support type
checking in a generic context, the API of the type needs to be defined
independent of which specialization is selected. This would have introduced
complexity into the language:

-   Would the API of the type be represented by an `interface`?
-   Would the type's API be explicitly declared or inferred from the type
    declaration by some process, at the risk of including details that are not
    necessarily stable?
-   How would public data members be handled, since interfaces (currently) don't
    support them?
-   How would we support non-monomorphizing generic strategies? With the current
    proposal, the layout of a parameterized type is known unless it uses an
    associated type.

The main disadvantage of the proposed approach is that the author of the type
needs to define the ways that the type can be customized. We will need to see if
this ends up being a problem in practice. It may turn out to be a benefit, by
giving more information about the implementation of a class to readers.
