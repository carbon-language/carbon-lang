# Clarify support for imported object-like macros

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7308)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

This proposal clarifies some unclear aspects of the interop support for
object-like macros. In particular:

-   Carbon supports importing an object-like macro if its definition can be
    evaluated as a constant expression, without further restrictions on that
    definition.
-   When importing the result of that evaluation, C++ lvalues are imported as
    references, and rvalues are imported as values.

## Problem

Proposal [p006676](p006676-carbon-c-interop-importing-c-c-object-like-macros.md)
introduced support for importing object-like macros by evaluating their
definitions as C++ constant expressions. By the time the proposal was adopted,
this feature was intended to support most if not all macros that permit such
evaluation, but due to its drafting history, the proposal text seems to limit
the macro definition to a narrow allow-list of C++ operations (which excludes
function calls, among other things).

In addition, p006676 states that if the constant value refers to a named
constant, it is imported as an alias rather than a value, but it's not entirely
clear what counts as a named constant, or what the alias/value distinction means
(particularly when applied to C++ rvalues like enumerators). Here again, the
proposal text doesn't fully reflect the design intent, which was that the
expression category of the imported constant reflects the C++ value category of
the constant expression.

## Proposal

This proposal rephrases the design of this feature to avoid the impression of an
allow-list. It also replaces the discussion of named constants with a statement
that lvalues are imported as references, and rvalues are imported as values.

## Details

See the changes in `docs/design/interoperability/macros.md` in the PR for this
proposal.

## Rationale

This proposal advances the
[Community and culture](/docs/project/goals.md#community-and-culture) goal, by
ensuring that the design choices we made for this feature are clearly
documented, and that the actual design (as adopted by the evolution process)
matches the design as understood by the leads and the team.

## Alternatives considered

We considered treating this as an ordinary PR editing the design documentation,
without going through the evolution process, under the rationale that these
changes are merely clarifications that reflect what we intended p006676 to mean
when we adopted it. However, that rationale seems questionable in this case,
given the extent of the changes, and the lack of evidence for that intent in the
adopted proposal text.
