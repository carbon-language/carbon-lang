# What can be done with an incomplete interface

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2347)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow declaration and definition in different files](#allow-declaration-and-definition-in-different-files)
    -   [No specific restriction on `where` clauses on incomplete constraints](#no-specific-restriction-on-where-clauses-on-incomplete-constraints)
    -   [Don't allow combinations of incomplete constraints as in `C & D`](#dont-allow-combinations-of-incomplete-constraints-as-in-c--d)

<!-- tocstop -->

## Abstract

Clarify what can and cannot be done with a incomplete interface or named
constraint.

## Problem

For purposes of this proposal, the term _constraints_ refers to interfaces and
named constraints. Interfaces do not require any different treatment from named
constraints, and treating them the same is good for simplicity and uniformity.

We want to support forward declaring constraints so that they may be used before
they are allowed to be defined, due to recursion or a reference cycle. Some uses
require the definition of the constraint to be visible to be checked, however.
In some cases, those checks may be delayed at the cost of complexity in the
implementation of the compiler. This proposal aims to specify what uses are
allowed, particularly those cases that were not covered by
[the previous proposal on forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084).

## Background

Proposal
[#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
originally laid out the rules for incomplete constraints. Since then, the
[discussion on 2022-10-12](https://docs.google.com/document/d/1tEt4iM6vfcY0O0DG0uOEMIbaXcZXlNREc2ChNiEtn_w/edit#heading=h.q7afaawbc5k)
considered how we needed to update and fill in those rules, as part of an effort
to implement those rules in the Carbon Explorer.

## Proposal

The new rule details have been added by this proposal PR to the
[Forward declarations and cyclic references: declaring interfaces and named constraints](/docs/design/generics/details.md#declaring-interfaces-and-named-constraints)
section of
[Generics: details design document](/docs/design/generics/details.md).

## Rationale

This proposal is balancing the expressivity needed to make code easier to write
with implementation simplicity, both concerns of Carbon's
["code that is easy to read, understand, and write"](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal.

## Alternatives considered

### Allow declaration and definition in different files

In particular, we considered allowing the declaration of an interface or named
constraint in an API file and the definition in the impl file of the same
library. We considered some use cases that might benefit from this in
[#931](p0931.md#private-interfaces-in-public-api-files) and
[#971](https://github.com/carbon-language/carbon-lang/issues/971). In
[#generics-and-templates on 2022-10-24](https://discord.com/channels/655572317891461132/941071822756143115/1034207895392358431),
we decided those use cases would be okay including the definition of the private
interface in the API file. So to support checking for invalid uses of an
incomplete interfaces with information local to that file, we reaffirmed the
decision in #971 and implemented in proposal
[#1084](https://github.com/carbon-language/carbon-lang/pull/1084) to require the
definition to be in the same file as the declaration.

### No specific restriction on `where` clauses on incomplete constraints

We considered not having a specific rule preventing any `where` clause on an
incomplete constraint. The idea is that the problematic cases, such as those
that access members of the constraint like `where .X = T`, are already
forbidden. By removing this rule, we would allow constructions like
`C where Vector(.Self) is Hashable` for incomplete `C`. We decided to wait until
we had a motivating use case to allow this, and start with the more restrictive
rule.

### Don't allow combinations of incomplete constraints as in `C & D`

We considered forbidding the combination of incomplete constraints with the `&`
operator, because otherwise we would not be able to diagnose conflicts between
the two constraints. For example,

```carbon
constraint C;
constraint D;

// Can't tell if there is a conflict between `C` and `D`.
fn F[T:! C & D](x: T);

// These definitions of `C` and `D` conflict.
constraint C {
  extends I where .X = i32;
}
constraint D {
  extends I where .X = bool;
}
```

Since our existing examples of using forward declarations to define a graph with
cyclic references between the edge and node interfaces needed this feature, we
decided to support it. This motivated the requirement that
[the definition be in the same file as the declaration](#allow-declaration-and-definition-in-different-files),
so the error could be detected once the compiler reached the definitions.
