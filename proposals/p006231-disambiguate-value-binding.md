# Disambiguate "value binding"

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6231)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Rename "a binding declared by a value binding pattern"](#rename-a-binding-declared-by-a-value-binding-pattern)
    -   [Other names for "primitive conversion from reference to value"](#other-names-for-primitive-conversion-from-reference-to-value)

<!-- tocstop -->

## Abstract

This proposal removes the definition of the term "value binding" as a primitive
category conversion from reference to value, replacing it with the term "value
acquisition". The other meaning of "value binding", a binding declared by a
value binding pattern, is unchanged.

## Problem

The design docs currently define "value binding" in two conflicting ways: it can
mean the binding declared by a value binding pattern, or it can mean a primitive
category conversion from reference to value. The two can usually be
disambiguated based on context, but it's not always straightforward, and the
double meaning complicates naming within the toolchain implementation.

## Proposal

This proposal removes the definition of the term "value binding" as a primitive
category conversion from reference to value, replacing it with the term "value
acquisition".

## Details

See the changes elsewhere in the
[proposal PR](https://github.com/carbon-language/carbon-lang/pull/6231).

## Rationale

Using unambiguous terminology advances our
[community and culture](/docs/project/goals.md#community-and-culture) goals, by
facilitating clear communication.

## Alternatives considered

### Rename "a binding declared by a value binding pattern"

We could instead rename the other meaning of "value binding", but that would be
considerably more difficult because that meaning appears to be more common, and
because it's part of a cluster of other heavily-used terms, such as "reference
binding" and "binding pattern", which we would need to rename for consistency.

### Other names for "primitive conversion from reference to value"

We considered several alternative names before settling on "value acquisition":

-   "Value borrowing" highlights the close analogy to Rust borrowing, which
    similarly forbids mutation of the object for the lifetime of the borrow.
    However, this naming choice somewhat prejudges the safety story for this
    operation.
-   "Value snapshotting" and "value observation" may not effectively communicate
    the ongoing coupling between the object and the value.
-   "Value capturing" reuses and extends the existing meaning of "capturing" in
    lambdas. However, it may be confusing that a lambda can have value captures
    that are not initialized by value capturing (for example because the
    initializer is a value, not a reference).
-   "Value expression conversion" is straightforward and hard to misunderstand.
    However, we sometimes use "value binding" to refer to the _result_ of a
    reference-to-value conversion, for example "the lifetime of a value
    binding". "Value expression conversion" doesn't seem to lend itself to that
    usage, possibly because it's too generic to be a recognizable term of art.
