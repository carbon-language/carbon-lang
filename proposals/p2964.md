# Expression phase terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2964)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Update terminology around expression properties:

-   "value phase" -> "expression phase"
-   "symbolic value" -> "symbolic constant"
-   "constant" -> "template constant"
-   "constant or symbolic value" -> "constant"

Implements the decision in
[#1391](https://github.com/carbon-language/carbon-lang/issues/1391).

## Problem

There are a few concerns with the current terminology:

-   We were not happy with it at the time it was introduced in
    [proposal #1378](https://github.com/carbon-language/carbon-lang/pull/1378/files/198c7bd152bc4e223f1fc2484455e17ba31f19f9#r921592116),
    but didn't want to block that proposal on finding better names.
-   It is a property of expressions, not values. We are switching from "value
    categories" to "expression categories" for the same reason. See
    [proposal #2006](https://github.com/carbon-language/carbon-lang/pull/2006)
    and [PR #2744](https://github.com/carbon-language/carbon-lang/pull/2744).
-   No good term for the bindings that use `:!`, which share a number of
    properties, and often we want to refer to those together.

The concerns about the names of the individual phases led to
[questions-for-leads issue #1391: New name for "constant" value phase](https://github.com/carbon-language/carbon-lang/issues/1391).
This proposal implements the resolution from that issue.

## Background

The current names were introduced in
[proposal #1378: Design overview update part 7: values](https://github.com/carbon-language/carbon-lang/pull/1378)
based on a discussion in
[#typesystem on Discord](https://discord.com/channels/655572317891461132/708431657849585705/996547601451204630).
I am not aware of corresponding terminology in other programming languages.

## Proposal

We update terminology around expression properties, specifically what we
previously referred to as "value phase" and values it can have, as follows:

-   "value phase" -> "expression phase": since this is a property of
    expressions, not values.
-   "symbolic value" -> "symbolic constant": for symbolic compile-time values
    like checked-generic parameters.
-   "constant" -> "template constant": for compile-time values where the value
    is available during type checking, like literals and `template` parameters.

By making these last two terms both end in "constant," we allow their
combination (including all compile-time values) to be collectively referred to
using the term "constant." For example, either kind of constant may be passed to
a function with either kind of constant binding.

## Rationale

The goals of this proposal are:

-   clearer and more concise communication, and
-   making it easier for people to learn Carbon by using more consistent
    terminology.

This supports Carbon's goal of having
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).

## Alternatives considered

We were happy with this option and did not spend time coming up with more
alternatives.
