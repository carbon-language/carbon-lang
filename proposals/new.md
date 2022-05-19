# Principle: All APIs are library APIs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [TODO: Initial proposal setup](#todo-initial-proposal-setup)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Built-in primitive types](#built-in-primitive-types)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./new_proposal.py "TITLE"` to do new proposal setup.

1. Copy this template to `new.md`, and create a commit.
2. Create a GitHub pull request, to get a pull request number.
    - Add the `proposal` and `WIP` labels to the pull request.
3. Rename `new.md` to `/proposals/p####.md`, where `####` should be the pull
   request number.
4. Update the title of the proposal (the `TODO` on line 1).
5. Update the link to the pull request (the `####` on line 11).
6. Delete this section.

TODOs indicate where content should be updated for a proposal. See
[Carbon Governance and Evolution](/docs/project/evolution.md) for more details.

## Problem

We need a clear and consistent "division of labor" between the core language and
the standard library.

## Background

See
[the principle doc](/docs/project/principles/library_apis_only.md#background).

See also #57, a previous iteration of a similar idea.

## Proposal

In Carbon, every public function will be declared in some Carbon `api` file, and
every public `interface`, `impl`, and first-class type will be defined in some
Carbon `api` file. In some cases, the bodies of public functions will not be
defined as Carbon code, or will be defined as hybrid Carbon code using
intrinsics that aren't available to ordinary Carbon code. However, we will try
to minimize those situations.

Thus, even "built-in" APIs can be used like user-defined APIs, by importing the
appropriate library and using qualified names from that library, relying on the
ordinary semantic rules for Carbon APIs.

## Details

See [the principle doc](/docs/project/principles/library_apis_only.md).

## Rationale

This principle facilitates
[software evolution](/docs/project/goals.md#software-and-language-evolution), by
helping to ensure that code written in terms of a Carbon-provided type can be
migrated to use a suitable user-defined type instead. It also helps make Carbon
code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
because the syntax and core concepts are consistent between user-defined and
language-defined APIs. It also indirectly helps Carbon support
[performance-critical software](/docs/project/goals.md#performance-critical-software):
by using Carbon's API abstraction mechanisms for even the most fundamental
types, we ensure that those mechanisms do not impose any performance overhead.

## Alternatives considered

### Built-in primitive types

We could follow the general outline of C++, where arithmetic and pointer types
are built-in. However, that would substantially erode the advantages outlined
above. We expect Carbon to have multiple kinds of pointers (for example, to
represent different kinds of ownership), and multiple kinds of arithmetic types
(for example, to handle overflow in different ways). They can't all be built-in,
so putting even the common-case types in the library helps ensure that Carbon
has enough expressive power for the uncommon-case library types.
