# Are We Explorer Yet?

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1891)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [AreWeYet](#areweyet)
    -   [Carbon Explorer Status Brainstorm](#carbon-explorer-status-brainstorm)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Are We Explorer Yet?](#are-we-explorer-yet-1)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

It is currently difficult to see the status of Carbon explorer and where effort
is needed. We propose creating a AreWeYet-styled dashboard to address this.

## Problem

While the Carbon project has many published documents, it can be a challenge to
get a high-level picture of work that has been done and work that remains to be
done. This can make it difficult to both track its progress and identify
contribution areas that will have the biggest impact. Left unchecked, this can
lead to the team equivalent of micro-optimization where resources are deployed
in a way that isn’t consistent with high-level goals.

## Background

### AreWeYet

Mozilla, a non-profit behind many Open Source projects, has created and utilized
what is known as the AreWeYet meme as an approach to solve the aforementioned
problem. In this approach, a highly structured dashboard is built that
succinctly states a goal, what its status is, and links to issues or other means
to track subgoals.

One example is [“Are We XBL Still?”](https://bgrins.github.io/xbl-analysis/))
which tracks a project to remove all XBL bindings from Firefox. The general
format is a question, “Are we X still?”, followed by a list of items that, when
complete, means the answer to the question becomes yes. Other examples can be
found on Mozilla’s [Areweyet page](https://wiki.mozilla.org/Areweyet).

The success of this methodology has led to other projects, such as Rust,
adopting it.

### Carbon Explorer Status Brainstorm

A portion of the June 29th, 2022 Carbon weekly sync was used as a brainstorming
session to identify both work that has been accomplished and work that remains
for the Carbon explorer tool. While a comprehensive snapshot in time, there
wasn’t an explicit agreement on how to utilize this document or keep it updated
going forward. This proposal is the logical next step in this work.

## Proposal

We propose creating and utilizing an AreWeYet for the explorer project. The
methodology has already been successful elsewhere and we already have the data
collected for the explorer project. Although this proposal narrowly targets
Carbon explorer, experience with this methodology may indicate value for its use
elsewhere.

## Details

In summary, we propose the creation of a new page on the carbon-lang Wiki called
“Are We Explorer Yet?”, population of this page with results of the explorer
status brainstorm, and the creation of issues for parts that are not complete.

The following section will be used as the basis for the “Are We Explorer Yet?”
page. Items that are specifically out of scope for this AreWeYet are C++ interop
(likely never a target for explorer), metaprogramming, parallel programming, and
coroutines. The last three are likely substantial enough to merit their own
AreWeYet pages.

### Are We Explorer Yet?

-   ❌ Structured programming
    -   ✅ While loops
    -   ✅ Variable declarations
    -   ❌ Variable initialization tracking
    -   ❌ Returned var
    -   ❌ Variadics
-   ❌ User defined types
    -   ✅ structs
    -   ✅ classes
    -   ❌ choice
-   ✅ Alias system
-   ❌ OO programming
    -   ❌ Inheritance
    -   ❌ Parameterized class methods w/ inheritance
    -   ❌ Destructors
    -   ✅ Methods
    -   ✅ Static functions / Class functions
-   ❌ Generic programming
    -   ✅ Generic classes
    -   ❌ Generic methods
    -   ✅ Generic functions
    -   ✅ Interfaces
    -   ✅ Generic Interfaces
    -   ✅ Impls
    -   ✅ Generic Impls
    -   ❌ Impl specialization
    -   ❌ Templates
-   ❌ Operator overloading
    -   ✅ ==
    -   ❌ /=
    -   ❌ Other operators
    -   ❌ Constraints
    -   ✅ Implicit “as”
-   ❌ Error handling
-   ✅ Prelude
    -   ✅ Print function
-   ❌ Types
    -   ✅ i32
    -   ❌ Other integral types
    -   ❌ Integral types as library types instead of native
    -   ✅ Tuples
    -   ✅ Pointer
    -   ✅ Functions
    -   ✅ Bool
    -   ✅ String
    -   ❌ Floating point types
    -   ❌ Raw string literals
-   ❌ Code organization
    -   ❌ Mixins
    -   ❌ Imports
    -   ❌ Separate packages
    -   ❌ Modules
    -   ❌ Namespaces

## Alternatives considered

The default alternative is status quo where carbon explorer progress isn’t being
tracked from a high-level. Due to the problems outlined in the problems section,
this seems like an undesirable option.

Another alternative is to use github issue tags and come up with a custom search
to see the outstanding issues. This has the benefit of being fully automated. On
the other hand, it would be difficult to get a hierarchical view which is useful
to being able to grasp the big picture.
