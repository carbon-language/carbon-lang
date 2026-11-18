# Checked and template generic terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2138)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Status quo: "generic" means "checked"](#status-quo-generic-means-checked)
    -   [Use the name "templates" instead of "template generics"](#use-the-name-templates-instead-of-template-generics)
    -   [Use the name "unchecked generics" instead of "template generics"](#use-the-name-unchecked-generics-instead-of-template-generics)

<!-- tocstop -->

## Abstract

Change terminology from "generics" and "templates" to "checked generics" and
"template generics". Afterwards, "generics" will be an umbrella term that
include templates.

## Problem

The C++ community does "generic programming" using templates in C++, and so
thinks of templates as a mechanism for implementing generics. In Carbon,
templates and other generic features are more similar than they are different,
so it is worthwhile to have an easy way to talk about things that apply to both.

## Proposal

Change terminology from "generics" and "templates" to "checked generics" and
"template generics". Afterwards, "generics" will be an umbrella term that
include templates.

Similarly, for parameters, use "checked generic parameter" and "template generic
parameter", with the umbrella term "generic parameter" describing both kinds.
The word "generic" can be omitted from these terms if it's clear from context,
which will usually be the case for template parameters.

Some existing design docs give examples of the difference before and after this
proposal:

-   The [generics overview](/docs/design/generics/overview.md), for example,
    currently demonstrates the current approach of using "generics" and
    "templates".
-   The
    [generics section of the design overview](/docs/design/README.md#generics)
    demonstrates the use of the proposed new terminology.

## Rationale

Using the terminology expected by the community supports these goals:

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    since our documentation governing the interpretation of that code will be
    more easily understood and with greater accuracy.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    since there will be a smaller terminology gap between Carbon and C++.

## Alternatives considered

### Status quo: "generic" means "checked"

Use "generic" for checked generics, "template" for template generics, and
"template or generic" when we mean both.

Advantages:

-   The terms "generic" and "template" are more concise than "checked generic"
    and "template generic".
-   The term "template" is more familiar than "template generic" to people
    coming from C++.

Disadvantages:

-   Does not provide a unifying term for templates and generics, leading to our
    needing to talk about both in various places, given that they are very
    similar concepts in Carbon.
-   Does not acknowledge that templates are used for generic programming and as
    such are a kind of generic.
-   Carbon's template generics aren't exactly the same as C++'s templates, so
    this may give the wrong intuition for people coming from C++ in some cases.

### Use the name "templates" instead of "template generics"

Use "checked generic" for checked generics, "template" for template generics,
and "generic" when we mean both.

Advantages:

-   More concise term than in this proposal.
-   More familiar to people coming from C++.
-   People will likely use this term anyway rather than saying "template
    generics".

Disadvantages:

-   Does not parallel the term "checked generics".
-   Loses the implication that a template generic is a kind of generic.
-   Carbon's template generics aren't exactly the same as C++'s templates, so
    this may give the wrong intuition for people coming from C++ in some cases.

### Use the name "unchecked generics" instead of "template generics"

Use "checked generic" for checked generics, "unchecked generic" for template
generics, and "generic" when we mean both.

Advantages:

-   Creates a better parallel between "checked generics" and "unchecked
    generics".
-   Describes the semantics rather than the implementation strategy, especially
    given that templating / monomorphization is also the implementation strategy
    we intend to use for unchecked generics.

Disadvantages:

-   Unfamiliar to people coming from C++.
-   Would either need to rename the `template` keyword or pick new non-keyword
    syntax for it, or have a mismatch between keywords and terminology.
-   Template generics still involve checking, it just can't complete until later
    when the argument values are known.

This was suggested in
[#1443](https://github.com/carbon-language/carbon-lang/pull/1443).
