# One way principle

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/829)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Alternatives considered](#alternatives-considered)
    -   [Provide multiple ways of doing a given thing](#provide-multiple-ways-of-doing-a-given-thing)

<!-- tocstop -->

## Problem

We have repeatedly run into cases where we could offer equivalent functionality
multiple ways.

## Proposal

Add a principle noting the preference is to only provide one way of doing
things.

## Alternatives considered

### Provide multiple ways of doing a given thing

Carbon could focus on providing a lower bar for overlapping functionality,
encouraging overlapping syntax. This could be considered as similar to Perl's
["There is more than one way to do it."](https://en.wikipedia.org/wiki/There%27s_more_than_one_way_to_do_it).
Overlapping syntax should still receive some scrutiny, but use-cases with small
marginal benefits might be considered sufficient to create divergent syntax.

For example:

-   For matching C++ legacy:
    -   We might include `for (;;)`.
    -   We might re-add support for optional braces on `if` and similar.
    -   We might support `class` and `struct` with their C++-equivalent default
        visibility rules.
    -   We might support both `and` and `&&`, `0xa` and `0XA`, etc.
-   We might make `;` optional, as there's precedent in modern languages to do
    so.
-   Both templates and generics might be embraced with feature parity such that
    both should be good solutions for generic programming problems, as much as
    possible.

It's worth noting Perl also has a related motto of "There is more than one way
to do it, but sometimes consistency is not a bad thing either." It's best to
avoid interpreting this alternative to the extreme: for example, this is not
encouraging providing all of `extensible`, `extendable`, and `extendible` as
equivalent keywords, as that is divergence with very minimal benefit
(particularly `extendable` versus `extendible`).

Advantages:

-   More likely that developers will find syntax that they like.
-   Can make it easier to migrate code because we can actively support a
    C++-like dialect in addition to a more Carbon-centric dialect.

Disadvantages:

-   Developers would either accept personal styles, or create more style guides.
    -   Either can be considered a language dialect, whether at the personal or
        organizational level.
-   Increases the difficulty of building syntax parsing and typo correction in
    the language, as there are more options that need to be corrected between,
    multiple of which may be valid in a given context.

We are declining this alternative because we value the language simplicity
provided by minimizing overlap of features.
