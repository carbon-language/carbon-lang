# Change operator precedence

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4075)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [`as` and `where` could be peers of `if`...`then`...`else`](#as-and-where-could-be-peers-of-ifthenelse)
    -   [Make `T as I where R` mean `T as (I where R)`](#make-t-as-i-where-r-mean-t-as-i-where-r)
    -   [Make fewer changes](#make-fewer-changes)
    -   [Different `where` syntax](#different-where-syntax)

<!-- tocstop -->

## Abstract

Update the operator precedence to achieve a few goals:

-   Form operators into groups which behave similarly
-   Make the group of operators ("top-level operators") that capture everything
    to the right, like `if`...`then`...`else`, behave similarly to the left, so
    that rearranging expressions won't change how they group.
-   Add the `where` operator, used to specify constraints on facet types, to the
    precedence chart, to define how it interacts with other operators.
-   Make the operator precedence diagram prettier, so that it eventually can be
    made into a poster that Carbon programmers can hang on their walls.

## Problem

The `where` operator is particularly tricky:

-   It is used in an `impl` declaration to specify the values of associated
    constants (such as associated types). In that context, `impl T as I where R`
    is interpreted conceptually as `impl T as (I where R)`. It would be nice if
    `T as I where R` would mean the same thing in other contexts. If not, we'd
    rather it to be invalid rather than meaning `(T as I) where R`. That is,
    That is, considered in isolation, we would prefer `T as (I where R)` over
    invalid over `(T as I) where R`.
-   The `where` operator will frequently be used with the binary `&` operator,
    since that is how facet types are combined. It is desirable that
    `I & J where R` be interpreted as `(I & J) where R`. If not, we'd rather it
    be invalid than be interpreted as `I & (J where R)`. This usage of `&` with
    `where` is expected to be more common than combining `where` and `as`
    outside of an `impl` declaration.
-   The "restriction" on the right side of a `where` uses operators that mean
    something else in an expression context: `and`, `==`, `=`. We would like to
    minimize the confusion when both kinds of uses of those operators appear in
    the same expression.

These goals are in conflict with the current precedence partial order.

## Background

The initial operator precedence approach, including using a partial precedence
ordering instead of a total ordering as found in most languages, was established
by [propsoal #555](https://github.com/carbon-language/carbon-lang/pull/555).
[PR #1070](https://github.com/carbon-language/carbon-lang/pull/1070) established
the current precedence chart, which has been incrementally added to since then.

## Proposal

We are making a number of changes:

-   `x as T` is no longer allowed on either side of a comparison operator, or
    the short-circuiting operators `and` & `or`.
-   `x where R` is a peer to `as`, but its arguments can be binary operators
    (like `&`). This matches the comparison operators, which are either illegal
    or reinterpreted as an argument to `where`.
-   The type constructors `T*` and `const T` are no longer separate from the
    other unary operators, and can now be the argument of any binary operator.

## Details

Please see the new precedence diagram in
[docs/design/expressions/README.md](/docs/design/expressions/README.md).

## Rationale

Precedence is about
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
We don't want to require parentheses too often since that makes the code harder
to write, and if it goes too far even reading becomes difficult. However, we do
want parentheses to mark code that would otherwise be misinterpreted. This is a
balancing act we expect to have to refine with experience.

## Alternatives considered

### `as` and `where` could be peers of `if`...`then`...`else`

We considered making all the "top-level" operators act the same for precedence,
but we expect users to want to use `as` to force the two branches of an
`if`...`then`...`else` expression to a common type often enough, and we didn't
expect the result of doing that to be confusing to read.

### Make `T as I where R` mean `T as (I where R)`

We wanted to make `T as I where R` mean the same facet type as that same
sequence of tokens in an `impl` declaration. However, this was in conflict with
the arguments to `where` being the same as the arguments to comparison
operators. We didn't want to allow an expression mixing binary operators with
`as` since we expected users to expect that to mean performing the operation
with that casted-to type. For example, `x + y as i64` would mean
`(x + y) as i64`, which would perform the addition and only then cast to `i64`,
which is probably not what would be intended by that expression. We thought it
better to make `x + y as i64` illegal to force users to use parentheses, even if
that meant also using parentheses with `T as I where R` in an expression
context.

### Make fewer changes

We considered making fewer changes to precedence, but that lead to an operator
precedence diagram with crossing edges (it was
[non-planar](https://en.wikipedia.org/wiki/Planar_graph)). This was felt to be a
sign that the graph was too complex, making it harder for humans to understand
and remember. It was suggested that developers using Carbon may want to have the
precedence graph posted for reference, and a planar graph would make a
more-appealing poster.

This was
[discussed in open discussion on 2024-06-20](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p524bg7cnd32).

### Different `where` syntax

We considered other ways of marking the end of a `where` restriction expression,
such as requiring parens `(`...`)` (either around the argument or the whole
`where` expression) or having a keyword at the end. We ultimately decided none
of those options were satisfactory since they added noise that reduced clarity,
and decided to go with a greedy approach ("all the way to the right") instead.

This was discussed in
[open discussion on 2024-06-13](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p46elxrmhh8x)
