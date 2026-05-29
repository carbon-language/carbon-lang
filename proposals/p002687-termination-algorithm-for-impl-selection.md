# Termination algorithm for impl selection

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2687)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Non-type arguments](#non-type-arguments)
    -   [Proof of termination](#proof-of-termination)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Measure complexity using type tree depth](#measure-complexity-using-type-tree-depth)
    -   [Consider each type parameter in an `impl` declaration separately](#consider-each-type-parameter-in-an-impl-declaration-separately)
    -   [Consider types in the interface being implemented as distinct](#consider-types-in-the-interface-being-implemented-as-distinct)
    -   [Require some count to decrease](#require-some-count-to-decrease)
    -   [Require non-type values to stay the same](#require-non-type-values-to-stay-the-same)

<!-- tocstop -->

## Abstract

This proposal replaces the termination algorithm for `impl` selection. The
previous algorithm relied on a recursion limit, which is counter to
[our goal for predictability](/docs/design/generics/goals.md#predictability).

The replacement is to terminate if any `impl` lookup performed while considering
an `impl` declaration depends transitively on the same `impl` declaration with a
"strict superset" of the types in the query.

## Problem

Consider this `impl` declaration:

```
interface I;
impl forall [T:! type where Optional(.Self) is I] T as I;
```

A type like `i32` is a valid value of `T`, and so implements `I`, if
`Optional(i32)` implements `I`. This `impl` declaration could also possibly be
used to give an implementation of `Optional(i32)` for interface `I`, but only if
there was an implementation of `Optional(Optional(i32))` for interface `I`. The
job of the termination rule is to report an error instead of being caught in an
infinite loop in this situation.

Ideally, a termination rule would identify the loop in a minimal way. This has a
few benefits, including reducing compile times and making error messages as
short and understandable as possible. One downside of the original recursion
limit rule is its tendency to only detect a problem after the loop had been
repeated many times. This problem is worse if the recursion limit is large.

Another concern with using a recursion limit is that refactorings that would
otherwise be legal can increase the depth of recursion, causing spurious
failures. The workaround for this and other spurious failures is to increase the
recursion limit. This makes the other problems with using a recursion limit
worse.

Note that determining whether a particular set of `impl` declarations terminates
is
[equivalent to the halting problem](https://sdleffler.github.io/RustTypeSystemTuringComplete/)
(content warning: contains many instances of an obscene word as part of a
programming language name), and so is undecidable in general. So any termination
rule will have some false positives or spurious failures, where it reports an
error even though it would in fact complete if allowed to continue running. We
would like a criteria that correctly classifies the examples that arise in
practice.

## Background

The first termination rule was introduced in proposal
[#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920),
following Rust and C++. The problems with using a recursion limit were
[recognized at the time that proposal was written](https://github.com/carbon-language/carbon-lang/blob/f282bca20e41e2f8dc05881d9d6b38213d6c6c87/docs/design/generics/details.md#termination-rule),
but no alternative was known.

Alternatives termination rules have since been discussed:

-   in open discussion on
    [2022-04-13](https://docs.google.com/document/d/1tEt4iM6vfcY0O0DG0uOEMIbaXcZXlNREc2ChNiEtn_w/edit#heading=h.cja3fkwzv9tr),
    prompted by a question on
    [#1088: Generic details 10: interface-implemented requirements](https://github.com/carbon-language/carbon-lang/pull/1088);
    and
-   in issue
    [#2458: Infinite recursion during impl selection](https://github.com/carbon-language/carbon-lang/issues/2458),
    which includes summaries of discussions including those on
    [2023-02-07](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ#heading=h.9u2u6078figt).

PR
[#2602: Implement the termination algorithm for impl selection described in #2458](https://github.com/carbon-language/carbon-lang/pull/2602)
implements the termination rule of this proposal in Explorer.

## Proposal

We replace the termination criteria with a rule that the types in the `impl`
query must never get strictly more complicated when considering the same `impl`
declaration again. The way we measure the complexity of a set of types is by
counting how many of each base type appears. A base type is the name of a type
without its parameters. For example, the base types in this query
`Pair(Optional(i32), bool) impls AddWith(Optional(i32))` are:

-   `Pair`
-   `Optional` twice
-   `i32` twice
-   `bool`
-   `AddWith`

A query is strictly more complicated if at least one count increases, and no
count decreases. So `Optional(Optional(i32))` is strictly more complicated than
`Optional(i32)` but not strictly more complicated than `Optional(bool)`.

This rule, when combined with
[the acyclic rule](/docs/design/generics/details.md#acyclic-rule) that a query
can't repeat exactly, [guarantees termination](#proof-of-termination). This rule
is expected to identify a problematic sequence of `impl` declaration
instantiations in a way that is easier for the user to understand. Consider the
example from before,

```
interface I;
impl forall [T:! type where Optional(.Self) is I] T as I;
```

This `impl` declaration matches the query `i32 impls I` as long as
`Optional(i32) impls I`. That is a strictly more complicated query, though,
since it contains all the base types of the starting query (`i32` and `I`), plus
one more (`Optional`). As a result, an error can be given after one step, rather
than after hitting a large recursion limit. And that error can state explicitly
what went wrong: we went from a query with no `Optional` to one with one,
without anything else decreasing.

Note this only triggers a failure when the same `impl` declaration is considered
with the strictly more complicated query. For example, if the declaration is not
considered since there is a more specialized `impl` declaration that is
preferred by the
[type-structure overlap rule](/docs/design/generics/details.md#overlap-rule), as
in:

```
impl forall [T:! type where Optional(.Self) is I] T as I;
impl Optional(bool) as I;

// OK, because we never consider the first `impl`
// declaration when looking for `Optional(bool) impls I`.
let U:! I = bool;

// Error: cycle with `i32 impls I` depending on
// `Optional(i32) impls I`, using the same `impl`
// declaration, as before.
let V:! I = i32;
```

The rule is also robust in the face of refactoring:

-   It does not depend on the specifics of how an `impl` declaration is
    parameterized, only on the query.
-   It does not depend on the length of the chain of queries.
-   It does not depend on a measure of type-expression complexity, like depth.

## Details

### Non-type arguments

For non-type arguments we have to expand beyond base types to consider other
kinds of keys. These other keys are in a separate namespace from base types.

-   Values with an integral type use the name of the type as the key and the
    absolute value as a count. This means integer arguments are considered more
    complicated if they increase in absolute value. For example, if the values
    `2` and `-3` are used as arguments to parameters with type `i32`, then the
    `i32` key will have count `5`.
-   Every option of a choice type is its own key, counting how many times a
    value using that option occurs. Any parameters to the option are recorded as
    separate keys. For example, the `Optional(i32)` value of `.Some(7)` is
    recorded as keys `.Some` (with a count of `1`) and `i32` (with a count of
    `7`).
-   Yet another namespace of keys is used to track counts of variadic arguments,
    under the base type. This is to defend against having a variadic type `V`
    that takes any number of `i32` arguments, with an infinite set of distinct
    instantiations: `V(0)`, `V(0, 0)`, `V(0, 0, 0)`, ...
    -   A `tuple` key in this namespace is used to track the total number of
        components of tuple values. The values of those elements will be tracked
        using their own keys.

Non-type argument values not covered by these cases are deleted from the query
entirely for purposes of the termination algorithm. This requires that two
queries that only differ by non-type arguments are considered identical and
therefore are rejected by the acyclic rule. Otherwise, we could construct an
infinite family of non-type argument values that could be used to avoid
termination.

### Proof of termination

Let's call a (finite or infinite) sequence of type expressions "good" if no
later element is strictly more complex than an earlier element, and no type
expression is repeated. We would like to prove that any good sequence of type
expressions with a finite set of keys is finite.

We can restrict to good sequences that don't repeat any multiset of keys, since
there are only a finite number of types with a given multiset of keys. Proof: If
none of the types have a variadic parameter list, then there is at most one type
for every distinct permutation of base types. If some types are variadic, then
we can get a conservative finite upper bound by multiplying the number of
distinct permutations by the number of different possible arity combinations.
The number of arity combinations is finite since, ignoring non-type arguments,
the total arity must equal the number of base types in the type minus 1.

The proof of termination is by induction on the number `N` of distinct keys.

-   If `N == 1`, then types map to a multiset of a single key, which can be
    represented by the count of the number of times that key appears. That
    number must be non-negative and decreasing in the sequence, and so the
    length of the sequence is bounded by the value of the first element. So good
    sequences with `N == 1` must be finite.
-   Assuming that good sequences with `N` distinct keys must be finite, consider
    a good sequence with `N+1` distinct keys. Its first element will be
    represented by a non-negative integer `(N+1)`-tuple, `(i_0, i_1, ..., i_N)`.
    Every element after that will be in at least one of the
    `i_0 + i_1 + ... + i_N` hyperplanes (co-dimension 1) given by these
    equations:
    -   `x_0 = 0`, `x_0 = 1`, ..., `x_0 = i_0 - 1` (`i_0` different equations,
        each defining a separate hyperplane)
    -   `x_1 = 0`, `x_1 = 1`, ..., `x_1 = i_1 - 1`
    -   ...
    -   `x_N = 0`, `x_N = 1`, ..., `x_N = i_N - 1`
-   Any point not in one of those hyperplanes has components all >= the first
    element, and so can't be in the sequence if it is good.
-   The restriction of the sequence to the subsequence in each of those
    hyperplanes is finite, by the induction hypothesis.
-   The sequence visits points in this finite union of finite sets without
    repetition, and so must be finite.
-   Conclusion: Any good sequence with `N+1` distinct keys is finite, completing
    the induction.

This bound given by this construction is not completely tight, since there is
overlap between the hyperplanes. It is tight once that overlap is taken into
account, though. We can construct sequences that reach the upper bound by
visiting the points in the union of the hyperplanes in descending order of their
L1-norm (sum of the components).

Note: The text of this argument was derived from comments on
[issue #2458: Infinite recursion during impl selection](https://github.com/carbon-language/carbon-lang/issues/2458).

## Rationale

This proposal advances these [Carbon goals](/docs/project/goals.md):

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    by improving the quality of diagnostics.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    since we've chosen alternatives that avoid introducing failures as the
    result of refactorings, particularly those outside the files changed in the
    refactoring.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    by having a relatively simple language rule, that is predictable when
    authoring code, and allows independent modules to compose without triggering
    errors.

## Alternatives considered

### Measure complexity using type tree depth

We
[considered](https://github.com/carbon-language/carbon-lang/issues/2458#issuecomment-1371412985)
a rule which would ensure termination by forbidding the depth of the type tree
in the query from increasing. This depth could either be measured in the query
or in the values of types used to parameterize `impl` declaration. Either way,
this raises a concern that otherwise safe refactorings might trigger spurious
termination errors. Specifically, refactorings that replace a type, like
`String`, with an alias to a parameterized type, like `BasicString(Char8)`,
could change the tree depths in `impl` declarations in files that were not part
of the refactoring.

### Consider each type parameter in an `impl` declaration separately

Instead of measuring the complexity of the `impl` query as a whole, we
considered measuring the complexity of the argument values of parameters in an
individual `impl` declaration. The advantage of this would be fewer spurious
failures due to the termination rule.

We decided against it because it is a more complex rule and it is sensitive to
the specifics of how `impl` declarations are parameterized. This raises concerns
about refactorings introducing termination rule failures. We did not want to
incorporate this change without evidence that those spurious failures would be a
problem in practice.

### Consider types in the interface being implemented as distinct

Instead of measuring the complexity of the entire `impl` query together, we
could consider keys in the type and interface parts of the query to be in
distinct namespaces. This would reduce the spurious failures due to the
termination rule, but not as much as the previous alternative. It avoids the
problem of the previous alternative, since it is not sensitive to the specifics
of how `impl` declarations are parameterized.

We are not choosing this alternative now since it is a more complicated rule to
explain. But we would consider this alternative in the future if we find it
beneficial in practice to support the additional cases this rule permits.

### Require some count to decrease

We considered a rule that would forbid repeating the multiset of types. This
would simplify the termination argument. However, we thought it important to
support `impl` declarations that effectively shuffled the terms around into some
canonical form, as in this example:

```
impl forall [T:! I(Optional(.Self))] Optional(T) as I(T);
```

Here, `Optional(bool) impls I(bool)` if `bool impls I(Optional(bool))`. This
rule can only be applied a finite number of times, and is something we imagine
might arise naturally, so it seemed good to support.

### Require non-type values to stay the same

We considered different handling for
[non-type argument values](#non-type-arguments) that did not have an integral or
choice type. The alternative rule required the value to stay constant. This led
to
[a number of edge cases](https://github.com/carbon-language/carbon-lang/pull/2687#discussion_r1151028867)
to consider, like how to identify the same argument when the type constructor
may be called a different number of times. The rule we chose to use instead has
the advantages of being simpler and also accepting more cases. With the current
rule, the value of those arguments may change freely, they just don't create
different type expressions for purposes of detecting termination.
