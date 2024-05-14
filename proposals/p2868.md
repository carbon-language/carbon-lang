# Allow overlap with a `final impl` if identical

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2868)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow interfaces with member functions to compare equal](#allow-interfaces-with-member-functions-to-compare-equal)
    -   [Mark associated constants as `final` instead of an `impl` declaration](#mark-associated-constants-as-final-instead-of-an-impl-declaration)
    -   [Allow type inequality constraints](#allow-type-inequality-constraints)
    -   [Prioritize a `final impl` over a more specific `impl` on the overlap](#prioritize-a-final-impl-over-a-more-specific-impl-on-the-overlap)

<!-- tocstop -->

## Abstract

Allow an `impl` to overlap with a `final impl` if they agree on the overlap.
Agreement is defined as all values comparing equal, and functions never
comparing equal. Implements the decision in question-for-leads
[issue #1077: find a way to permit impls of CommonTypeWith where the LHS and RHS type overlap](https://github.com/carbon-language/carbon-lang/issues/1077).

## Problem

The
[current design](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/expressions/if.md#same-type)
includes a `final impl` declaration for the `CommonTypeWith(T)` interface:

```carbon
final impl forall [T:! type]
    T as CommonTypeWith(T)
    where .Result = T {}
```

[Marking an `impl` declaration `final`](/docs/design/generics/details.md#final-impl-declarations),
prevents any overlapping implementation that would be considered more specific
by the [overlap rule](/docs/design/generics/details.md#overlap-rule). This
includes cases where the overlap is harmless, such as:

```carbon
impl forall [U:! type, T:! CommonTypeWith(U)]
    Vec(T) as CommonTypeWith(Vec(U))
    where .Result = Vec(T.Result) {}
```

This is an implementation we would like to define, along with a number of
similar cases. And this `impl` declaration doesn't actually conflict with the
previous `final impl` because the value of `Result`, the only member of the
`CommonTypeWith` interface, agrees where the two implementations overlap.

## Background

The `CommonTypeWith(T)` interface and `final impl` above were introduced in
[proposal #911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911).

[Proposal #983: Generics details 7: final impls](https://github.com/carbon-language/carbon-lang/pull/983)
introduced and defined the rules for `final impl` declarations.

The overlap rule was introduced in
[proposal #920](https://github.com/carbon-language/carbon-lang/pull/920).

There were a number of different resolutions for this problem considered in
question-for-leads
[issue #1077: find a way to permit impls of CommonTypeWith where the LHS and RHS type overlap](https://github.com/carbon-language/carbon-lang/issues/1077).
This proposal codifies the resolution of that issue.

## Proposal

We allow an `impl` declaration to overlap with a `final impl` declaration if it
agrees on the overlap. Since we do not require the compiler to compare the
definitions of functions, agreement is only possible for interfaces without any
function members. The details about how the intersection is computed and how
templated `impl` declarations are handled have been added to
[the section on `final` impl declarations in the generics design doc](/docs/design/generics/details.md#final-impl-declarations).

## Rationale

This proposal is intentionally keeping the language small by making a simple
rule that addresses the only identified use case and nothing more. This benefits

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)

by relying on Carbon's commitment to
[software and language evolution](/docs/project/goals.md#software-and-language-evolution)
to update our approach as needed, rather then trying to proactively address
concerns ahead of time.

## Alternatives considered

### Allow interfaces with member functions to compare equal

There are some specific cases where the compiler can verify that two functions
are the same without having to compare their definitions. For example, two
implementations that don't implement a function and instead inherit the default
from the interface could be considered equal. This creates an evolution hazard,
though, that copying the definition from the interface into the implementation
means that the interface could now compare not equal without any change in
behavior. For now, the simple rule that we don't compare functions at all is
sufficient for our identified [use case](#problem). This is something we would
reconsider given new use cases.

### Mark associated constants as `final` instead of an `impl` declaration

The biggest benefit from knowing that an `impl` declaration won't be specialized
is being able to use the values of the associated constants, particularly
associated types. Thus, it is natural to focus on associated constants, which
don't have the same concerns as functions with comparing for equality.

However, for the [motivating use case](#problem), we would still need this
proposal, just restricted to the associated constants that are declared `final`.
So we may still add this feature, if it is warranted by demand, but we did not
yet have that justification. This is essentially the same position as when this
feature was considered in
[proposal #983](/proposals/p0983.md#final-associated-constants-instead-of-final-impls).

### Allow type inequality constraints

Another approach would be to provide type inequality constraints so the more
specialized implementation could exclude the overlapping cases. This has some
downsides:

-   The more specialized implementation has to be aware of the `final` impl to
    specifically exclude it. This would add extra steps to the development
    process since this discovery is likely to occur as the result of a failed
    compile.
-   The more specialized implementation becomes more verbose, with extra
    conditions that don't add any value.
-   The current approach for establishing whether two types are equal doesn't in
    general provide a way to show two types are not equal in generic code.

### Prioritize a `final impl` over a more specific `impl` on the overlap

This was a possible fix, but was seen as a bigger change that we didn't yet have
justification for.
