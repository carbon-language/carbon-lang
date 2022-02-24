# Add idiomatic code performance and developer-facing docs to goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/120)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Justification](#justification)
-   [Alternatives considered](#alternatives-considered)
    -   [Use a principle to address performance of idiomatic code](#use-a-principle-to-address-performance-of-idiomatic-code)
-   [Rationale](#rationale)

<!-- tocstop -->

## Problem

[Issue #106](https://github.com/carbon-language/carbon-lang/issues/106) raises a
few small issues with the goals doc as approved. Some are small, but in
particular I'll emphasize the question:

> Do we want to say anything about "reasonably fast by default" or "favors
> constructs that can be compiled to efficient code" or something like that?

I think this is something that we clearly want for performance, and should be
laid out.

Additionally, while considering justification for changes in
[PR 80](https://github.com/carbon-language/carbon-lang/pull/80), I noted there
is no explicit goal to provide developer-facing documentation. I believe this is
an intended part of the community goals, and could be inferred from current
ecosystem text, but may be better if explicit.

## Proposal

Add paragraphs to address performance of idiomatic code and developer-facing
documentation, as well as making other small fixes.

## Justification

Performance of idiomatic code:

Under "Performance-critical software", we establish that it should be possible
to write high-performance code with Carbon. However, if taken strictly, we could
be saying something like "it's okay if idiomatic code is predictably slow, as
long as developers have tools to 'open up the hood'." That is not the intent,
and so addressing the case of routine code performance offers the reassurance
that Carbon will prioritize performance consistently, regardless of whether
performance tuning is done.

Developer-facing documentation:

"Language tools and ecosystem" addresses the specification and tooling
explicitly. However, developer-facing documentation is also part of the
ecosystem, and part of supporting ramp-up training by new Carbon developers.
Such documentation should be an explicit project priority.

Other changes are incremental improvements to the goal text, and mainly
presented in this change for consistency of review.

## Alternatives considered

### Use a principle to address performance of idiomatic code

A principle is another way of addressing non-obvious conclusions based on goals.
However, performance of idiomatic code seems quick to state, and not worth
splitting off to a separate doc. I believe the cost-benefit favors keeping it in
the goals doc.

## Rationale

This addresses a number of rough edges and small missing pieces in the original
proposal, providing useful clarification. This follow up is expected as part of
the launch and iterate process we use to keep our velocity up.

User-facing documentation and speed-by-default should be first-order priorities
for Carbon.
