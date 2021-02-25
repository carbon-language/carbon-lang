# Decision for: Basic Syntax #162

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-11-10

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [noncombatant](https://github.com/noncombatant)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [tituswinters](https://github.com/tituswinters)

## Rationale

Using code to validate our specification is a really promising direction, and
this proposal seems like a good starting point. A reference implementation
that's simple enough to be part of the design iteration process should help us
move faster, by quickly uncovering the places where our specifications are
ambiguous, syntactically or semantically unsound, or don't give the behavior we
expect. In other words, it will help us keep ourselves honest, even at the
proposal stage, which will help us avoid wasting time and effort implementing
designs that turn out to be unworkable.

This can be considered as sort of a counterpart to
[In-progress design overview #83](p0083.md), in that the design specifics are
being approved in order to bootstrap the specification process. We aren't
necessarily adopting the specific syntax and semantics expressed by this
proposal, and those choices will need to be presented and justified from scratch
by future proposals.

This decision is deferring the implementation to code review. The specific
tooling used to implement the syntax checker, such as Bison, is a detail which
may be changed, now or later, without requiring a proposal for core team review.
