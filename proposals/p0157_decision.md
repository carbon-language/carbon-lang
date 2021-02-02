# Decision for: Design direction for sum types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2021-02-02

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [noncombatant](https://github.com/noncombatant)
-   [tituswinters](https://github.com/tituswinters)

## Open questions

None

## Rationale

This proposal gives us solid ground on which to continue developing features
that rely on sum types and pattern matching thereof. The approach taken seems
plausible, and while there's a good chance that we will revise it significantly
before we finish our first pass over the complete Carbon design (for example, by
switching to pattern matching proxies or by supporting mutation of matched
elements), we don't think it will anchor us too much on this one particular
direction.

With regard to Carbon's goals:

-   Performance: while the approach taken herein potentially has some
    performance cost for a common operation that is likely to appear in
    performance-critical code (requiring an indirect call and the generation of
    continuations for user-defined pattern matching), such cost should be
    practically removable by inlining. We'll need to take care to ensure this
    abstraction penalty is reliably removed in common cases, but this seems
    sufficiently feasible to be worth attempting.
-   Evolution: software evolution is supported by allowing user-defined pattern
    matching to specify (via the presence/absence of operator default) whether
    the set of patterns is intended to be extended in the future.
-   Ergonomics: custom pattern matching for user-defined types promotes language
    consistency and removes boilerplate

Note: At the decision meeting, it was stated that geoffromer will update the
proposal to add a rationale to address austern's questions about the layered
approach.
