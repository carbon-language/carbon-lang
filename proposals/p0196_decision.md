# Decision for: Language-level safety strategy

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
-   [tituswinters](https://github.com/tituswinters)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [noncombatant](https://github.com/noncombatant)

## Open questions

### Does the core team believe that we should put a cap on how much performance should be sacrificed for safety, putting more emphasis on probabilistic methods that would allow more attacks through?

There is no cap at this point. Where possible, static checking should be used.
In practice, there will be some who will not use the safest option until the
cost gets low enough.

## Rationale

Most of Carbon's goals can be addressed in a somewhat piecemeal fashion. For
example, there's probably no need for our designs for generics and for sum types
to coordinate how they address performance or readability. Safety, on the other
hand, is much more cross-cutting, and so it's important for us to approach it
consistently across the whole language. This proposal gives us a common
vocabulary for discussing safety, establishes some well-motivated common
principles, and provides an overall strategy based on those principles, all of
which will be essential to achieving that consistency.

This proposal gives a solid basis for thinking about safety in future proposals.
The pragmatic choice to focus on the security aspects of safety seems
well-aligned with Carbon's goals. In particular, a more idealistic approach to
safety, in which every language construct has bounded behavior, would be likely
to result in safety being prioritized over performance. By instead considering
safety largely from the perspective of security vulnerabilities, and accepting
that there will be cases where the pragmatic choice is hardening rather than
static or dynamic checks, we can focus on delivering practical safety without
being overly distracted from other goals.

It is critical that Carbon use build modes to enable writing
performance-optimized code (much like C++ today) that can still be built and
deployed at reasonable engineering cost with strong guarantees around memory
safety for the purposes of security. I think this proposal provides that
foundation.

Note: The decision by the core team included a request to clarify the wording on
build modes as described in
[chandlerc's post](https://forums.carbon-lang.dev/t/request-for-decision-language-level-safety-strategy/196/6)
on the decision thread.
