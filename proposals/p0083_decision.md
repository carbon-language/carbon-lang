# Decision for: An incomplete, early, and in-progress overview of the language design

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-07-28

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

There were no open questions.

## Rationale

This proposal provides some much-needed context for a lot of our discussions and
deliberations about Carbon. Having a skeleton in place for the language design
is an important step forward. We can't make language design decisions in a
vacuum -- we need a "blurry outline" of the big picture before we can start
filling in the details of any particular area. We need to establish a common
frame of reference regarding the overall shape of the language, so that we can
parallelize the in-depth design work.

That means that, by necessity, this proposal suggests lots of concrete design
decisions that have not yet had sufficient analysis for us to affirm them. There
is a shared understanding that we are not committing to any of those decisions,
only to the broad picture painted by the combination of those decisions, and
that all such decisions need to be revisited by another proposal before we
consider them to be agreed on. There is a substantial risk of anchoring how we
think about Carbon -- we’ll just have to do our best to avoid taking this doc as
a given when evaluating subsequent proposals. Those propopsals must justify a
direction that agrees with this doc as much as one that does not agree with it.

This doc sets the stage for increasingly incremental steps towards a complete
design. Establishing a structure for the design is especially helpful as it will
show how things do or do not connect across the language.

Adopting this largely work-in-progress overview in order to see the structure of
things, while still needing to resolve the specifics in each area, will directly
help reinforce our goal of language evolution over time. This will help us learn
how to effectively iterate, and how to compensate and overcome the risks of
anchoring, change aversion, and other challenges.
