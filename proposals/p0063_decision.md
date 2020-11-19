# Decision for: Criteria for Carbon to go public

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-11-03

Affirming:

-   [chandlerc](https://github.com/chandlerc)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [noncombatant](https://github.com/noncombatant)
-   [tituswinters](https://github.com/tituswinters)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [austern](https://github.com/austern)
-   [geoffromer](https://github.com/geoffromer)

## Open questions

None.

## Rationale

"An open, inclusive process for Carbon changes" requires us to make the decision
to go public in a clear and unsurprising way, with criteria that are written
down.

Going public too early introduces risks to the long-term evolution and
maintenance of the language by increasing the costs of the community members
developing it.

Ensuring that Carbon meets its functional goals, especially that of
interoperability and migration, will inherently require large scale
experimentation that is infeasible to do without becoming public at some point.

The core team is aligned on the core policy decisions, which are:

-   The decision to go public will go through the usual proposal process.
-   The structure of the proposal gives the things that we are looking for prior
    to going public.
-   We expect to delay announcing to the extent we can.
-   We do not expect that to be so late that everything is done and we are ready
    to ship. That is, we are not going to wait until version 1.0 is ready.
    -   We don't expect this to be so late that Carbon is no longer an
        experiment.

Changes to the text and wording that align with these items should be submitted
as code reviews. The core team members chandlerc and zygloid will both approve
each code review.

An example change that would be covered by a code review: there will be no
automatic going public just because the criteria are met--it will be a decision
of the core team.
