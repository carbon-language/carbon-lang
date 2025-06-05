# Keep design documents current

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5606)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Require documentation updates as part of the proposal](#require-documentation-updates-as-part-of-the-proposal)
    -   [One TODO per document](#one-todo-per-document)
    -   [Link to the PR instead of the proposal document](#link-to-the-pr-instead-of-the-proposal-document)

<!-- tocstop -->

## Abstract

Require language design proposals to either update the design documents to
reflect the proposed changes, or add "TODO" comments to mark where those changes
will be needed, with links back to the proposal. This is intended to ensure that
the design documentation accurately informs readers about the current language
design, without excessively burdening the proposal process.

## Problem

The current evolution process allows us to adopt language design changes while
deferring the corresponding changes in `/docs/design`. We have no process for
ensuring that those follow-up changes actually happen, and in practice some
adopted proposals have gone for well over a year without corresponding design
document changes.

This problem is not limited to proposals for new features, but also applies to
proposals to change existing, documented features. As a result, the design
documents are not merely incomplete, but in some cases actually misleading,
which has led to miscommunication and wasted effort within the Carbon team.

## Proposal

This proposal modifies our evolution process to require proposals to either
implement the corresponding changes to the design documents, or mark the places
that will need to be changed with "TODO" comments that point back to the
proposal. The presence of those comments should help ensure that readers are not
misled by the design documentation, and the links to the proposals should help
ensure readers can discover the actual design for a given feature with
reasonable effort.

The proposal PR also adds "TODO" comments to `/docs/design`, both as an
illustration of what this policy asks for, and because the policy needs to be
applied retroactively in order to actually solve the problem. However, these
changes are necessarily best-effort, because it wasn't feasible for me to fully
evaluate every proposal against the current state of the docs. In particular, I
didn't look at proposals numbered below 2000, and I assumed that proposals fully
updated `/docs/design` if they touched it at all. In addition, I did not add
"TODO" comments for the terminology changes in [p2964](/proposals/p2964.md),
because they would be pervasive, and probably add little value for the reader.
Instead, I filed issue
[#5599](https://github.com/carbon-language/carbon-lang/issues/5599) to track the
task of making those changes.

## Details

See the changes in the proposal PR.

## Rationale

This proposal will help advance our goal of
[promoting a healthy and vibrant community with an inclusive, welcoming, and pragmatic culture](/docs/project/goals.md#community-and-culture):
in order to effectively participate in extending, implementing, or evaluating
the design of Carbon, people need to be able to find accurate information about
that design (and avoid inaccurate information) with reasonable effort.

## Alternatives considered

### Require documentation updates as part of the proposal

We could require all language design proposals to implement the corresponding
changes in `/docs/design`. This would more thoroughly address the risk of
readers being misled, because you might mistakenly read documentation that's
marked with a "TODO" comment, but you can't mistakenly read documentation that
doesn't exist anymore.

This would also push proposals to be more concrete, detailed, and fully
integrated with the rest of the language. However, that's a double-edged sword:
it could help us identify problems with a proposal before it's adopted, but it
could also close off the option of deferring those problems to future work in
order to make incremental progress. More mundanely, it would also increase both
the up-front cost of creating a proposal, and the cost of iteratively changing
it during review.

The loss of agility from those factors is likely to outweigh the somewhat
tenuous and speculative benefits of this approach, at least at this early stage
of Carbon's development.

### One TODO per document

This policy intentionally encourages fairly granular TODO comments attached to
the specific passages where changes are needed (see especially the proposed
changes in
[`details.md`](https://github.com/carbon-language/carbon-lang/pull/5606/files#diff-b84aebe5ad22a2be2b4c222cf68fd93981ebcd2451bafb56ed5fe46ec186a3c8)).
We could instead encourage having a single TODO comment per document, in order
to reduce the burden on proposal authors.

However, this would have several related drawbacks for readers of the
documentation:

-   It increases the risk that the reader will overlook the TODO comment
    altogether, because it may be quite far away from the passage they're
    reading.
-   It increases the risk of false positives, where passages that remain valid
    are caught up in the scope of a file-level TODO comment.
-   It may force the TODO comment to be less specific about how the proposal
    changes the content of the document. This compounds the previous problem,
    because it means the reader has to do more work to distinguish true from
    false positives.

Together, these drawbacks could severely undermine the purpose of this proposal.
By contrast, the burden of granular TODOs on proposal authors seems relatively
marginal. For example, I was able to add TODO comments for 8 proposals in a few
hours, despite in most cases being quite unfamiliar with their contents.

### Link to the PR instead of the proposal document

This policy specifies that the TODOs should include a link to the proposal
document, but we could instead link to the proposal PR. This would be more
consistent with existing documentation, which rarely links to proposal documents
(among other things, this makes it easier to find all references to a given
proposal). The PR also surfaces some information not available in the proposal
document, such as the discussion associated with a proposal, and the other
places where TODOs were added. Finally, the PR UI gives you access to a rendered
view of the proposal document (accessible through "View file" in the drop-down
menu for the file), in which links to other documents take you to the versions
that existed as of when the proposal was committed, which may help clarify the
historical context of the proposal if those documents have changed in the
meantime.

However, the TODO links serve a very different purpose than other proposal
links: when we link to a proposal in a "References" or "Alternatives considered"
section of a document, we're providing interested readers with supplementary
information about the rationale and history of the design being documented, and
for those purposes the additional information in the PR is directly relevant.
Here, by contrast, the proposal is acting as the sole documentation for _the
design itself_, and for that purpose all the relevant information is in the
proposal document itself.

Finally, in the event that the reader wants additional information from the PR,
every proposal document has a prominent PR link at the top, so the additional
burden on those readers is a single click. By contrast, the link from a proposal
PR to the corresponding rendered document is hidden in a drop-down menu
somewhere in the middle of the PR's "files changed" page.
