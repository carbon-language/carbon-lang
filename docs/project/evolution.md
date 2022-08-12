# Evolution and governance

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Proposals](#proposals)
    -   [Life of a proposal](#life-of-a-proposal)
    -   [Proposal roles](#proposal-roles)
        -   [Proposal authors](#proposal-authors)
        -   [Community](#community)
        -   [Active contributors](#active-contributors)
        -   [Carbon leads](#carbon-leads)
    -   [When to write a proposal](#when-to-write-a-proposal)
    -   [Proposal PRs](#proposal-prs)
        -   [What goes in the proposal document](#what-goes-in-the-proposal-document)
        -   [Open questions](#open-questions)
    -   [Review and RFC on proposal PRs](#review-and-rfc-on-proposal-prs)
    -   [Blocking issues](#blocking-issues)
        -   [Discussion on blocking issues](#discussion-on-blocking-issues)
-   [Governance structure](#governance-structure)
    -   [Carbon leads](#carbon-leads-1)
        -   [Subteams](#subteams)
    -   [Painter](#painter)
    -   [Adding and removing governance members](#adding-and-removing-governance-members)
-   [Acknowledgements](#acknowledgements)

<!-- tocstop -->

## Overview

Carbon's evolution process uses [proposals](#proposals) to evaluate and approve
[significant changes](#when-to-write-a-proposal) to the project or language.
This process is designed to:

-   Ensure these kinds of changes can receive feedback from the entire
    community.
-   Resolve questions and decide direction efficiently.
-   Create a clear log of the rationale for why the project and language have
    evolved in particular directions.

When there are questions, concerns, or issues with a proposal that need to be
resolved, Carbon uses its [governance](#governance-structure) system of
[Carbon leads](#carbon-leads-1) to decide how to move forward. Leads are
fundamentally responsible for encouraging Carbon's ongoing and healthy evolution
and so also take on the critical steps of the evolution process for proposals.

## Proposals

These are primarily structured as GitHub pull requests that use a somewhat more
formal document structure and process to ensure changes to the project or
language are well explained, justified, and reviewed by the community.

### Life of a proposal

-   Proposals consist of a PR (pull request) in GitHub that adds a document to
    the [`proposals/` directory](/proposals/) following
    [the template](/proposals/scripts/template.md).

-   Proposal PRs start in draft mode. When proposal PRs are ready, click on
    ["Ready for review"](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/changing-the-stage-of-a-pull-request#marking-a-pull-request-as-ready-for-review).
    This will:

    -   Route the proposal to a Carbon lead for review.

    -   Send the proposal as a broad RFC to the community.

    -   Add the "proposal rfc" label for tracking.

-   Contributors are encouraged to react with a _thumbs-up_ to proposal PRs if
    they are generally interested and supportive of the high-level direction
    based on title and summary. Similarly, other reactions are encouraged to
    help surface contributor's sentiment.

-   We use GitHub issues to discuss and track _blocking issues_ with proposals,
    such as open questions or alternative approaches that may need further
    consideration. These are assigned to carbon-leads to decide.

-   A [Carbon lead](#carbon-leads-1) will be assigned to a proposal PR. They are
    responsible for the basic review (or delegating that) as well as ultimately
    approving the PR.

-   The assigned lead should ensure that there is a reasonable degree of
    consensus among the contributors outside of the identified blocking issues.
    Contributors should have a reasonable chance to raise concerns, and where
    needed they should become blocking issues. Community consensus isn't
    intended to be perfect, and is ultimately a judgment call by the lead. When
    things are missed or mistakes are made here, we should just revert or
    fix-forward as usual.

-   Once a reasonable degree of community consensus is reached and the assigned
    lead finishes code review, the lead should
    [approve](/docs/project/code_review.md#approving-the-change) the PR. Any
    outstanding high-level concerns should be handled with blocking issues.

-   Optionally, the assigned lead can file a blocking issue for a one-week final
    comment period when they approve. This is rarely needed, and only when it is
    both useful and important for the proposal to give extra time for community
    comments.

-   The leads are responsible for resolving any blocking issues for a proposal
    PR, including the one-week comment period where resolving it indicates
    comments arrived which require the proposal to undergo further review.

-   The proposal PR can be merged once the assigned lead approves, all blocking
    issues have been decided, and any related decisions are incorporated.

-   If the leads choose to defer or reject the proposal, the reviewing lead
    should explain why and close the PR.

### Proposal roles

It is also useful to see what the process looks like for different roles within
the community. These perspectives are also the most critical to keep simple and
easily understood.

#### Proposal authors

For proposal authors, this should feel like a code review, with some broken out
issues for longer discussion:

-   Create a proposal document and draft PR following
    [the template](/proposals/scripts/template.md).

    -   [new_proposal.py](/proposals/scripts/new_proposal.py) helps create
        templated PRs.

    -   If you have open questions, filing [blocking issues](#blocking-issues)
        while preparing the PR can help resolve them quickly.

-   When ready, click on
    ["Ready for review"](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/changing-the-stage-of-a-pull-request#marking-a-pull-request-as-ready-for-review)
    in GitHub. This will:

    -   Route the proposal to a Carbon lead for review.

    -   Send the proposal as a broad RFC to the community.

    -   Add the "proposal rfc" label for tracking.

-   Address comments where you can and they make sense.

-   If you don't see an obvious way to address comments, that's OK.

    -   It's great to engage a bit with the commenter to clarify their comment
        or why you don't see an obvious way to address it, just like you would
        [in code review](/docs/project/code_review.md#responding-to-review-comments).

    -   If the commenter feels this is important, they can move it to a blocking
        issue for a longer discussion and resolution from the leads.

    -   You don't need to try to resolve everything yourself.

-   Incorporate any changes needed based on the resolution of blocking issues.
    Once the leads have provided a resolution, it's important to make progress
    with that direction.

-   When you both have
    [approval](/docs/project/code_review.md#approving-the-change) from the
    assigned lead and the last blocking issue is addressed, merge!

    -   If you end up making significant changes when incorporating resolved
        issues after the approval from the assigned lead, circle back for a
        fresh approval before merging, just like you would with code review.

#### Community

-   We use the
    ["proposal rfc" label](https://github.com/carbon-language/carbon-lang/pulls?q=is%3Apr+is%3Aopen+label%3A%22proposal+rfc%22)
    to track proposals that are in RFC.

    -   Anyone that is interested can participate once a proposal is ready for
        review and in RFC.

    -   It's OK to only comment when particularly interested in a proposal, or
        when asked by one of the leads to help ensure thorough review. Not
        everyone needs to participate heavily in every RFC.

    -   PRs that are in "draft" status in GitHub are considered
        works-in-progress. Check with the author before spending time reviewing
        these, and generally avoid distracting the author with comments unless
        they ask for them. The proposal may be actively undergoing edits.

-   Read the proposal and leave comments to try to help make the proposal an
    improvement for Carbon.

    -   Note that progress and improvement are more important than perfection
        here!

-   Try to make comments on proposals
    [constructive](/docs/project/code_review.md#writing-review-comments).
    Suggest how the proposal could be better if at all possible.

-   If there is an open question or a critical blocking issue that needs to get
    resolved, move it to its own issue that the PR depends on, and focus the
    discussion there.

    -   The issue should focus on surfacing the important aspects of the
        tradeoff represented by the issue or open question, not on advocacy.

#### Active contributors

Everyone actively contributing to the evolution of Carbon should try to
regularly:

-   Give a thumbs-up or other reaction on any interesting PRs out for RFC to
    help surface the community's sentiment around the high level idea or
    direction. Don't worry about "approving" or the detailed text of the
    proposal here.

-   If interested and time permitting, dive into some RFCs and provide
    [community feedback](#community).

#### Carbon leads

[Carbon leads](#carbon-leads-1) are responsible for making decisions rapidly and
ensuring proposal PRs land:

-   Rapidly resolve all blocking issues raised across any proposals.

-   When assigned a specific proposal PR:

    -   Make sure it gets both constructive general comments and good code
        review.

    -   Ideally, you should directly participate in the code review, but it's
        fine to ask others to help. However, ultimately you have to review and
        approve the PR.

    -   Escalate any blocking issues without a resolution that are slowing down
        the proposal to the other leads.

    -   Evaluate whether the community has had a reasonable chance to raise
        concerns and there is sufficient consensus to move forward given the
        decisions on the blocking issues. This doesn't need to be perfect
        though. Here too, we prioritize progress over perfection. We can revert
        or fix-forward mistakes whenever necessary, especially for low-risk
        changes. In rare cases, an extended final comment period can be used
        when warranted for a proposal.

    -   Once ready, approve and help the author merge the proposal.

### When to write a proposal

Any substantive change to Carbon -- whether the language, project,
infrastructure, or otherwise -- should be done through an evolution proposal.
The meaning of "substantive" is subjective, but will generally include:

-   Any semantic or syntactic language change that isn't fixing a bug.
-   Major changes to project infrastructure, including additions and removals.
-   Rolling back an accepted proposal, even if never executed.

Changes which generally will not require a proposal are:

-   Fixing typos or bugs that don't change the meaning or intent.
-   Rephrasing or refactoring documentation for easier reading.
-   Minor infrastructure updates, improvements, setting changes, tweaks.

If you're not sure whether to write a proposal, please err on the side of
writing a proposal. A team can always ask for a change to be made directly if
they believe it doesn't need review. Conversely, a reviewer might also ask that
a pull request instead go through the full evolution process.

### Proposal PRs

A proposal PR should use the `proposal` label, have a descriptive title, and
easily understood initial summary comment. Authors and leads are encouraged to
edit both as necessary to ensure they give the best high-level understanding of
the proposal possible.

A proposal PR will include a "P-numbered" _proposal document_,
`proposals/pNNNN.md`, where `NNNN` is the pull request number. This file should
be based on the [proposal template file](/proposals/scripts/template.md).

When writing a proposal, try to keep it brief and focused to maximize the
community's engagement in it. Beyond the above structure, try to use
[Inverted Pyramid](<https://en.wikipedia.org/wiki/Inverted_pyramid_(journalism)>)
or [BLUF](<https://en.wikipedia.org/wiki/BLUF_(communication)>) writing style to
help readers rapidly skim the material.

#### What goes in the proposal document

The purpose of the proposal document is to present the case for deciding to
adopt the proposal. Any information that feeds into making that decision, and
that should not be maintained as part of our living design documentation,
belongs in the proposal document. This includes background material to introduce
the problem, comparisons to any alternative designs that were considered and any
other current proposals in the same area, records of informal polls taken to
determine community preferences, and rationale for the decision based on the
project's goals.

The proposal PR can contain related changes to the Carbon project, such as
updates to the design documentation. Those changes form part of the proposal,
and need not be additionally described in the proposal document beyond a mention
in the "Proposal" section that such changes exist. For example:

```md
## Proposal

See the proposed changes to the design documents.
```

Readers of proposals are expected to consult the PR or the git commit that
merged the PR in order to understand the proposed changes.

The author of a proposal is not required to include changes to the design
documentation as part of a proposal, and it may in some cases be preferable to
decouple the proposal process from updating the design. When accepted, the
proposal would then be implemented through a series of future PRs to the rest of
the project, and the proposal document should describe what is being proposed in
enough detail to validate that those future PRs properly implement the proposed
direction.

#### Open questions

Feel free to factor out open questions in a proposal to issues that you assign
to the leads to resolve. You can even do this before sending the proposal for
review. Even after it's resolved, an open question issue can be reopened if new
information comes up during the RFC.

When opening issues, label them as
[leads questions](https://github.com/carbon-language/carbon-lang/issues?q=is%3Aissue+is%3Aopen+label%3A%22leads+question%22).
Carbon leads use this to locate and prioritize the issue for resolution.

### Review and RFC on proposal PRs

When a proposal PR is assigned to the
[carbon-leads GitHub group](https://github.com/orgs/carbon-language/teams/carbon-leads),
one of them will be assigned the PR. They are responsible for helping land that
proposal, or explaining why the project won't move forward in that direction.
The assigned lead is also ultimately responsible for the code review on the PR.
Proposals sent for review are also sent as an RFC to the entire community.

All active Carbon contributors are strongly encouraged to regularly skim the
title and summary comment of proposals under RFC that are interesting to them.
They should use GitHub reactions, including at least a thumbs-up, to show their
interest and enthusiasm about the proposal, and help encourage the author.
Writing proposals is _extremely hard work_, and we need to clearly show both
interest in the proposed direction of Carbon and appreciation for the work put
into the proposal. This is not about _approving_ the proposal, or any of its
details. It is completely fine and coherent to both give a thumbs-up to a
proposal _and_ provide a serious, blocking issue that needs to be resolved.

_Anyone_ in the community is welcome to participate in the RFC in detail if
interested. However, not everyone needs to participate in every RFC. If a
proposal is already getting actively and thoroughly reviewed, feel free to focus
your time on other proposals with fewer commenters. Even if there are issues or
problems discovered later, we can always fix them with follow-up proposals.

Both code review and high-level design comments are welcome. If an open question
comes up or a high-level blocking issue is uncovered, feel free to move it to
its own GitHub issue and assign it to the leads to resolve. That issue is also a
good place to focus discussion on that specific topic rather than the main PR.

The assigned lead should approve proposals once the following criteria are met:

-   It looks good from a code review perspective.

-   At least three thumbs-up reactions showing general community interest.

-   The community has had a sufficient opportunity to review the proposed
    change, given its scope and complexity.

-   Any remaining blocking issues are reasonably likely to resolve in a way that
    allows the proposal to move forward. It is fine if some are not fully
    decided, but a lead shouldn't approve a proposal that's unlikely to move
    forward.

The last two criteria are fundamentally judgement calls for the lead to make,
and we don't try to formulate a rigid or fixed bar for them. If resolving the
blocking issues requires significant changes, the author should also get a fresh
approval from the assigned lead after those changes, just like they would with
code review.

The assigned lead may also request a final comment period for the community when
approving. This signals to the community that the proposal is likely to be
merged once the blocking issues are resolved, and any remaining concerns need to
be surfaced. The goal is to help uncover concerns that were hidden until it was
clear that the proposal is likely to move forward. However, requesting a final
comment period is not the default; the assigned lead should only do this when
there is some reason to expect further community comment is especially important
to solicit. Common cases to consider are contentious, complex, or dramatic
changes to the language or project. Ultimately, whether this is important is a
judgement call for the lead. This will be modeled by filing a blocking issue
that resolves in one week when approving. This issue will also explain the
motivation for requesting a final comment period.

### Blocking issues

We use blocking GitHub issues to track open questions or other discussions that
the leads are asked to resolve. Any time a blocking issue is filed, that issue
forms both the primary discussion thread and where the leads signal how it is
resolved. We use issues both to track that there is a specific resolution
expected and that there may be dependencies.

We label blocking issues as
[leads questions](https://github.com/carbon-language/carbon-lang/issues?q=is%3Aissue+is%3Aopen+label%3A%22leads+question%22).

These issues can be created at any time and by any one. Issues can be created
while the proposal is being drafted in order to help inform specific content
that should go into the proposal. It is even fine to create an issue first, even
before a proposal exists, as an open question about whether to produce a
particular proposal, or what a proposal that is being planned should say. For
issues which don't (yet) have a specific proposal PR associated with them, at
some point the leads may ask that a proposal be created to help collect in a
more cohesive place a written overview of the issue and related information, but
this process need not be strictly or rigidly bound to having proposal text.

Avoid using issues for things that are just requests or suggestions on a
proposal PR. If in doubt, start off with a simple comment on the PR and see if
there is any disagreement -- everyone may already be aligned and agree. When a
comment does seem worth turning into an issue, don't worry about that as the
author or the commenter. Getting the leads to resolve disagreement isn't a bad
thing for anyone involved. This should be seen as a friendly way to move the
discussion out to its own forum where it'll get resolved, and focus the PR on
improving the proposal and getting it ready to merge.

When an issue is created from a discussion on a PR, and after the discussion on
the _issue_ all the original parties come to a happy agreement, it's totally OK
to close the issue and move back to the code review in the PR. Anyone who would
prefer the leads to still chime in can re-open the issue and the leads will
follow up, even if it's only to get confirmation that everyone _did_ end up
happy with the resolution. At the end of the day, while it's fine to resolve an
issue that _everyone_ actually ended up agreeing about (maybe once some
confusion is addressed), ultimately the leads are responsible for resolving
these issues and there is no pressure on anyone else to do so.

#### Discussion on blocking issues

Discussion on these issues, especially contentious ones, should endeavor to
focus on surfacing information and highlighting the nature of the tradeoff
implied by the decisions available. This is in contrast to focusing on advocacy
or persuasion. The goal of the issues shouldn't be to persuade or convince the
leads to make a specific decision, but to give the leads the information they
need to make the best decision for Carbon.

It's fine that some people have a specific belief of which decision would be
best; however, framing their contributions to the discussion as surfacing the
information that underpins that belief will make the discussion more
constructive, welcoming, and effective. Overall, everyone should strive to focus
on data-based arguments to the extent they can, minimizing their use of appeals
to emotion or excessive rhetoric.

None of this should preclude gathering information like polls of opinion among
groups, or signaling agreement. Where community members stand and how many agree
with that stance on any issue _is_ information, and useful to surface.

## Governance structure

### Carbon leads

Carbon leads are responsible for reviewing proposals and
[setting Carbon's roadmap](roadmap_process.md) and managing evolution. This team
should broadly understand both the users of Carbon and the project itself in
order to factor different needs, concerns, and pressures into a
[consensus decision-making process](https://en.wikipedia.org/wiki/Consensus_decision-making).

While leads may approve proposals individually, they should decide on issues
raised to them using
[blocking consensus](https://en.wikipedia.org/wiki/Consensus_decision-making#Blocking)
with a quorum of two.

Carbon's current leads are:

-   [chandlerc](https://github.com/chandlerc)
-   [KateGregory](https://github.com/KateGregory)
-   [zygoloid](https://github.com/zygoloid)

#### Subteams

As Carbon grows, the leads may decide to form subteams that provide leadership
for specific areas. These subteams are expected to largely organize in a similar
fashion to the Carbon leads, with a more narrow focus and scope. Subteam
decisions may be escalated to the Carbon leads.

### Painter

Whenever possible, we want Carbon to make syntax and other decisions based on
understanding its users, data, and the underlying goals of the language.
However, there will be times when those don't provide a clear cut rationale for
any particular decision -- all of the options are fine/good and someone simply
needs to choose which color to paint the bikeshed. The goal of the painter role
is to have a simple way to quickly decide these points.

Leads and teams may defer a decision to the painter if there is a consensus that
it is merely a bikeshed in need of paint. They may also open an issue to revisit
the color with data and/or user studies of some kind. This allows progress to be
unblocked while also ensuring we return to issues later and attempt to find more
definite rationale.

The painter is a single person in order to keep decisions around taste or
aesthetics reasonably consistent.

The current painter is:

-   [chandlerc](https://github.com/chandlerc)

### Adding and removing governance members

Any member of Carbon governance may step down or be replaced when they are no
longer able to contribute effectively. The Carbon leads can nominate and decide
on adding, removing, or replacing members using the usual evolution processes.

## Acknowledgements

Our governance and evolution process is influenced by the
[Rust](https://github.com/rust-lang/rfcs),
[Swift](https://swift.org/contributing/), and C++ processes. Many thanks to
these communities for providing a basis.
