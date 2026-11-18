# Governance & evolution revamp

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/426)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
    -   [Governance](#governance)
    -   [Evolution process](#evolution-process)
    -   [Specific changes to members](#specific-changes-to-members)
-   [Evolution process details](#evolution-process-details)
    -   [Proposal PRs](#proposal-prs)
    -   [Review and RFC on proposal PRs](#review-and-rfc-on-proposal-prs)
    -   [Open questions or blocking issues with proposal PRs](#open-questions-or-blocking-issues-with-proposal-prs)
-   [Rationale for the proposal based Carbon's goals](#rationale-for-the-proposal-based-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Status-quo](#status-quo)
    -   [Retain a large (8+ people) core team, change role and process](#retain-a-large-8-people-core-team-change-role-and-process)
    -   [Retain consistent use of fixed duration comment periods](#retain-consistent-use-of-fixed-duration-comment-periods)

<!-- tocstop -->

## Problem

Today's governance and evolution process are not meeting the current needs of
the Carbon project in a few ways:

-   Expensive both in effort expended and in wall clock latency to make
    decisions, at a time when both resources are in short supply on the Carbon
    project.
    -   This won't allow Carbon to achieve its goals given the time available.
    -   These costs are not proportional. The high constant overhead places a
        relatively higher cost on simple, uncontroversial proposals.
-   Proposal authors get a disproportionately large amount of feedback
    (exacerbating the costs for them of pushing the proposal forward) and a
    disproportionate amount of _negative_ feedback.
    -   Because there are always many more core team members than authors, even
        with the best intentions of all involved there will inevitably be an
        amplification of even accidental negative or unconstructive feedback.
-   Doesn't allow for specific intermediate questions to come up and be resolved
    at a smaller granularity than "a proposal".
    -   Sometimes, this results in paying the full overhead of a proposal to
        merely answer an intermediate question.
    -   Other times this has resulted in a near combinatorial explosion of
        complexity in the proposal due to a collection of intersecting open
        questions rather than serializing them eagerly and simplifying
        accordingly.
-   The mechanisms used for the evolution process spread elements of the
    discussion across a number of different tools and "inboxes". While each of
    these has specific features lacking in the others, the spread carries with
    it high cost, especially for new community members.

An orthogonal problem that is worth solving while here is to update the set of
people providing governance for Carbon. This is not a pressing problem, but the
set should grow to reflect the many new and active members of the project.

## Proposal

This proposal makes significant changes to both the governance model and
evolution process. Rather than describing the delta, it focuses on the newly
proposed model and process.

### Governance

The primary governance model for Carbon shifts to be handled directly by the
arbiters from the prior system. There remain three of them, and they make
decisions by blocking consensus with a quorum of two. The group is renamed to
the "Carbon Leads". They may in the future create other similarly small groups
of leads and delegate specific domains or decisions.

The core team, as well as other teams, are replaced with a directory of experts.
This is an informal list of experts with different backgrounds, experience,
expertise, and interest. It is indexed by these different areas and focuses on
breadth. People listed should be actively interested in Carbon, invested in its
success, and available to engage on their relevant topic area. The intent is to
both surface different interested parties, and provide resources for
contributors to Carbon for advice, information, or insight on specific areas.

### Evolution process

An overview of the new evolution process, but note that details are saved for
the detailed section below:

-   Proposals consist of a PR (pull request) in GitHub that adds a new document
    to the `proposals/` directory of the project, using a similar template to
    our current one.
-   Sending a _proposal_ PR for review also signifies a RFC (request for
    comment) from the entire community.
-   One of the three leads should be added as the _assigned_ reviewer.
-   Contributors should react with a _thumbs-up_ to the proposal PR if they are
    generally interested and supportive of the high level direction based on
    title and summary.
-   Comments during the RFC that represent a _blocking concern_ are moved to
    their own GitHub issue, and assigned to the leads to decide.
-   The assigned lead should ensure that at least three contributors (possibly
    including the lead) are generally supportive and react with thumbs-up. If a
    proposal doesn't have these thumbs-up, the leads together need to decide
    whether to move forward, and if so provide those thumbs-up.
-   If the lead chooses to defer or reject the proposal, they should explain why
    and close the PR.
-   Once the thumbs-up are present and the assigned lead finishes code review,
    the lead should [approve](/docs/project/code_review.md#approving-the-change)
    the PR. Any outstanding high-level concerns should be handled with blocking
    issues.
-   Optionally, the assigned lead can file a blocking issue for a one week final
    comment period when they approve if it is both useful and important for the
    proposal.
-   The proposal PR can be submitted once the assigned lead approves, all
    blocking issues have been decided, and any related decisions are
    incorporated.

It is also useful to see what the process looks like for different roles within
the community. These perspectives are also the most critical to keep simple and
easily understood.

**Proposal author:** this should feel like a code review, with some broken out
issues for longer discussion:

-   Create a proposal document and PR following the template, potentially using
    a script.
-   When ready, send this for review to the leads GitHub team to get an assigned
    reviewer.
    -   This will also send the proposal as a broad RFC to the community.
-   Address comments where you can and they make sense.
-   If you don't see an obvious way to address comments, that's OK.
    -   It's great to engage a bit with the commenter just like you would in
        code review to clarify their comment or why you don't see an obvious way
        to address it.
    -   If the commenter feels this is important, they can move it to a blocking
        issue for a longer discussion and resolution from the leads.
    -   You don't need to try to resolve everything yourself.
-   Incorporate any changes needed based on the resolution of blocking issues.
    Once the leads have provided a resolution, it's important to make progress
    with that direction.
-   When you both have an
    [LGTM](/docs/project/code_review.md#approving-the-change) from the assigned
    lead and the last blocking issue is addressed, submit!
    -   If you end up making significant changes when incorporating resolved
        issues after the LGTM from the assigned lead, circle back for a fresh
        LGTM before landing, just like you would with code review.

**Community:** anyone that is interested can participate in the RFC:

-   It's OK to only do this when particularly interested in a proposal, or when
    asked by one of the leads to help ensure thorough review. Not everyone needs
    to participate heavily in every RFC.
-   Once a proposal is sent for review and RFC, read it and leave comments to
    try to help make the proposal an improvement for Carbon.
    -   Note that progress and improvement are more important than perfection
        here!
-   Try to make comments on proposals constructive. Suggest how the proposal
    could be better if at all possible.
-   If there is an open question or a critical blocking issue that needs to get
    resolved, move it to its own issue that the PR depends on, and focus the
    discussion there.
    -   The issue should focus on surfacing the important aspects of the
        tradeoff represented by the issue or open question, not on advocacy.

**Active contributors:** everyone actively contributing to the evolution of
Carbon should try to regularly:

-   Give a thumbs-up or other reaction on any interesting PRs out for RFC to
    help surface general enthusiasm for the high level idea / direction. Don't
    worry about "approving" or the details here.
-   If interested and time permitting, dive into some RFCs with community
    feedback.

**Carbon Leads:** responsible for making decisions rapidly and ensuring proposal
PRs land:

-   Rapidly resolve all blocking issues raised across any proposals.
-   When assigned a specific proposal PR:
    -   Make sure it gets both constructive general comments and good code
        review.
    -   Ideally, you should directly participate in the code review at least,
        but it's fine to ask others to help. However, ultimately you have to
        give an LGTM.
    -   Escalate any blocking issues without a resolution that are slowing down
        the proposal to the other leads.
    -   Evaluate whether an extended final comment period is important for the
        community given the nature of the proposal.

### Specific changes to members

This proposal asks Kate Gregory to join the leads, replacing Titus Winters who
doesn't currently have the bandwidth to participate as actively. Given the shift
to the leads handling a more direct share of the evolution process, it seems
prudent to make this change at the same time.

No specific composition of the experts directory is proposed here. Instead, it
is suggested to collaboratively build that with interested parties adding
themselves and helping structure the domains / areas to be covered. Those
changes shouldn't require any formal proposal, and can simply be approved by any
one of the leads.

## Evolution process details

### Proposal PRs

A proposal PR should use the `proposal` label, have a descriptive title, and
easily understood initial summary comment. Authors and leads are encouraged to
edit both as necessary to ensure they give the best high-level understanding of
the proposal possible. The proposals should then use the template markdown file
to describe itself fully. This template will have an additional section to
contain the rationale for the proposal being accepted.

The purpose of the rationale section will match the prior process's rationale:
it must explain how the proposal furthers the goals for Carbon. It is important
that proposals stay aligned with Carbon's goals, and where they need to change
over time we change the goals themselves rather than inventing new rationale
incrementally. While the author of the proposal should suggest an initial
rationale, the reviewers and leads can also help improve this to make sure it
captures the basis on which the proposal is accepted. Authors shouldn't stress
about getting the rationale right out of the box, this is among the easiest
parts to improve with help from the community.

Proposal PRs can also include changes to the rest of the Carbon project as part
of the PR, in subsequent PRs that are referenced for context, or they can be
stand-alone changes that are implemented through a series of future PRs to the
rest of the project. All of these options are fine.

There won't be a separate "decision" file for proposals as the rationale will be
incorporated, and the decision will be marked by closing the PR. Existing
proposals should be updated to merge their decisions into the document so that
we have a simpler layout of the proposals tree and can have simple and easy
tooling around it.

PRs that are in "draft" status in GitHub are considered works-in-progress
(rather than prior `WIP` label). Check with the author before spending time
reviewing these, and generally avoid distracting the author with comments unless
they ask for them. The proposal may be actively undergoing edits.

Feel free to factor out open questions in a proposal to issues that you assign
to the leads to resolve. You can even do this before sending the proposal for
review. Even after resolved, an open question issue can be reopened if new
information comes up during the RFC.

### Review and RFC on proposal PRs

When a proposal PR is sent for review by the leads, one of them will be assigned
the PR and is responsible for helping land that proposal, or explaining why the
project won't move forward in that direction. The assigned lead is also
ultimately responsible for the code review on the PR. Proposals sent for review
are also sent as an RFC to the entire community.

All active Carbon contributors are strongly encouraged to regularly skim the
title and summary comment of proposals under RFC that are interesting to them.
They should use GitHub reactions, including at least a thumbs-up, to show their
interest and enthusiasm about the proposal, and help encourage the author.
Writing proposals is _extremely hard work_, and we need to clearly show both
interest in the proposed direction of Carbon and appreciation for the work put
into the proposal. This is not about _approving_ the proposal, or any of its
details. It is completely fine and coherent to both give a thumbs-up to a
proposal _and_ provide a serious, blocking issue that needs to be resolved.

_Anyone_ in the community, of course including active contributors, is
encouraged to participate in the RFC in detail if interested. However, not
everyone needs to participate in every RFC. If a proposal is already getting
actively and thoroughly reviewed, feel free to focus your time on other
proposals with fewer comments. Even if there are issues or problems discovered
later, we can always fix them with follow-up proposals.

Both code review and high-level design comments are welcome. If an open question
comes up or a high level blocking issue is uncovered, feel free to move it to
its own issue and assign it to the leads to resolve. That issue is also a good
place to focus discussion on that specific topic rather than the main PR.

The assigned lead should provide an LGTM on proposals once the following
criteria are met:

-   It looks good from a code review perspective.
-   At least three thumbs-up reactions showing general community interest.
-   The community has had a sufficient opportunity to review the proposed
    change, given its scope and complexity.
-   Any remaining blocking issues are reasonably likely to resolve in a way that
    allows the proposal to move forward. It is fine if some are not fully
    decided, but a lead shouldn't provide an LGTM for a proposal unlikely to
    move forward.

The last two criteria are fundamentally judgement calls for the lead to make,
and we don't try to formulate a rigid or fixed bar for them. If resolving the
blocking issues requires significant changes, the author should also get a fresh
LGTM from the assigned lead after those changes, just like they would with code
review.

The assigned lead may also request a final comment period for the community when
giving an LGTM. This signals to the community that the proposal is likely to
move forward once the blocking issues are resolved, and any remaining concerns
need to be surfaced. The goal is to help uncover concerns that were hidden until
it was clear that the proposal is likely to move forward. However, this is not
the default. The assigned lead should only do this when there is some reason to
expect further community comment is especially important to solicit. Common
cases to consider are contentious, complex, or dramatic changes to the language
or project. Ultimately, whether this is important is a judgement call for the
lead. This will be modeled by filing a blocking issue that resolves in one week
when giving the LGTM. This issue will also explain the motivation for requesting
a final comment period.

### Open questions or blocking issues with proposal PRs

Any time an open question or a blocking issue is filed for the leads to resolve,
that issue forms both the primary discussion thread and where the leads signal
how it is resolved. We use issues both to track that there is a specific
resolution expected and that there may be dependencies.

These issues can be created at any time and by any one. Issues can be created
while the proposal is a WIP or draft in order to help inform specific content
that should go into the proposal. It is even fine to create an issue first, even
before a proposal exists, as an open question about whether to produce a
particular proposal, or what a proposal that is being planned should say. For
issues which don't (yet) have a specific proposal PR associated with them, at
some point the leads may ask that a proposal be created to help collect in a
more cohesive place a written overview of the issue and related information, but
this process need not be strictly or rigidly bound to having proposal text.

Discussion on these issues, especially contentious ones, should endeavor to
focus on surfacing information and highlighting the nature of the tradeoff
implied by the decisions available. This is in contrast to focusing on advocacy
or persuasion. The goal of the issues shouldn't be to persuade or convince the
leads to make a specific decision, but to give the leads the information they
need to make the best decision for Carbon. It is of course fine that some people
have a specific belief of which decision would be best. However, by framing
their contributions to the discussion as surfacing the information that
underpins that belief the discussion is more likely to be constructive,
welcoming, and effective. Overall, everyone should strive to minimize their use
of [rhetoric](https://en.wikipedia.org/wiki/Rhetoric) or other
[persuasive methods](https://en.wikipedia.org/wiki/Persuasion#List_of_methods)
to the extent they can. However, none of this should preclude gathering
information like polls of opinion among groups, or signaling agreement. Where
community members stand and how many agree with that stance on any issue _is_
information, and useful to surface.

Avoid using issues for things that are just requests or suggestions on a
proposal PR. If in doubt, start off with a simple comment on the PR and see if
there is any disagreement -- everyone may already be aligned and agree. When a
comment does seem worth turning into an issue, don't worry about that as the
author or the commenter. Getting the leads to resolving a disagreement isn't a
bad thing for anyone involved. This should be seen as a friendly way to move the
discussion with more disagreement out to its own forum where it'll get resolved,
and focus the PR on improving the proposal and getting it ready to land.

When an issue is created from a discussion on a PR, and after the discussion on
the _issue_ all the original parties actually come to a happy agreement, it's
totally ok to just close the issue and move back to the code review in the PR.
Anyone who would prefer the leads to still chime in can just re-open the issue
and the leads will follow up, even if its just to get confirmation that everyone
_did_ end up happy with the resolution. At the end of the day, while it's fine
to resolve an issue that _everyone_ actually ended up agreeing about (maybe once
some confusion is addressed), ultimately the leads are responsible for resolving
these issues and there is no pressure on anyone else to do so.

## Rationale for the proposal based Carbon's goals

Our goals document identifies that "[t]he community needs to be able to
effectively engage in the direction and evolution of the project and language,
while keeping the process efficient and effective. That means we need an open,
inclusive process where everyone feels comfortable participating. Community
members should understand how and why decisions are made, and have the ability
to both influence them before they occur and give feedback afterward."

Our prior process, in a well-intentioned effort to ensure we have scope for
community engagement, is not as efficient and effective as we would like, and
that lack of efficiency ironically has a negative impact on community engagement
that outweighs the benefits. The proposed process removes several artificial
delays from our process and enables much earlier feedback and decision-making as
part of proposal development, ideally without introducing a negative impact on
any of the other aspects of this goal.

## Alternatives considered

### Status-quo

We could simply not make a change here.

Advantages:

-   No need to make a change.

Disadvantages:

-   Does not solve any of the problems.

### Retain a large (8+ people) core team, change role and process

We could attempt to solve the problems cited while keeping the larger core team
structure. This would largely involve carefully changing the role that members
of the core team fill when making decisions, and the process used to make those
decisions.

Advantages:

-   Less dramatic change to the governance structure
-   Retains a leadership group that can have some realistic breadth across the
    community

Disadvantages:

-   Extremely difficult to formulate a role that both serves a purpose and
    addresses the cited problems.
-   Every idea for both process and role that seemed to potentially help the
    problems listed also ended up heavily leaning on the arbiters and thus
    reducing the utility of the role.
-   Continues to require forming and sustaining the core team.
-   Would inherently result in slower progress.

### Retain consistent use of fixed duration comment periods

These were added based on the repeated failure modes seen in other language
communities where members of the community didn't feel they had sufficient
opportunity to participate in discussions. We could retain them with the new
process.

Advantages:

-   Defends against failure modes seen in other language evolution processes.
-   Ensures a predictable amount of time for the community to comment.

Disadvantages:

-   Would dramatically slow the rate of progress.
-   Reduces the ability of proposal authors to fully feel satisfied and rewarded
    for their effort by separating all of their work from any of the benefits
    being accrued.
    -   This kind of delay decreases human feelings of satisfaction with work,
        which in turn disincentives people making proposals and seeing them
        through to completion.
-   Unclear whether the failure modes observed in other language evolution
    processes will actually occur for Carbon.
    -   Many other aspects of the projects and communities are sufficiently
        different to make the prediction difficult.
    -   Initially, the size of the project alone makes these concerns a
        non-issue.
-   Makes it too easy to delay commenting until the end of the period.
    -   This in turn can cause comments to be rushed to avoid the period
        expiring.
