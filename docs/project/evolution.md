# Governance and Evolution

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The Carbon project aims to provide consistent and clear governance and language
evolution over time. We want to provide a reasonably broad representation for
the community contributing to Carbon, including their diverse backgrounds and
expertise. The governance also needs to remain scalable and effective. It is
essential to be able to make even controversial decisions in a reasonable and
bounded timeframe.

## Governance Structure

Our governance structure supports
[consensus decision-making](consensus_decision_making.md):

- Community members write proposals.
- [Review managers](#review-managers) escort proposals through our consensus
  decision process.
- A [core team](#core-team) makes consensus decisions about Carbon's evolution.
- Three [arbiters](#arbiters) respond to escalations about lack of consensus,
  making decisions through majority vote.
- One [painter](#painter) who, where a consensus exists that multiple competing
  options are reasonable, decides between the provided options.

## Evolution process

Any substantive change to Carbon (whether the language, project, infrastructure,
or otherwise) must be made through a process designed to both engage with the
broader community and come to reasonable decisions in a timely fashion. We have
guidelines for
[when to follow the evolution process](#when-to-follow-the-evolution-process).

The process is:

1.  [Draft the proposal](#draft-the-proposal)

    1.  [(optional) Discuss the idea early](#optional-discuss-the-idea-early)
    2.  [Make a proposal](#make-a-proposal)
    3.  [(optional) Elicit early, high-level feedback on the proposal](#optional-elicit-early_high_level-feedback-on-the-proposal)

2.  [Solicit and address proposal feedback](#solicit-and-address-proposal-feedback)

    1.  [Request comments](#request-comments)
    2.  [Community and reviewing team comments on proposal](#community-and-reviewing-team-comments-on-proposal)
    3.  [(optional) Pause for a major revision](#optional-pause-for-a-major-revision)
    4.  [Request a review manager](#request-a-review-manager)

3.  [Reviewing team makes a proposal decision](#reviewing-team-makes-a-proposal-decision)

    1.  [Prepare the document for a proposal decision](#prepare-the-document-for-a-proposal-decision)
    2.  [Ask the reviewing team for a proposal decision](#ask-the-reviewing-team-for-a-proposal-decision)
    3.  [(optional) Use the meeting to make a proposal decision](#optional-use-the-meeting-to-make-a-proposal-decision)

4.  [Finalize the proposal decision](#finalize-the-proposal-decision)

    1.  [Publish the proposal decision](#publish-the-proposal-decision)
    2.  [Community comments on proposal decision](#community-comments-on-proposal-decision)
    3.  [(optional) Rollback the decision](#optional-rollback-the-decision)
    4.  [Execute on proposal decision](#execute-on-proposal-decision)

## Coordination Tools

We use several tools to coordinate changes to Carbon:

- **GitHub pull requests** contain the proposals and related discussion.
  Resolved proposals wil be committed with the associated decision. The pull
  request's description should link all related Discourse Forum topics and other
  references for easy browsing.
- **Discourse Forum** topics will be used for the early idea discussion, any
  deeper discussions, or more high-level and meta points.
- **Discord Chat** can be used for quick and real-time chats and Q&A.
  - If there are important technical points raised or addressed, they should get
    summarized on a relevant Discourse Forum topic.
- **Google Docs** may be used for early draft proposals. This facilitates
  collaborative editing and easy commenting about wording issues.
- **Google Hangouts Meet** will be used for VC meetings, typically for
  decisions.
  - Meetings should typically be summarized on a relevant Discourse Forum topic.
- **Google Calendar** will be used to track team meeting and vacation times.

## Governance structure

### Review managers

Review managers exist to help ensure proposals are reviewed in a timely fashion,
regardless of whether it came from a long-time contributor or someone new to the
community. They are expected to set aside personal opinions when ensuring a
proposal is correctly reviewed, including abstaining from managing their own
proposals. Proposal authors can always contact review managers if they're not
sure what to do with a proposal.

The number of review managers isn't tightly restricted. Community members may
volunteer their assistance as a way of contributing to Carbon, although the Core
team will still review participation.

Our current review managers are:

- [jperkins@google.com](mailto:jperkins@google.com)
- [shummert@google.com](mailto:shummert@google.com)

### Core team

The core team is responsible for [setting Carbon's roadmap](roadmap_process.md)
and managing evolution. This team should broadly understand both the users of
Carbon and the project itself in order to factor different needs, concerns, and
pressures into a
[consensus decision-making process](consensus_decision_making.md).

The team is expected to remain relatively small for efficiency, although members
may be added when necessary to expand representation.

Our current core team members are:

- austern@google.com
- chandlerc@google.com
- dmitrig@google.com
- gromer@google.com
- joshl@google.com
- palmer@google.com
- richardsmith@google.com
- titus@google.com

**TODO**: We want this team to eventually include non-Googlers for a broader set
of perspectives.

#### Subteams

As Carbon grows, the core team may decide to form subteams that provide
leadership for specific areas. These subteams are expected to largely organize
in a similar fashion to the core team, with a more narrow focus and scope. The
escalation path for subteams goes first to the core team and then to the
arbiters.

### Arbiters

There may be issues where the core team cannot reach consensus even after
careful discussion. In some cases, this simply means that the status quo must
hold until a more compelling case for change has been made. However, to avoid
important issues being unable to progress, any member of the community may
request that an issue which failed to gain consensus be considered by the
arbiters.

When acting as arbiters, the arbiters are only empowered to select between
different options among which none achieved consensus, including maintaining
status quo. All options must first be deliberated by the core team.

It is even more important that arbiters set aside any personal perspectives to
the extent possible and make decisions in the interest of the project. They
should typically bias towards maintaining the status quo and letting the
proposal come back with more information rather than overriding it. It is
expected to be relatively rare that the arbiters need to take on this role.

Arbiters may make a decision with a majority (two-to-one) vote. If they cannot
reach a majority vote, e.g. due to an arbiter being unavailable, the decision
returns to the core team.

There should always be three arbiters.

Our current arbiters are:

- chandlerc@google.com
- richardsmith@google.com
- titus@google.com

### Painter

Whenever possible, we want Carbon to make syntax and other decisions based on
understanding its users, data, and the underlying goals of the language.
However, there will be times when those don't provide a clear cut rationale for
any particular decision -- all of the options are fine/good and someone simply
needs to choose which color to paint the bikeshed. The goal of the painter role
is to have a simple way to quickly decide these points.

Any team may defer a decision to the painter if there is a consensus that it is
merely a bikeshed in need of paint. The team may also open an issue to revisit
the color with data and/or user studies of some kind. This allows progress to be
unblocked while also ensuring we return to issues later and attempt to find more
definite rationale.

The painter is a single person in order to keep decisions around taste or
aesthetics reasonably consistent.

The current painter is:

- [chandlerc@google.com](mailto:chandlerc@google.com)

### Adding and removing governance members

Any member of Carbon governance may step down or be replaced when they are no
longer able to contribute effectively. The core team can nominate and decide on
adding, removing, or replacing members using the usual evolution processes.

## Evolution process

### When to follow the evolution process

Any substantive change to Carbon (whether the language, project, infrastructure,
or otherwise) should follow the evolution process. The meaning of "substantive"
is subjective, but will generally include:

- Any semantic or syntactic language change that isn't fixing a bug.
- Major changes to project infrastructure (including additions and removals).
- Changes to the process itself.
- Rolling back a finalized decision (even if never executed).

Changes which generally will not require this process are:

- Fixing typos or bugs that don't change the meaning and/or intent.
- Rephrasing or refactoring documentation for easier reading.
- Minor infrastructure updates, improvements, setting changes, tweaks.

If you're not sure whether to follow the process, please err on the side of
following it. A team can always ask for a change to be made directly if they
believe it doesn't need review. Conversely, a reviewer might also ask that a
pull request instead go through the full evolution process.

### Draft the proposal

#### (optional) Discuss the idea early

We encourage proposal authors to discuss problems they're trying to solve and
their ideas for addressing it with the community early using any of our
coordination tools. The goal should be to get a feel for any prior work, as well
as other ideas or perspectives on both the problem and ways of approaching it.

These discussions can also help socialize both the problem and ideas with the
community, making subsequent discussions around a concrete proposal more
efficient.

##### Actions

- **Author**: Create an `Evolution > Ideas` forum topic to discuss the issue
  before writing the proposal.
- **Community**: Provide [constructive commentary](commenting_guidelines.md) for
  ideas when feedback is solicited.

#### Make a proposal

We use a [template for proposals](/proposals/template.md) to make it easier for
readers to recognize the style and structure. Follow the instructions under
"TODO: Initial proposal setup".

When writing a proposal, try to keep it brief and focused to maximize the
community's engagement in it. Beyond the above structure, try to use
[Inverted Pyramid](<https://en.wikipedia.org/wiki/Inverted_pyramid_(journalism)>)
or [BLUF](<https://en.wikipedia.org/wiki/BLUF_(communication)>) writing style to
help readers rapidly skim the material.

The proposal's pull request may include changes in the same repo. Please be
thoughtful about how much effort you invest this way: it can help illustrate the
intent of a proposal and avoid duplicating text in the proposal, but proposals
may also need to be rewritten substantially or be deferred/declined.

Where parts of a proposal may have several ways to address them, feel free to
list options and mark them as "open questions". When describing an open
question, it is a good idea to describe a proposed solution as well as other
options, compare their advantages, disadvantages, and non-trivial consequences.
These may be resolved during discussion, or be left for decision by the
reviewing team.

Where the proposal makes a decision between multiple options, move them to the
"alternatives" section so that it's clear why a given choice was made.

##### Drafting using Google Docs

You may optionally use the
[Google Docs template](https://docs.google.com/document/d/1sqEnIWWZKTrtMz2XgD7_RqvogwbI0tBQjAZIvOabQsw/template/preview)
for early proposal versions, which can be transferred to Markdown later. Using
Google Docs can especially help iterate on a propsal with multiple authors.

This template includes things like license headers and standard formatting. If
you already have a non-templated Doc, please create a new Doc using the template
and copy content over (without original formatting).

If you use Google Docs for drafting, be sure to still use a Markdown pull
request for the RFC.

##### Actions

- **Author**:
  - Write the proposal using [the template](/proposals/template.md).
    - The template has additional actions under "TODO: Initial proposal setup".

#### (optional) Elicit early, high-level feedback on the proposal

Authors may continue to use the `Evolution > Ideas` forum topic to advertise the
proposal and elicit early, high-level feedback. Community commenters should
favor GitHub comments (vs forum topic replies).

##### Actions

- **Author**: Update (or create if needed) the `Evolution > Ideas` forum topic
  to advertise the proposal and elicit early, high-level feedback.
  - Add the topic's link to the GitHub pull request.
- **Community**: Provide [constructive commentary](commenting_guidelines.md) for
  ideas when feedback is solicited.

### Solicit and address proposal feedback

#### Request comments

Once authors feel the proposal is in good shape for wider evaluation from the
relevant reviewing team (the core team, at present), they begin the more formal
process of evaluation by creating an `Evolution > RFCs` forum topic for
technical review of the proposal.

The topic should start off with a brief summary of the proposal and any prior
discussion, as well as links to prior discussion topics.

##### Actions

- **Author**:
  - Replace the GitHub pull request's `WIP` label with `RFC`.
  - Create an `Evolution > RFCs` forum topic.
    - Summarize the discussion points, along with a link to the pull request.
    - Add the topic's link to the pull request's description.

#### Community and reviewing team comments on proposal

Anyone in the community is welcome to
[comment on an RFC](commenting_guidelines.md). The reviewing team is _expected_
to participate at least enough to provide any relevant feedback.

Proposal authors should actively engage in the discussion and incorporate
changes to reflect feedback. Work to improve the proposal through incorporating
feedback. Authors should explicitly mention when they believe a comment has been
addressed, and commenters should be clear and explicit about whether they
believe more changes are needed.

When significant alternatives are pointed out, include them in the proposal
regardless of whether they're adopted. The "alternatives" section should be used
to document rejected alternatives as well as the original approach when an
alternative is adopted, with pros and cons either way. New "open questions" may
also be added where the author isn't confident about the best approach.

##### Actions

- **Author**:
  - Update the proposal and/or reply to comments to address feedback.
  - Create GitHub issues for any open questions to be revisited later.
- **Reviewing team and community**: Provide
  [constructive commentary](commenting_guidelines.md) for proposals.

#### (optional) Pause for a major revision

Significant changes to the proposal may become necessary as a result of the
discussion. At that point, the authors should clearly state that they are
planning a revision and the issue should be marked as `WIP` again. At this
point, the community should prioritize giving the authors time and space to work
on the revision rather than sending more feedback.

The author should treat this as going back to the draft phase. When a new
revision is ready, the authors start a new
[request for comments](#request-comments) with an updated summary of the
discussion points thus far (with links to those prior topics).

##### Actions

- **Author**:
  - Announce to the Discourse Forum topic that the proposal is undergoing major
    revision.
  - Replace the GitHub pull request's `RFC` label with `WIP`.
- **Reviewing team and community**: Refrain from commenting until the author
  solicits feedback again.

#### Request a review manager

Once discussion settles down and all comments have been resolved, the author
should request a review manager by creating a
`Evolution > Review manager requests` topic. A review manager should respond
within a day or two. This may not be needed if a review manager has already
taken ownership.

The review manager is responsible for validating that the proposal is ready for
the reviewing team to make a decision. They will ensure that at least a couple
members of the reviewing team have reviewed the document before it leaves RFC.

The review manager will announce a planned end to the community comment period on the "Evolution > RFCs" topic. The message will specify the end date, which
will be at least one week (or four working days, if longer) after
the announcement. If nothing significant arises, the
reviewing team will be asked to start making a decision as scheduled. Otherwise,
the review manager may extend the comment period by posting another message to
the "Evolution > RFCs" topic.

##### Actions

- **Author**:
  - Ensure all comments are resolved.
  - Create a `Evolution > Review manager requests` topic asking for a review
    manager, providing a link to the proposal's GitHub pull request.
    - Add the topic's link to the GitHub pull request.
- **Review manager**:
  - Ask reviewing team members to review the proposal when needed.
  - Double-check that comment threads are addressed by the proposal.
  - Update the `Evolution > RFCs` topic with a last call for comments.

### Reviewing team makes a proposal decision

#### Prepare the document for a proposal decision

Going into a decision, it's expected that no more significant changes will be
made. The proposal author should stop making any non-typo edits to the text. Any
significant edit will be treated as a major revision, cancelling the decision
request. The author should still respond to review comments, just without
making/accepting edits.

##### Actions

- **Author**: Stop making changes to the proposal.

#### Ask the reviewing team for a proposal decision

The review manager should ask the reviewing team for a decision by creating an
`Evolution > Proposal decisions` forum topic asking for consensus. The proposal
should also be added as a topic to the team's meeting agenda one week in advance
(or four working days, if longer). The agenda item gives a deadline for a
decision if no consensus can be reached on the Discourse Forum, and can be
removed if a decision is made before the meeting.

Team members should familiarize themselves with the proposal and related
discussion. Try to respond to the Discourse Forum topic promptly with:

- A position (either affirming or objecting) is strongly preferred, although
  standing aside is allowed.
  - Rationales for positions should be based on discussion on the proposal's
    `Evolution > RFCs` topic, and providing links helps write the decision.
- A request for more time to review materials, to make it clear the intent is to
  participate in the decision.
- Discussion regarding positions or the decision itself.
  - The reviewing team will participate in the proposal community comment
    period, so that substantive feedback can be incorporated by the author prior
    to requesting a decision.
- A request to use the meeting for discussion.
  - All topics for discussion will be captured either in the agenda or as a
    comment on the pull request, to ensure they're ready for the meeting.

The review manager should monitor the forum topic for consensus. If a decision
is made before the meeting, the item should be removed from the meeting agenda.
If no decision can be reached before the meeting, the meeting will be used to
make decisions.

##### Actions

- **Author**:
  - Respond to comments.
- **Review manager:**
  - Replace the GitHub pull request's `RFC` label with `needs decision`.
  - Create an `Evolution > Proposal decisions` topic for pre-meeting discussion.
  - Tentatively add the decision to the meeting one week in advance (or four
    working days, if longer), and use that meeting if necessary to reach
    consensus.
  - Monitor the topic for a consensus decision.
    - If a consensus is reached, ensure there's enough information to write a
      decision.
- **Every reviewing team member:**
  - Review the proposal again and make comments if needed.
  - Participate in reaching a consensus, or explicitly stand aside.
  - Offer justifications towards a decision.

#### (optional) Use the meeting to make a proposal decision

If the reviewing team fails to reach consensus before the meeting, the
[weekly meeting](consensus_decision_making.md#meetings) should be used to reach
consensus. The review manager is encouraged to volunteer as the note taker for
the meeting, and should help ensure there's a moderator. The author and other
community members may attend, but should behave as observers unless explicitly
called on to answer a question by the core team.

The reviewing team is responsible for producing a decision at the end of the
meeting, even if it is to defer the proposal. The review manager should verify
they understand the decision, because they will be responsible for publishing
it.

- **Author**: (optional) Consider attending the meeting to better understand the
  proposal decision.
- **Review manager**:
  - Help identify a
    [moderator and note taker](consensus_decision_making.md#roles) for the
    meeting, possibly volunteering as note taker.
  - Ensure the meeting provides enough information to write a decision.
- **Reviewing team**:
  - Participate in reaching a consensus, or explicitly stand aside.
  - Offer justifications towards a decision.

### Finalize the proposal decision

#### Publish the proposal decision

Once a decision has been reached, the review manager will draft a
[formal decision](consensus_decision_making.md#formal-decision-content) based on
Discourse Forums discussion and (if relevant) the meeting notes. They should
prepare a pull request with a PDF of the proposal for reference of what was
approved. They will post the draft decision to the `Evolution > Announcements`
forum within two working days. The post will start the proposal decision comment
period.

If the proposal is accepted, the author may now commit it. If it is deferred or
declined, the author may decide how to proceed based on whether they'd like to
continue working on it.

##### Actions

- **Review manager**:
  - Write the
    [formal decision](consensus_decision_making.md#formal-decision-content),
    possibly with help from the reviewing team.
    - (optional): Create a GitHub issue for issues that should be revisited in
      the future. Link to these from the GitHub pull request.
  - Create an `Evolution > Announcements` forum topic with the decision and a
    summary of the rationale.
    - Add the topic's link to the GitHub pull request.
  - If the proposal is accepted, approve the pull request for commit.
- **Author**:
  - If the proposal is accepted:
    - Replace the GitHub pull request's `needs decision` label with `accepted`.
    - Commit the approved pull request.
  - If the proposal is declined or deferred, decide how best to proceed:
    - If iterating on the proposal, replace the GitHub pull request's
      `needs decision` label with `WIP`.
    - If retracting the proposal, close the pull request.
- **Reviewing team**: Help draft any rationale needed by the review manager for
  the decision.

#### Community comments on proposal decision

When the proposal decision is published, it enters a comment period. The comment
period lasts two weeks, ensuring at least five working days. During this period,
the decision is in effect, although dependent changes should be made only if
they are easy to roll back.

When commenting on the decision, the community should keep in mind that the goal
of the decision is to continue to evolve Carbon in a positive direction.
Constructive criticism can improve the framing of the decision. The comment
period should not be used to continue to debate decisions unless raising
specific new information. When commenting, some questions you might want to
address are:

- Is the decision clear in its conclusion?
- Does the decision explain its rationale well?
- Have concerns or alternatives been effectively understood, acknowledged, and
  addressed to the extent possible?

If the decision is to approve the change, the author may start making changes
described in the proposal which are easy to roll back before the decision review
ends. If the author starts making changes, they must agree to help roll back
changes if the decision is rolled back. This does not mean that the decision is
final; however, we prefer to maintain velocity and roll back when needed. The
reviewing team may additionally decide that some changes _must_ wait until the
decision review is complete, e.g. if members are concerned that rollback costs
are non-obvious.

##### Actions

- **Author:** (optional) Start making dependent changes which are easy to roll
  back, and be prepared to roll back if needed.
- **Review manager:** Respond to comments and bring any significant issues to
  the reviewing team's attention.
- **Community and reviewing team**: Provide
  [constructive commentary](commenting_guidelines.md) for the proposal decision.

#### (optional) Rollback the decision

If important new information is provided, the reviewing team will engage and, if
necessary, rollback their decision. Any reviewing team member may start this by
stating their new position on the reviewing team's decision topic, although this
should be exceptional.

##### Actions

- **Author**: Roll back the committed proposal and any dependent changes.
- **Reviewing team member**: State new, non-consensus position on
  `Evolution > Decisions` forum topic.
- **Review manager**: Return to
  [asking the reviewing team for a proposal decision](#ask-the-reviewing-team-for-a-proposal-decision).

#### Execute on proposal decision

When the comment period ends without objections from the reviewing team, the
review manager should finalize the decision (approved, rejected, deferred,
etc.). The review manager should commit the proposal PDF for archival purposes.

If the decision is to approve the change, the author may make the changes
described in the proposal. There may still be review comments, but those should
exclusively deal with the **document** (formatting, structure, links, etc.), and
not the proposal. The issue should **not** be re-argued on the pull request or
other code reviews.

That does not mean that all decisions are final! Everyone involved in a decision
is human and we all make mistakes. However, once there is a clear decision,
reversing or reconsidering it should take the form of a new proposal and follow
the process of a new proposal. It may be substantially shorter to write up, but
often it will be important to clearly and carefully explain the reason to
revisit the decision.

##### Actions

- **Review manager**:
  - Commit the proposal decision.
    - Add a link to the committed decision to the proposal's pull request.
  - Update the `Evolution > Announcements` forum topic with the final decision.
- **Author**: Start making dependent changes to apply the proposal.

## Acknowledgements

Our governance and evolution process is influenced by the
[Rust](https://github.com/rust-lang/rfcs),
[Swift](https://swift.org/contributing/), and C++ processes. Many thanks to
these communities for providing a basis.
