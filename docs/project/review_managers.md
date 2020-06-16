# Review managers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Evolution process duties

As part of [assisting in the evolution process](evolution.md#review-managers),
review managers are expected to:

- Monitor and respond to topics in
  [Evolution > Review manager requests](https://forums.carbon-lang.dev/c/evolution/review-manager-requests/15).
- Announce when RFCs are approaching
  [readiness for a decision](evolution.md#request-a-review-manager).
- [Escort the proposal to a decision.](evolution.md#ask-the-reviewing-team-for-a-proposal-decision)
- [Author and publish proposal decisions.](evolution.md#finalize-the-proposal-decision)

Review managers should additionally provide advice and assistance to proposal
authors where appropriate, to help ensure the smooth operation of the evolution
process.

## Templates

### Comment deadline

When an RFC approaches readiness for a decision, the review manager must post a
1-week comment deadline.

Post the deadline on the topic in
[Evolution > RFCs](https://forums.carbon-lang.dev/c/evolution/rfcs/6):

```markdown
COMMENT DEADLINE - YYYY-MM-DD

We’re asking that interested contributors please try to comment on this
proposal, either in the pull request or this topic, by end-of-day DAYOFWEEK, MONTH DD,
Pacific Time. We’d like to resolve discussion at that point in order to request
a decision.
```

### Decision request

When a proposal moves to decision, the review manager must start the decision
topic.

Request a decision on a new topic in
[Evolution > Decisions](https://forums.carbon-lang.dev/c/evolution/decisions/7),
titled "Request for decision: PROPOSAL":

```markdown
@core_team: Please provide a decision on "PROPOSAL".

Links:

- [Proposal PR](LINK)
- [RFC topic](LINK)

Please focus on affirm, object, and stand aside comments in this topic; other
things specific to reaching consensus may be included. Affirm and object
comments should include justification for a decision. If you have proposal
questions, please comment on the proposal itself.

I’ve added this decision as a tentative item for the YYYY-MM-DD core team
meeting. If consensus is reached before then, the agenda item will be removed.
```

Link to the decision request on the topic in
[Evolution > RFCs](https://forums.carbon-lang.dev/c/evolution/rfcs/6):

```markdown
A decision is now [formally requested](LINK) on this proposal. Contributors may
still comment here, but keep in mind the proposal should not change while the
reviewing team is considering it.
```

### Decision announcement

When a proposal has a decision made, the review manager must publish and
announce the decision.

Decisions should use the [decision template](/proposals/template-decision.md).
Accepted proposals should create a pull request for the decision;
declined/deferred proposals proposals should have the same template used as a
comment on the proposal.

Announce the decision on a new topic in
[Evolution > Announcemented](https://forums.carbon-lang.dev/c/evolution/announcements/8),
titled "[DECISION] PROPOSAL":

```markdown
"PROPOSAL" has been DECISION.

Please read the [formal decision PR](LINK) for details.

- [Proposal PR](LINK)
- [RFC topic](LINK)
- [Decision topic](link)

This decision is now entering the
[decision comment period](https://carbon-lang.dev/docs/project/evolution.html#community-comments-on-proposal-decision),
and, assuming no substantive issues, will be finalized on YYYY-MM-DD. Please
keep in mind that objections may be posted here.
```

Link to the decision announcement on **both** topics in
[Evolution > RFCs](https://forums.carbon-lang.dev/c/evolution/rfcs/6) and
[Evolution > Decisions](https://forums.carbon-lang.dev/c/evolution/decisions/7):

```markdown
This proposal has been DECISION, and the [decision announced](LINK). Please keep
in mind, objections may still be posted to the decision topic as part of the
decision’s comment period.
```
