<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM Exceptions.
See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon: Consensus decision-making

Carbon's teams will use a
[blocking consensus decision-making process](https://en.wikipedia.org/wiki/Consensus_decision-making#Blocking)
to make decisions on proposals. We want all voices to be heard; we expect to
iterate, identify and address concerns in order to reach consensus. A decision
is approved when there is full agreement, although some team members may
actively choose to stand aside.

We expect most decisions to be made in public, although some exceptional
decisions will be private if a team member requests it. In any case, it's
expected that only the reviewing team will actively participate in consensus
discussions.

## Main consensus process

Any team member involved in a decision may object and break consensus.
Dissenters should present their concerns, ideally with proposed fixes. Another
round of discussion may occur in order to address concerns. The goal is to
ensure that team members are empowered to surface narrow but critical concerns
that may only be noticed by a minority.

Team members have a fundamentally different role while working towards consensus
on a decision than at other times on the project. They are expected to:

- Set aside personal advocacy and preferences, and focus on the best decision
  for the project and community.
- Critically evaluate whether their concerns are severe enough to warrant
  blocking.
- Focus on the information presented in a proposal, rather than adding new
  information. If new information is needed, the request for it should be the
  decision.
- Recognize their own biases and stand aside when unable to form an objective
  position.

More about consensus decision making may be found from
"[On Conflict and Consensus](https://web.archive.org/web/20111026234752/http://www.ic.org/pnp/ocac/)",
and https://www.consensusdecisionmaking.org/.

## Formal decision content

A formal decision consists of:

- The decision itself.
- A summary of the decision rationale.
- A summary of important discussion points.

Here are some possible decisions with their meanings:

- **accepted**: Yes, we want this now.
- **declined**: No, we don't think this is the right direction for Carbon.
- **needs work**:
  - We need more information or data before we can decide if this is the right
    direction.
  - We like the direction, but the proposal needs significant changes before
    being accepted.
- **deferred**:
  - We like the direction, but it isn't our priority right now, so bring the
    proposal back later.
  - We aren't sure about the direction, but it isn't our priority right now, so
    bring it back later.

When a proposal has open questions, the formal decision must include a decision
for each open question. That may include filing GitHub issues to revisit the
open questions later.

Each decision should have a rationale based on Carbon's
[goals, priorities, and principles](goals.md). When the rationale requires
explanation, the application of these goals should be documented. The reviewing
team should check any relevant prior decisions and update or resolve
inconsistencies in order to have a clear and consistent application of Carbon's
goals.

At times, the goals, priorities and principles will simply not cover or provide
a way to make a decision. In that case, the reviewing team is expected to shift
the discussion to start with adjusting the goals, priorities and principles to
cover the new context and questions. This should then be the first decision they
have to make, and the technical decision should only be made afterward, so that
it can cite the now pertinent goals, priorities and principles.

## Meetings

In order to ensure we can rapidly make decisions, each team will have a standing
weekly meeting slot, which observers may attend. Members are expected to do
their best to keep the slot available. Meetings will be held using Google
Hangouts Meet.

Each time a team needs to make a decision, e.g., about a change to Carbon, it is
expected that we'll try to make a decision without using a live meeting. We will
only hold meetings when decisions cannot be resolved before the meeting.

### Agenda

An agenda will be maintained for upcoming meetings. If there are no agenda
items, meetings will not be held.

Agenda items (e.g., proposals) should be added at least one week in advance (or
four working days, if longer), so that members have time to review items added
to the agenda. Sub-items (e.g., proposal discussion points) should be added at
least one day before the meeting; every open GitHub pull request comment is
implicitly a sub-item. Please feel free to add items to the agenda and remove
them later if the agenda item is resolved over other communication forums.

Team members are expected to prepare for the meeting by ensuring they're
familiar with the proposal and related discussion. The meeting will not include
any presentation or summary. Discussion will go over the proposal's agenda item
and sub-items, with each agenda item resolving with a decision.

### Roles

As part of organizing a meeting, we will have two roles:

- A **moderator** who is responsible for ensuring discussion stays on track.
- A **note taker** who will record minutes for each meeting.

Roles should be known before each meeting. These may or may not be staffed by
reviewing team members; anybody taking a role should plan to be less involved in
discussion in order to focus on their role. The roles should be handled by
different people.

During the meeting, the moderator is responsible for ensuring we move in an
orderly fashion from topic to topic. Meetings will have a Discord Chat that the
moderator will watch. Attendees should let the moderator know whether they want
to speak to the current topic or a new topic, and will be added to the
appropriate queue. When it's the next person's turn to speak, the moderator will
indicate whose turn it is. If there are few enough active participants, we may
choose to skip the queue for a meeting.

The moderator is also responsible for ensuring discussions stay on track. Only
the moderator should interrupt speakers or speak outside the queue, and the
moderator _should_ do so if discussion isn't progressing. The moderator is also
responsible for polling for consensus on decisions during the meeting.

All team members should keep in mind that the ultimate goal of each meeting is a
decision, and to consider what they discuss during a meeting in keeping with
that goal.

There may be multiple review managers in a meeting, depending on the number of
proposals being reviewed. Each needs to make sure they know decision information
for the proposal they're managing when the meeting ends.

### Observers

While community members outside of the team may attend meetings, they are
expected to be passive observers. They should only speak if prompted by a member
of the team (e.g., asking a question to a proposal author). Community discussion
belongs on forums, and the meetings are for team discussions and decisions.

## Unavailable team members

If a team member is asked for a pre-meeting decision and hasn't responded by two
working days before the meeting, the team may assume they're unavailable and
standing aside. The intent is to avoid unnecessary meetings. Not responding is
strongly discouraged; please respond if you want the meeting.

Team members are encouraged to notify others when they're on vacation or
otherwise unavailable for a while, and add it to the shared calendar. Members
are also encouraged to stand aside from decisions if unavailable for more than
half a week. If they have a specific interest in a topic or decision, they may
ask for a delay while they're out.
