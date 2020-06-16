# Change comment/decision timelines in proposal process

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

**_PLEASE_ DO NOT SHARE OUTSIDE CARBON FORUMS**

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

## Table of contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Problem](#problem)
- [Proposal](#proposal)
- [Details](#details)
  - [Open Question/Bikeshed](#open-questionbikeshed)
- [Alternatives considered](#alternatives-considered)
  - [Alternative 1:](#alternative-1)
  - [Alternative 2:](#alternative-2)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem

In the existing approval process, the deadline for a decision is set to be at
least a week after the comment period. The request for final comments currently
asks that remaining comments be provided within a day, essentially setting a
soft deadline for the end of the comment period unless issues arise. As such,
the end of the comment period--which is where most of the decision work usually
takes place--is not known very far in advance. This makes it difficult for core
team members to prioritize work to get proposals fully reviewed by the end of
the comment period.

## Background

People tend to prioritize work based on deadlines. In the current proposal
approval process, the deadline for final comments is not known until the day
before they are due (if not extended). Ideally, we want the core team to have
all issues with a proposal raised and addressed to the extent possible prior to
entering the decision phase. As such, the work of the core team is front-loaded
to the comment phase, and the decision is usually delivered shortly after the
end of the comment phase and well before the decision deadline. While the
comment period usually begins long before the final request for comments is
issued, there are often multiple outstanding proposals (as well as non-Carbon
work) competing for the attention of the core team.

## Proposal

Announce the end of the comment period further in advance, while reducing the
interval between the end of the comment period and the decision meeting.

The advanced notification will allow team members to more easily prioritize
their review work. It also reflects the importance of getting issues surfaced
during the review period rather than the decision period.

## Details

- A deadline for final comments will be published at least 7 calendar days (or 4
  working days, if longer) in advance (instead of the current 1 day). On or
  before the deadline date, the deadline may be extended if the review manager
  determines that there is still productive discussion going on.
  - At the time the comment period deadline is announced, the proposal will be
    added to the agenda of the next core team meeting following the comment
    period deadline.
- There must be a minimum of four working days between the end of the comment
  period and the day of the meeting.
  - If the deadline for comments is extended, the agenda item will be moved, if
    necessary, by the review manager.

### Open Question/Bikeshed

Do we want to add the proposal to the core team meeting agenda when the deadline
for comments is announced, or when the deadline is reached? The advantage of the
latter is that no adjustments need to be made if the deadline is extended. The
advantage of the former is that it gives people an idea of whether a core team
meeting can be canceled further in advance.

## Alternatives considered

### Alternative 1:

Leaving the approval process as it is.

### Alternative 2:

Reducing the amount of time between the end of the comment period and the
decision meeting.

This option was not chosen because the shorter period could result in more
meetings, as the deciders would have less time to get their decisions in. It has
the potential to reduce the velocity of decisions if it results in decisions
happening later than they would with an earlier meeting date (decision
deadline).
