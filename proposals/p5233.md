# Towards more async "sync"-ing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5233)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Last Week In Carbon](#last-week-in-carbon)
        -   [Discord channel to surface interesting topics](#discord-channel-to-surface-interesting-topics)
    -   [Discussion / demo session every two months](#discussion--demo-session-every-two-months)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep the existing structure](#keep-the-existing-structure)
    -   [Drop the discussion sessions entirely](#drop-the-discussion-sessions-entirely)
    -   [Drop the weekly summaries entirely](#drop-the-weekly-summaries-entirely)
    -   [Select a slightly different format or structure](#select-a-slightly-different-format-or-structure)

<!-- tocstop -->

## Abstract

Proposal to switch our week-to-week Carbon project development syncs to be more
async.

-   Start of each week, create a summary of what happened last week.
-   Publish this in GitHub discussions for async reading and further discussion.
-   Stop our weekly meeting focused on these summaries.
-   Start up a new discussion meetings every two months.
    -   Structure will be a 10-minutes-or-less update, and a "demo".
    -   Demo may be traditional: showcase a newly landed thing in Carbon.
    -   Or demo may showcase an interesting top-of-mind language design
        discussion.
    -   Either way, goal will be to field lots of questions about the topic and
        have a good discussion everyone understands, not to reach some
        "conclusion" or "decision".

## Problem

The weekly Carbon meeting is providing value by aggregating all the activity
across our forums and helping both active contributors and folks following the
Carbon project keep up-to-date on the different discussions that take place.

However, it has stopped being a forum with active discussions where we dive
deeply into any specific topics. The summaries are also presented live and
synchronously which isn't adding much value on top of having them written up in
our notes.

Somewhat tongue-in-cheek version of the problem: our weekly meeting in its
current format should probably be an email. ;]

But we don't want to just get rid of the meeting and lose the value of
aggregating activity across the project. We need a new structure that is closer
to that of an email. And ideally, we should re-capture some of the discussion
benefits that we have had in the past.

## Background

Currently, Carbon has a
[weekly sync](https://github.com/carbon-language/carbon-lang/blob/trunk/CONTRIBUTING.md#:~:text=The%20weekly%20sync%2C%20where%20contributors%20are%20welcome.)
meeting for contributors, that opens with a detailed summary of all the
discussions and activity over the past week.

## Proposal

We propose to restructure our weekly sync meeting to be an asynchronous process
that still provides weekly cross-project and cross-contributor visibility. We
suggest a "Last Week in Carbon" structure based on the weekly summary structure
we are already using in our weekly syncs. But these will instead be posted
towards the beginning of the week in a GitHub Discussion thread.

We also suggest adding a less frequent discussion meetings with a structure
designed to help elicit broader discussion across the community. We suggest a
cadence based on the Carbon Copy newsletter of every two months. The start will
be a _very_ brief 10-minute update on major developments in the project, and
then a 45-minute "demo" of some kind. Anyone working on Carbon who has a good
idea for something to demo can sign up to do so. If we don't have a technical
demo, we'll pick an active and interesting language design discussion and demo
our language design process by having a small part of that discussion live.
We'll encourage everyone to ask any questions needed to follow along and focus
on letting folks understand a slice of what is happening in Carbon development
over reaching a specific resolution or outcome.

## Details

### Last Week In Carbon

The goal is to base these heavily on the existing weekly sync meeting minute
summaries of discussion and activity in the Carbon project and weekly
newsletters like LLVM Weekly.

Note, these should _not_ be prose newsletters. They should focus on structured
information in a bulleted format that is easily skimmed.

The exact structure, process, and format used should be heavily driven by the
contributors producing these posts, but we suggest using a collaborative tool
(docs) for drafting, and then posting the results to GitHub Discussions for
broader visibility and discussion. We also suggest that these be time boxed to a
few hours of work total and the content filtered as needed to fit that time box
rather than pushing to include every topic for summarizing. Beyond that
high-level guidance, the exact format is up to those authoring it. We do
encourage authors to pull in other contributors to help wherever useful.

We specifically suggest that only issues-for-leads and proposals be
_exhaustively_ summarized. Whoever is creating these summaries should use their
own judgement to filter to the most relevant things to summarize while trying to
catch things that might otherwise be missed. As this is imperfect, we will also
create some specific ways that individuals can help surface a specific thing
they would like to see included in these.

#### Discord channel to surface interesting topics

We would like an easy way for folks to flag things that are interesting to
include into the summaries, especially PRs that might otherwise be missed. We
suggest one or both of two options far flagging these:

1. Create a dedicated discord channel that folks can post links / snippets to
   that they find interesting.
2. Create a discord channel that automatically gets bot postings for each PR
   that is merged, and encourage folks to use reactions to signal interest in
   that being included in the summary.

### Discussion / demo session every two months

The goal is to provide an easy way for a much wider audience to follow along
with Carbon's development and get to see interesting details about how that
development is progressing.

We suggest starting this off with a brief overview of project progress since the
previous session, but keep this overview _very_ brief: strictly less than 10
minutes of the meeting.

Then we suggest soliciting a specific "demo" of something recent and interesting
in the project. These can take a wide variety of forms, but some primary ones we
anticipate initially:

-   Demo of recently working functionality in the toolchain (or a recent &
    interesting bug).
-   "Demo" of our design process by selecting a current and relevant discussion
    topic and holding an initial discussion on that topic live for folks to
    observe and participate if interested.

We will explicitly encourage folks to persistently ask questions for context and
background they don't understand, as answering these questions is an effective
informal way of spreading basic understanding of Carbon as it develops.

The goal is not to resolve questions about the design or decide anything in
these sessions, but to give a wider audience an opportunity to see and join some
aspects of our development process periodically while also getting a very brief
update on the overall project status.

Initially we suggest scheduling these for every two months with the goal of
aligning them to newsletters. However, this is not a strict scheduling goal and
we should consider adjusting frequency as needed. We should also reschedule as
needed to maximize folks' availability and freely skip any where we don't have a
good demo ready-to-hand.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Creates a more timezone and working schedule inclusive way to track
        Carbon progress week-over-week.
    -   Creates a more easily accessed synchronous meeting by having it be
        scheduled infrequently to minimize schedule disruption.
    -   Makes the synchronous meeting a more welcoming and engaging environment
        by focusing on a relevant topic and using a structure that lets people
        engage in different ways.

## Alternatives considered

### Keep the existing structure

We could make no change, but there has been a consistent lack of fully utilizing
the current meeting structure and so it seems like some change is motivated
based on how people are engaging.

The current structure is also especially expensive for the team, and so it seems
important to at least explore some alternatives that may open up better scaling,
either in team size or timezones.

### Drop the discussion sessions entirely

A different approach would be to completely drop having a synchronous discussion
meeting. However, this would lose one of the historically impactful aspects of
the weekly meeting that we hope to recover with this approach: telegraphing to
the broader potential Carbon community (largely the existing C++ community) how
we are going about developing Carbon.

We have had several instances of specifically positive feedback from observers
or more casual participants in historical weekly meetings where deeper design
discussions took place or where we showcased specific issues or capabilities.
However, these kinds of topics are not always available and so not something we
can sustain at the pace of every week.

The proposal picks the alternative of reducing the frequency and adjusting the
structure of the synchronous meeting to try and capture this value at a much
more sustainable cost and with a larger and more consistent audience. If the
audience doesn't materialize or we get feedback that this impact isn't landing,
we should revisit this aspect of the structure.

### Drop the weekly summaries entirely

This would lose the main value that we currently get from the weekly meeting,
and one that many across the team have repeatedly cited: both a way to catch up
and a way to avoid missing discussions happening in a forum or at a time that
they missed.

Keeping this, and making it _even more_ asynchronous in nature, further our goal
of making the Carbon development processes and community open and welcoming of
people from different locations, working on different schedules, or with
different levels of involvement.

### Select a slightly different format or structure

There is a wide range of possible formats and structures for both weekly
summaries and a discussion meeting. This proposal picks two based on trying to
closely fit the distinct goals and value propositions of each component of our
historically weekly meeting. But there are many others that could potentially be
considered.

Currently, the suggestion is to try these out and iterate on these based on the
experience. When making this significant of a change it seems hard to predict
accurately how the details will land, and simply selecting a plausible starting
structure is more important than getting the _best_ structure. We should make
minor adjustments to both as needed based on our experience with the new format.
