# Make the Carbon experiment public

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1363)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Why go public now?](#why-go-public-now)
    -   [Planned process for going public](#planned-process-for-going-public)
        -   [Public site and communications plan](#public-site-and-communications-plan)
        -   [Plan for ACLs once public](#plan-for-acls-once-public)
    -   [Risks and mitigations](#risks-and-mitigations)
        -   [Too many cooks in the kitchen](#too-many-cooks-in-the-kitchen)
        -   [Community management overload](#community-management-overload)
        -   [Added distraction or confusion to the C++ evolution process](#added-distraction-or-confusion-to-the-c-evolution-process)
        -   [Added distractions from existing new programming languages.](#added-distractions-from-existing-new-programming-languages)
        -   [Friction with existing LLVM and Clang communities](#friction-with-existing-llvm-and-clang-communities)
        -   [Labeled as vaporware](#labeled-as-vaporware)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Delay going public until we have working interop demonstrated](#delay-going-public-until-we-have-working-interop-demonstrated)
    -   [Delay going public until Carbon is a compelling option for C++ developers to adopt](#delay-going-public-until-carbon-is-a-compelling-option-for-c-developers-to-adopt)

<!-- tocstop -->

## Problem

Open, transparent, and public development is often the best way to build a
developer community. Long-term evolution of Carbon behind closed doors does not
align with our core principles.

While early exploration with a limited group of participants has been effective
at bootstrapping, a primary goal for the Carbon experiment is to establish
whether there will be **broad industry interest and participation** in this
direction. We expect this kind of interest and participation to ultimately
result in widespread adoption if the technical components of the experiment
succeed. We both want to see how the industry reacts to Carbon and think that
reaction will be much more positive if the industry can actively participate and
help shape the language.

## Background

Historically, Carbon has followed a "quiet" development model. While developed
on GitHub using an open source process, the project was not publicly visible or
discussed outside the invited set of early participants.

Carbon developed an initial set of criteria that we expected to signal the
correct time for the project to become public:

-   Broader field experience required
-   Sustained interest from multiple organizations and individuals
-   A prototype implementation
-   A demonstration of potential
-   Learning material
-   Prepared for broader contributions and feedback
-   Prepared for a launch event

See the removed `going_public.md` document in this
[proposal](https://github.com/carbon-language/carbon-lang/pull/1363) for the
full historical criteria and analysis.

## Proposal

We should make the Carbon experiment public as soon as reasonably possible,
specifically at the upcoming [C++ North](https://cppnorth.ca) conference.

Many of the criteria outlined previously have not yet been met, and this
proposal specifically suggests shifting Carbon to be public without waiting for
them. Instead, we should make Carbon public largely as it is today. We should be
open and transparent about the current status. We should frame this as making a
_nascent_ project public in order to _build it in the open_, rather than
suggesting it is "done" or "ready" even for evaluation.

### Why go public now?

There is no perfect time to shift the project public, it is a tradeoff. The
earlier we move the project to be public, the earlier we can gain more broad
feedback and participation from the industry. However, it comes at the cost of
introducing people to the project at an earlier stage with less material to help
them get started. We believe Carbon has crossed the point where this tradeoff
suggests sooner would be overall better for the project than later.

**The Carbon experiment has accomplished what it can** to explore industry
interest and participation with limited and targeted outreach to experts. We
have built the critical community and design process and framework to support an
open technical discussion. We have also developed the key language design ideas
sufficiently to show what Carbon would look like clearly and unambiguously. To
broaden or deepen the feedback, we will need frank and public discussion with
wider groups of users and stakeholders. We will need to iterate with a larger
community and a broader set of stakeholders.

**The needs of the C++ community remain unmet, and they are eager** and asking
for exactly Carbon's approach. The technical approach of Carbon directly
addresses concrete and current requests from influential C++ developers. These
needs are specifically not met by C or C++ as they exist or by Rust or other
options, and we have a strategic opportunity to help drive a solution in this
space.

A recent example on Twitter:

> Speaking as a C/C++ developer, when I write in C/C++, I am doing so due to
> ecosystem/practical/legacy code constraints I have no control over, which
> means that being given a superior choice makes no difference to me because I
> didn't get a chance to choose C/C++ to start with
>
> -- [@mcclure111](https://twitter.com/mcclure111/status/1484987221136580610)

A recent HackerNews discussion directly suggested essentially the direction we
are exploring:

> The permanent-ABI compatibility decision was effectively the death-knell for
> C++. Can't improve the standard library performance, making it effectively
> useless for the real world. Can't correct mistakes - the language and library
> warts grow and grow as the years pass by, making it impossible for beginners
> to pick up and learn the language.
>
> -- [lenkite](https://news.ycombinator.com/item?id=31227269)

Last but not least, **the Carbon experiment will accelerate thinking about more
dramatic ways to improve C++ across the industry**.

### Planned process for going public

Our planned process is anchored around a few key principles:

-   Replicate LLVM's open-source approach and strategy. This has proven broadly
    successful over many years.
-   Use existing events and spaces where possible to engage with the C++
    community and industry.
-   Emphasize that it's an experiment, not a finished product.
-   Leverage existing C++ ecosystems rather than creating new ecosystems from
    scratch.

We expect three phases to move things to be public:

-   Get technical components ready: demo, documentation, etc.
-   Do an incremental "road show" with experts, organizations, and WG21 in weeks
    leading up to an official public announcement to seed early awareness.
-   Announce and make everything public at C++North.

During the second phase, we plan to engage with the following individuals and
groups:

-   Industry partners already active in C++ evolution.
-   Influential experts, especially those actively driving C++ evolution and
    prominently writing blog posts or speaking on C++.
-   Community organizations like #include <C++> and Boost.

Our intent is to invite large portions of these groups to join community spaces
to help engage and establish the community culture.

The third phase will be oriented around a keynote at CppNorth:

-   Introduce the audience to the motivation for Carbon and its goals.
-   Explain the project, community, and process.
-   Walk through the core language design so far, and touch on exciting future
    efforts.
-   Provide an extended Q&A to help kick off active discussions, ideally with as
    many of the current Carbon contributors on stage as possible.
-   Will open the GitHub project _read_ ACLs either right before or during the
    talk so that the audience can explore during the Q&A and afterward.
-   Will open the _contribution_ ACLs and Discord Chat at the _end_ so that the
    maximum team bandwidth is available to engage with folks as they begin
    discussing.

#### Public site and communications plan

-   Planning to use a GitHub project and GitHub hosted markdown documentation.
-   Work with existing C++ community blogs to post information about the project
    after announcement.

We are specifically not planning on a website, blog, or other more "official"
presence. This should anchor on Carbon being an experiment and attracting
contributors and participants rather than users.

#### Plan for ACLs once public

GitHub will move to the normal for a public repository:

-   Anyone can create PRs, issues, discussions. This will be fully public.
-   A fairly small number of committers (initially the current set) can merge
    PRs.

Discord will be fully public, using the community features to gate entry on CoC
and CLA.

The shared Google drive and docs:

-   Make viewable publicly.
-   Use the existing group (carbon-lang-contributors@googlegroups.com) to
    provide comment and edit access.
-   New contributors will need to request to join the group to begin
    contributing in this space, but only to avoid spam. This will be a very
    low-risk and trivially handed out access.

Meetings and calendar:

-   As currently, the meeting links would be posted to Discord and anyone there
    can click through and join.
-   The shared calendar and weekly meeting can also be directly made available
    to the same group as the drive.
-   In the future, we can explore enabling live-streaming from the weekly
    meeting and providing a fully public live stream link for those interested.

### Risks and mitigations

Making Carbon public does raise some significant risks that we need to be aware
of and work to mitigate.

#### Too many cooks in the kitchen

-   While opening up to the public, we may get a huge influx of feedback and
    even contributions.
-   This can quickly turn into a situation of too many folks all trying to drive
    Carbon forward in their own direction.

Planned mitigations:

-   We're hoping that many of the critical parts of the language design we've
    prioritized have enough maturity to show a reasonably focused direction that
    folks should focus on to be useful.
    -   We will likely work to avoid immediately reconsidering directions that
        are already well under way to stay focused, capturing any feedback and
        marking it as deferred for a time to revisit.
-   Critical aspects of the language design that are still green fields, we're
    hoping to clearly document as being intentionally deferred in part just to
    reduce the level of churn.
    -   However, we can and should carefully watch for sustained contributors
        who are looking for an area to drive. _That_ needs to be identified and
        enabled. We just want to make sure it can be a focused effort.
-   We should make an extra effort to use the more formal process of raising
    questions for the Carbon Leads to direct the incoming flow. That process is
    set up in a way that should enable it to queue and process things without
    ending up in total chaos.
-   Where possible, teams working on Carbon should start making contingency
    plans to pull in more resources if needed to sustain the engagement.

#### Community management overload

-   We may have a sudden influx of community members, with all of them trying to
    figure out how to collaborate and communicate effectively. This may even
    include publicity and other factors.
-   Responding to and engaging with this will be both time consuming, and
    difficult.
-   It will be tempting for the Carbon Leads to attempt to drive these
    responses, in turn starving other parts of the project.

Planned mitigations:

-   Provide early access to a growing set of people, especially those interested
    and effective and helping with community and communication.
-   Contract with a dedicated community engagement specialist.
    -   Work with them to recruit moderation teams and other scalable groups to
        tackle these issues.
-   Leverage moderation scaling tools across our communication platforms to keep
    the technical work focused and moving.

#### Added distraction or confusion to the C++ evolution process

-   Carbon is a very different direction from the continued C++ evolution in
    WG21.
-   Bringing the C++ community into Carbon and getting them engaged and
    participating will at least reduce the available resources for WG21 in the
    C++ community.
-   The existence, and especially any excitement around such a different
    approach may distract from important work that still needs to get done in
    WG21.

Planned mitigations:

-   Carbon's documentation and all of our public communication should include a
    key message: **If C++ fully meets your needs today, you should keep using
    it.** Carbon is _only_ exploring a direction to address specific concerns
    and problems some users have with C++ today. For other users whose needs are
    fully met, they should stay focused on the thing that exists and works
    today: C++.
-   Carbon is building on top of the C++ ecosystem, and we need to be very clear
    that means it is important to not distract from critical work that needs to
    happen there. We should be active and vocal in supporting folks who are
    continuing to drive WG21 forward.
-   We need to be extremely explicit and clear at every stage that Carbon is
    still an experiment that may not work out.
    -   We do not want critical work that needs to happen for C++ and WG21 to
        stay healthy to be diverted towards Carbon.
-   We plan to engage with committee members actively so they can learn as much
    as possible from the Carbon experiment and incorporate any and all of our
    ideas into C++ where they see a path to do so.

#### Added distractions from existing new programming languages.

-   Another programming language in the world might dilute some of the efforts
    going towards new and exciting but existing languages, especially ones with
    similar performance goals such as Rust.
-   It may also distract users who are unsure of which language to use.

Planned mitigations:

-   Throughout Carbon's documentation and any public communication, we will take
    a specific stance to avoid and minimize this effect:
    -   **If you can use an existing language like Go, Kotlin, or Rust, do so.**
        This experiment is exploring an option for when existing languages
        _don't_ work, particularly due to needing to interoperate with a large
        body of C++ code.
-   We should be careful in the tone whenever discussing existing languages or
    comparing with them, to approach them positively. We should be actively
    supporting other languages, and our focus should be trying to fill a _gap_
    in that ecosystem rather than directly competing.

#### Friction with existing LLVM and Clang communities

While somewhat unlikely, we don't want to create unnecessary friction with the
LLVM and especially the Clang community, as they have heavily invested in a C++
compiler and tooling stack and supporting the C++ language.

Planned mitigations:

-   Need to carefully emphasize that we are build on top of the tremendous work
    done by the LLVM and Clang communities.
    -   LLVM is a critical part of our ability to make a high-performance
        implementation and have no performance overhead when interoperating with
        C++ by using a shared compilation environment.
    -   Clang is the key to our C++ interoperability implementation plans --
        without a production quality frontend with broad cross-platform support
        it would be impossible to achieve the level of interop we need between
        C++ and Carbon.
    -   Many of the insights that have led to Carbon's design come from
        experimentation with Clang and C++ extensions, as well as our experience
        working on the LLVM and Clang codebases themselves.
-   We will clearly and publicly drive improvements to LLVM and Clang upstream
    when needed for Carbon rather than carrying local patches for long
    durations.
-   We will also actively engage members of the LLVM community in the Carbon
    experiment and ensure they are able to participate where interested.
-   Lastly, all Carbon code uses the same license as LLVM so that any of it can
    be immediately merged into LLVM if useful to the broader project or
    community.

#### Labeled as vaporware

-   We are intentionally moving public while Carbon is still very nascent.
-   Many parts of the language still need to be built.
-   We are front-loading design work heavily, which means we have designed
    significantly more than is implemented.

Planned mitigations:

-   Our communication around Carbon will be focused around _building_ the Carbon
    experiment to emphasize both that it is _not_ yet even a complete
    experiment, and that our goal is to get help shaping it rather than suggest
    it is ready for evaluation.
-   We also plan to have clear roadmaps and other artifacts that help give a
    realistic and credible story around the path from where Carbon is to being a
    more complete experiment.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Becoming a fully open project has always been a goal for the project.
    -   Moving public will substantially broaden the feedback we get,
        strengthening our community and directly furthering our goal of being an
        welcoming and inclusive community.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   We need to ensure we can actively engage with users of different C++
        codebases to effectively build a compelling interoperability and
        migration story. This kind of broad engagement is only possible with a
        public project.

## Alternatives considered

### Delay going public until we have working interop demonstrated

An appealing time to go public would be when we have a toolchain that implements
the core language design _and_ provides a demonstration of the banner feature of
Carbon -- C++ interop -- working effectively. If we made this our top priority,
we expect we could complete this in the order of a year.

Advantages:

-   Make the project much more "real" when people first learn of it, and avoid
    the perception of it being vaporware.
-   Avoids spending significant time and effort engaging with a large community
    and industry when the project is new and small. That effort could be
    directed at furthering the design and implementation.

Disadvantages:

-   It would be difficult or impossible to expand the group working on Carbon
    significantly, which will at some point limit the rate at which we can
    execute on the project.
-   We would be have significantly less broad input on the design of Carbon
    itself and especially of its C++ interop, risking that the design would not
    actually address the needs of the broader industry.
-   We wouldn't be able to build Carbon in the open and develop trust of the
    community and industry.
-   We wouldn't be able to involve the broader community and industry in the
    design and development of Carbon, which might lessen both interest and
    enthusiasm.
-   We won't start to understand how broadly the industry is interested in the
    Carbon direction until significantly more effort has been invested.

The desire to build Carbon in the open, develop strong trust with the community,
and understand the breadth of industry interest outweigh the costs and risks of
making Carbon public earlier.

### Delay going public until Carbon is a compelling option for C++ developers to adopt

We could delay still further and build Carbon into a compelling and ready-to-use
programming language before going public.

Advantages:

-   This would in some ways be the most _efficient_ strategy, as it would keep
    everyone working on Carbon focused without distraction.
-   We would have the best possible first impression on developers by having a
    largely complete and high quality language ready to go.

Disadvantages:

-   We still wouldn't know whether the experiment was a success -- both
    developers and the industry at large might not be interested in Carbon. The
    result is that this would have the maximum sunk cost at the point where we
    begin to learn about the industry interest.
-   It is unlikely we could sustain the effort required to reach this point
    without any incremental milestones.
-   We would likely have very little trust from the community due the prolonged
    secrecy.
-   Given the timeline, it is very likely that Carbon would leak and lose some
    of the advantages.
-   This would maximize the risk of some aspect of Carbon being designed in a
    way that doesn't meet the larger industry needs due to a lack of feedback.
    It also maximizes the cost of correcting this once discovered.

This has never been seen as a realistic approach for Carbon, and that doesn't
seem to have changed. While it does have some advantages, the tradeoff seems
sharply wrong for the needs of the Carbon project.
