# Criteria for Carbon to go public

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Criteria](#criteria)
    -   [Broader field experience required](#broader-field-experience-required)
    -   [Ready to make long-term commitments](#ready-to-make-long-term-commitments)
    -   [Long-term commitments from multiple organizations and individuals](#long-term-commitments-from-multiple-organizations-and-individuals)
    -   [Prototype implementation](#prototype-implementation)
    -   [Demonstration of potential](#demonstration-of-potential)
    -   [Learning material](#learning-material)
    -   [Prepare for new contributions and feedback](#prepare-for-new-contributions-and-feedback)
    -   [Launch event](#launch-event)
-   [Risks](#risks)
    -   [Relationship to C++](#relationship-to-c)
    -   [Perception of ownership by a single organization](#perception-of-ownership-by-a-single-organization)
-   [Leak contingencies](#leak-contingencies)
    -   [Minor leaks](#minor-leaks)
    -   [Major leaks](#major-leaks)

<!-- tocstop -->

## Overview

Open, transparent, and public development is often the best way to build a
developer community.

Long-term evolution of Carbon behind closed doors does not align with our core
principles.

However, there are risks associated with going public too soon.

We only get one chance to make a first impression.

At the end of the day, Carbon's success does not rely solely on technical merit.

Messaging, marketing, and branding are also vital to Carbon's success.

Carbon must not only be a technically excellent language, we must also tell the
right narrative about Carbon.

Additionally, there are costs associated with going public.

Greater participation brings greater administrative and logistical overheads.

Carbon will go public when most or all criteria are satisfied:

-   A critical mass of Carbon evolutionary decisions need broader field
    experience to determine the right choice.
-   Community members are ready to make a long-term commitment to Carbon.
-   Carbon has support from multiple organizations and individuals.
-   Carbon has a prototype implementation.
-   Carbon's potential can be demonstrated.
-   Learning material is available for Carbon.
-   Community members are prepared to handle a large influx of new contributions
    and feedback.
-   The community is ready to run a launch event.

Any proposal to go public will be reviewed as with other evolution proposals.

## Criteria

### Broader field experience required

We expect many of Carbon's evolution decisions to be based on our experiences
with and knowledge of other programming languages, particularly C++.

As Carbon matures, we expect some decisions will be made where we are making a
temporary choice pending broader field experience.

For example, we may need to see how Carbon is used in order to understand the
best approach for a feature.

Field experience is a combination of:

-   Implementation Experience: \
    The experience and feedback gained by implementing a design, process, or idea.
    \
    We should be able to obtain much of the implementation experience we need
    without going public.
-   Usage Experience: \
    The experience and feedback gained by using an implementation of a design, process,
    or idea. _ We will be able to obtain some usage experience without going public,
    but the diversity of that experience may be limited. \
    Our usage experience will also be biased by our involvement in the design
    and implementation of Carbon. \
    We will be more likely to use things in the way that the designers and implementers
    intended. _ To ensure our designs are robust, we need to experience the Hyrum's
    Law effects that come from broad and surprising usage of Carbon. \
    This can only truly be achieved through a public developer community.
-   Deployment Experience: \
    Deployment experience is usage experience over time, as the underlying designs
    and implementations change and tooling-assisted migration is utilized. \
    Because time is required, the earlier we expand our usage experience, the
    quicker we will be able to expand our deployment experience.

At first, few aspects of Carbon's evolution will be blocked on usage and
deployment experience.

As Carbon ages and we obtain implementation experience, we will start to find
that we cannot make progress or build confidence in certain things without wider
usage and deployment experience.

As Carbon evolves, we should consider and document where we would benefit from
or be blocked by field experience that requires going public.

We should record these needs in terms of concrete open questions that need to be
addressed by field experience.

We should regularly review this set of open questions.

At some point, the value of addressing this set of open questions via public
field experience will become significant enough to warrant the costs associated
with going public.

### Ready to make long-term commitments

Once Carbon goes public, it will be infeasible to guarantee that everyone
understands Carbon to be an idea for an experiment, rather than a promise with a
long-term commitment.

If we end up abandoning the Carbon experiment, some users will inevitably view
the experience in a highly negative manner.

Even with a private community, we must consider the balance of where the
experiment needs long-term commitments to continue advancing.

Carbon should only go public when the core team is fully confident that there
are sufficient community members ready to make a long-term commitment to Carbon.

To gain such confidence, community members will need evidence for themselves and
their organizations that Carbon delivers on its goals.

### Long-term commitments from multiple organizations and individuals

We have a desire to make Carbon a collaboration of multiple organizations and
individuals to ensure the longevity and broad applicability of Carbon.

However, Carbon initially began as a project within a single organization.

Thus, we can expect that in Carbon's infancy the vast majority of contributions
will come from the initial contributing organization.

When Carbon goes public, we do not want it to be perceived as an initiative of a
single organization.

Therefore, we should consider the diversity of participation and the number of
organizations willing to make long-term commitments in Carbon before going
public.

Some individuals must be ready to contribute to Carbon's design and
implementation, but that's not the only kind of long-term commitment that we
need.

Adoption of Carbon also offers a way to build crucial field experience and
generate feedback.

If the diversity is insufficient, then we will work to increase it by inviting
additional parties to participate.

### Prototype implementation

Carbon must have a prototype implementation to go public.

We will be unable to build excitement and a user base without a prototype
implementation, and we will be unable to get the valuable field experience that
justifies the administrative and logistical costs of going public.

We need to develop a set of requirements and schedule for a minimal viable
public release of the language and implementation.

Determining that set of requirements is outside of the scope of this document.

### Demonstration of potential

Carbon should only go public once it can be proven that it can deliver on its
goals.

That means that we need to have a set of demonstrations of Carbon's potential.

This will likely include both "micro" demonstrations (specific examples that
highlight particular capabilities) and "macro" demonstrations of applications
and libraries implemented in Carbon.

### Learning material

Learning material is critical for making it easy for newcomers to understand
Carbon, how it is differentiated from other programming languages, and our plans
for the future. We should have:

-   Documentation, including a specification.
-   User guides, including an introduction targeting new engineers.
-   Example code.
-   Blog posts.

We don't expect to offer training courses at this point, as the Carbon prototype
is still expected to be relatively immature.

### Prepare for new contributions and feedback

When we launch, we should expect to receive many new contributions, comments,
and bug reports.

To successfully build a developer community, we need to have mechanisms in place
to receive all that input and be responsive to it.

Prior to the launch, we should review all of our processes and systems and
ensure they are ready and suitable for a large influx of new contributors.

We should also ensure that members of the Carbon community are prepared to
commit time to assist new contributors in the days and weeks after the launch.

### Launch event

We should have a compelling launch event when we go public. We should be ready
with:

-   Technical talks from multiple speakers and organizations. \
    This could be at a conference or in our own remote or physical event.
-   A website.
-   A social media game plan.
-   Tech press lined up.

## Risks

We recognize a few unfortunate realities about how Carbon may be perceived when
it goes public.

These present a risk to Carbon.

If Carbon goes public too soon or without the right narrative, Carbon, Carbon's
goals, and Carbon's principles may become misperceived.

Such misperceptions could undermine the success of Carbon, regardless of
technical merit.

During our launch, we must work hard to address and minimize these risks.

### Relationship to C++

It is possible that Carbon will be perceived as a competitor to C++.

Some may view Carbon as an indictment of C++ and those who have dedicated their
career to developing C++.

Those who work on Carbon, especially those who also currently or previously work
on C++, may be perceived in a negative light by the C++ community.

We must make every effort to not create a hostile relationship between the
Carbon and C++ communities.

One step that we will take to avoid that is to ensure that launch announcements
clearly acknowledge and thank the C++ community and language designers.

After all, much of the experience that is driving the creation of Carbon comes
from C++ and an appreciation of its advantages over other programming languages.

### Perception of ownership by a single organization

Carbon may be perceived as being owned and pushed by a single organization.

This may lead to a variety of different claims of nefarious intent in the
development of Carbon.

We must take steps during the development of Carbon to ensure that multiple
organizations are involved.

We will ensure that launch announcements clearly acknowledge and thank the
individuals and organizations which have been involved during its private
development.

## Leak contingencies

Community members should strive to control the timeline and narrative for
launching Carbon.

Having the decision of going public forced upon us by leaks would be
unfortunate.

We should err on the side of not letting leaks influence our launch.

Over time, some leaks are going to be inevitable.

The closer that Carbon gets to a planned launch and the larger the community is
Carbon, the greater the risk of leaking.

It is important for us to distinguish between minor leaks which do not warrant a
response, and major leaks which do.

### Minor leaks

A minor leak is an unintentional disclosure of Carbon which does not have the
potential for exponential growth in the number of people aware of Carbon.

In a minor leak, most or all parties are friendly to us and are unlikely to
intentionally spread information about Carbon against our wishes.

For example:

-   Accidentally mentioning Carbon to someone you believed was already aware of
    it.
-   Accidentally mentioning Carbon in an email to a C++ committee list.
-   Accidentally mentioning Carbon on social media and then deleting the mention
    shortly thereafter.

Minor leaks should be avoided, but will generally be ignored.

### Major leaks

A major leak is a disclosure of Carbon which has the potential to exponentially
grow the number of people aware of Carbon.

Major leaks will typically involve either substantial social media or tech press
exposure outside of our control.

For example:

-   A tech press site learns about Carbon and decides to publish a news article
    about it.

    -   If a reporter has rumors, email leaks, or documentation leaks, community
        members should not offer comments.

        We want to avoid having comments accidentally substantiate rumors and
        result in an article.

-   Someone aware of Carbon discloses information about it on social media and
    draws substantial attention.

In the event of a major leak, the core team will critically evaluate possible
next steps:

-   If the Carbon experiment isn't showing enough promise to continue, a major
    leak may lead to development stopping.
-   Give permission to a member of the Carbon community to comment in order to
    defuse the situation.
-   Take Carbon public prematurely.

This will be based on considerations such as diminished value of remaining
private, as well as the value of adding more information to a dialogue that may
help preserve Carbon's reputation.
