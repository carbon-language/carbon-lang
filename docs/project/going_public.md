# Criteria for Carbon to go public

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Criteria](#criteria)
    -   [Broader field experience required](#broader-field-experience-required)
    -   [Sustained interest to use Carbon from multiple organizations and individuals](#sustained-interest-to-use-carbon-from-multiple-organizations-and-individuals)
    -   [Prototype implementation](#prototype-implementation)
    -   [Demonstration of potential](#demonstration-of-potential)
    -   [Learning material](#learning-material)
    -   [Prepare for new contributions and feedback](#prepare-for-new-contributions-and-feedback)
    -   [Launch event](#launch-event)
-   [Risks](#risks)
    -   [Relationship to C++](#relationship-to-c)
    -   [Perception of ownership by a single organization](#perception-of-ownership-by-a-single-organization)
    -   [Over-promising](#over-promising)
-   [Leak contingencies](#leak-contingencies)
    -   [Minor leaks](#minor-leaks)
    -   [Major leaks](#major-leaks)

<!-- tocstop -->

## Overview

Open, transparent, and public development is often the best way to build a
developer community. Long-term evolution of Carbon behind closed doors does not
align with our core principles.

However, there are risks associated with going public too soon. We only get one
chance to make a first impression. At the end of the day, Carbon's success does
not rely solely on technical merit. Messaging, marketing, and branding are also
vital to Carbon's success. Carbon must not only be a technically excellent
language, we must also tell the right narrative about Carbon.

Additionally, there are costs associated with going public. Greater
participation brings greater administrative and logistical overheads.

Carbon can be expected to go public when most or all criteria are satisfied:

-   A critical mass of Carbon evolutionary decisions need broader field
    experience to determine the right choice.
-   Sustained interest to use Carbon from multiple organizations and
    individuals.
-   Carbon has a prototype implementation.
-   Carbon's potential can be demonstrated.
-   Learning material is available for Carbon.
-   Community members are prepared to handle a large influx of new contributions
    and feedback.
-   The community is ready to run a launch event.

These criteria are guidelines. Any proposal to go public with Carbon, including
whether criteria are sufficiently met, will be reviewed as with other evolution
proposals.

## Criteria

### Broader field experience required

We expect many of Carbon's evolution decisions to be based on our experiences
with and knowledge of other programming languages, particularly C++. As Carbon
matures, we expect some decisions will be made where we are making a temporary
choice pending broader field experience. For example, we may need to see how
Carbon is used in order to understand the best approach for a feature.

Field experience is a combination of:

-   Implementation Experience: The experience and feedback gained by
    implementing a design, process, or idea. We should be able to obtain much of
    the implementation experience we need without going public.
-   Usage Experience: The experience and feedback gained by using an
    implementation of a design, process, or idea.
    -   We will be able to obtain some usage experience without going public,
        but the diversity of that experience may be limited. Our usage
        experience will also be biased by our involvement in the design and
        implementation of Carbon. We will be more likely to use things in the
        way that the designers and implementers intended.
    -   To ensure our designs are robust, we need to experience the Hyrum's Law
        effects that come from broad and surprising usage of Carbon. This can
        only truly be achieved through a public developer community.
-   Deployment Experience: Deployment experience is usage experience over time,
    as the underlying designs and implementations change and tooling-assisted
    migration is utilized. Because time is required, the earlier we expand our
    usage experience, the quicker we will be able to expand our deployment
    experience.

At first, few aspects of Carbon's evolution will be blocked on usage and
deployment experience. As Carbon ages and we obtain implementation experience,
we will start to find that we cannot make progress or build confidence in
certain things without wider usage and deployment experience.

As Carbon evolves, we should consider and document where we would benefit from
or be blocked on field experience that requires going public. We should record
these needs in terms of concrete open questions that need to be addressed by
field experience. We should regularly review this set of open questions. At some
point, the value of addressing this set of open questions by way of public field
experience will become significant enough to warrant the costs associated with
going public.

### Sustained interest to use Carbon from multiple organizations and individuals

We should ensure there is sustained interest to continue with Carbon, even if
circumstances later shift. This interest should come from multiple organizations
and individuals. It is important that Carbon be a collaboration in order to
ensure its longevity and broad applicability. Even if most of Carbon's early
contributions come from one organization, it remains crucial that others have
sufficient interest to contribute to its ecosystem.

### Prototype implementation

Carbon must have a prototype implementation to go public. We will be unable to
build excitement and a user base without a prototype implementation, and we will
be unable to get the valuable field experience that justifies the administrative
and logistical costs of going public.

We need to develop a set of requirements and schedule for a minimal viable
public release of the language and implementation. Determining that set of
requirements is outside of the scope of this document.

### Demonstration of potential

Carbon should only go public if it's showing promise towards delivering on its
goals. That means that we need to have a set of demonstrations of Carbon's
potential.

This will likely include both "micro" demonstrations (specific examples that
highlight particular capabilities) and "macro" demonstrations of applications
and libraries implemented in Carbon.

### Learning material

Learning material is critical for making it easy for newcomers to understand
Carbon, how it is differentiated from other programming languages, and our plans
for the future. For example:

-   Documentation, including a specification.
-   User guides, including an introduction targeting new engineers.
-   Example code.
-   Blog posts.

### Prepare for new contributions and feedback

When we launch, we should expect to receive many new contributions, comments,
and bug reports. To successfully build a developer community, we need to have
mechanisms in place to receive all that input and be responsive to it.

Prior to the launch, we should review all of our processes and systems and
ensure they are ready and suitable for a large influx of new contributors. We
should also ensure that members of the Carbon community are prepared to commit
time to assist new contributors in the days and weeks after the launch.

### Launch event

We should have a compelling launch event plan when we go public. For example:

-   A website.
-   Technical talks from multiple speakers and organizations. This could be at a
    conference or in our own remote or physical event.
-   A game plan for social media and tech press.

## Risks

We recognize a few unfortunate realities about how Carbon may be perceived when
it goes public. These present a risk to Carbon.

If Carbon goes public too soon or without the right narrative, Carbon, Carbon's
goals, and Carbon's principles may become misperceived. Such misperceptions
could undermine the success of Carbon, regardless of technical merit. During our
launch, we must work hard to address and minimize these risks.

### Relationship to C++

It's important for the Carbon and C++ communities to work together effectively,
given that Carbon is trying to provide a path forward for some (although not
all) users of C++.

We must take steps during the development of Carbon to ensure that multiple
organizations are involved. We will ensure that launch announcements clearly
acknowledge and thank the individuals and organizations which have been involved
during its private development.

### Perception of ownership by a single organization

Carbon may be perceived as being owned and pushed by a single organization. We
must take steps during the development of Carbon to ensure that multiple
organizations are involved. We will ensure that launch announcements clearly
acknowledge and thank the individuals and organizations which have been involved
during its private development.

### Over-promising

While Carbon is private, it's easy to have discussions with everyone involved to
explain its experimental nature, and avoid having anybody become overly reliant
while the future is still in doubt. Once Carbon goes public, it's likely that
regardless of what we say, some developers will expect that the experiment won't
be abandoned.

It's important that the launch plans and communications clearly and accurately
communicate the experimental nature of Carbon, and discourage viewing it as a
"ready-to-go" system. There are things we might do later, such as training
classes, which may mislead developers at an early stage and thus should be
approached with caution. The degree to which we encourage developers to use
Carbon should be commensurate with risks of Carbon being abandoned.

## Leak contingencies

Community members should strive to control the timeline and narrative for
launching Carbon. Having the decision of going public forced upon us by leaks
would be unfortunate. We should err on the side of not letting leaks influence
our launch.

Over time, some leaks are going to be inevitable. The closer that Carbon gets to
a planned launch and the larger the community is Carbon, the greater the risk of
leaking. It is important for us to distinguish between minor leaks which do not
warrant a response, and major leaks which do.

### Minor leaks

A minor leak is an unintentional disclosure of Carbon which does not have the
potential for exponential growth in the number of people aware of Carbon. In a
minor leak, most or all parties are friendly to us and are unlikely to
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
grow the number of people aware of Carbon. Major leaks will typically involve
either substantial social media or tech press exposure outside of our control.
For example, if a tech press site publishes an article about Carbon.

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
