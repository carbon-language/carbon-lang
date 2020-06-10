# Criterion for Carbon to go Public

## Overview

Open, transparent, and public development is often the best way to build a
  developer community.
Long term evolution of Carbon behind closed doors does not align with our core
  principles.

However, there are risks associated with going public too soon.
We only get one chance to make a first impression.
At the end of the day, Carbon's success does not rely solely on technical merit.
Messaging, marketing, and branding are also vital to our success.
We must not only build a technically excellent language, we must also tell the
  right narrative about that language.

Additionally, there are costs associated with going public.
Greater participation brings great administrative and logistical overheads.

We propose that we should only go public when:

* There is a critical mass of Carbon evolutionary decisions that require
    field experience with a wider audience.
* We are ready to make long term investments in Carbon.
* Carbon has support from multiple organizations and individuals.
* We have a prototype Carbon implementation.
* We can demonstrate Carbon's potential.
* We have a launch event and learning material.
* We are prepared to handle a large influx of new contributions and feedback.

## Criteria

### Criteria: Broader Field Experience Required

As Carbon matures, we will increasingly find that we require field experience
  to inform the Carbon evolutionary process.

Field experience is a combination of:

* Implementation Experience: The experience and feedback gained by implementing
      a design, process, or idea.
    We should be able to obtain much of the implementation experience we need
      without going public.
* Usage Experience: The experience and feedback gained by using an
      implementation of a design, process, or idea.
    We will be able to obtain some usage experience without going public, but
      the diversity of that experience may be limited.
    Our usage experience will also be biased our involvement in the design and
      implementation of Carbon.
    We will be more likely to use things in the way that the designers and
      implementers intended.
    To ensure our designs are robust, we need to experience the Hyrum's Law
      effects that come from broad and surprising usage of Carbon.
    This can only truly be achieved through a public developer community.
* Deployment Experience: Deployment experience is usage experience over time as
      the underlying designs and implementations change and tooling-assisted
      migration is utilized. 
    Because time is required, the sooner we get usage experience, which requires
      a public developer community, the sooner we will have deployment
      experience.

At first, few aspects of Carbon's evolution will be blocked on usage and
  deployment experience.
As Carbon ages and we obtain implementation experience, we will start to find
  that we cannot make progress or build confidence in certain things without
  wider usage and deployment experience.

As we evolve Carbon, we should be consider and documenting where we would
  benefit from or be blocked by field experience that requires going public.
We should record these needs in terms of concrete open questions that need to
  be addressed by field experience.
We should regularly review this set of open questions.
At some point, the value of addressing this set of open questions via public
  field experience will become significant enough to warrant the costs
  associated with going public.

### Criteria: Ready to Make Long Term Investments

The community will not distinguish between a plan and promise.
The community will not distinguish between a exploration and a long term
  commitment.

Once we go public, it will be assumed that we are making a long term investment
  in Carbon.
It is unlikely that we will be able to avoid that assumption through any amount
  of messaging.

Therefore, we should only go public when we are 100% confident that we will
  invest in Carbon long term.
To gain such confidence, we will need to prove to ourselves and our
  organizations that Carbon delivers on its goals.

### Criteria: Support from Multiple Organizations and Individuals

We have a desire to make Carbon a collaboration of multiple organizations and
  individuals to ensure the longevity and broad applicablity of Carbon.
However, Carbon initially began as a project within a single organization.
Thus, we can expect that in Carbon's infancy the vast majority of contributions
  will come from the initial contributing organization.

When Carbon goes public, we do not want it to be perceived as an initiative
  of a single organization.
Therefore, we should consider the diversity of participation and the number of
  organizations willing to make long term investments in Carbon before going
  public.
If the diversity is insufficient, then we should continue increasing it by
  inviting additional parties to participate.

### Criteria: Prototype Implementation

We need to have a prototype implementation to go public.
We will be unable to build excitement and a user base without a prototype 
  implementation, and we will be unable to get the valuable field experience
  that justifies the adminstrative and logistical costs of going public.

We need to develop a set of requirements and schedule for a minimal viable
  public release of the language and implementation.
Determining that set of requirements is outside of the scope of this document.

### Criteria: Demonstration of Potential

When we go public, we should be able to prove that Carbon can deliver on its 
  goals.
That means that we need to have a set of demonstrations of Carbon's potential.

This will likely include both "micro" demonstrations (specific examples that
  highlight particular capabilities) and "macro" demonstrations of applications
  and libraries implemented in Carbon.

### Criteria: Launch Event and Learning Material

We should have a compelling launch event when we go public and supporting
  content to encourage early adopters.

For the launch event, we should have:
* Technical talks from multiple speakers and organizations.
  This could be at a conference or in our own remote or physical event.
* A website.
* A social media game plan.
* Tech press lined up.

For learning material, we should have:
* Documentation.
* Blog posts.
* Examples.

### Criteria: Prepared for New Contributions and Feedback

When we launch, we should expect to receive many new contributions, feedback,
  and bug reports.
To successfully build a developer community, we need to have mechanisms in place
  to recieve all that input be responsive to it.

Prior to the launch, we should review all of our processes and systems and
  ensure they are ready and suitable for a large influx of new contributors.
We should also ensure that the Carbon team is prepared to commit time to
  responding to new contributors in the days and weeks after the launch.

## Risks

We must recognize a few unfortunate realities about how Carbon may be perceived
  when it goes public.
These present a risk to Carbon.

If Carbon goes public too soon or without the right narrative, Carbon, Carbon's
  goals, and Carbon's principles may become misperceived.
Such misperceptions could be fatal to Carbon despite technical merit.

During our launch, we must work hard to address and minimize these risks.

### Risk: Relationship to C++

It is very likely that Carbon will be perceived as competitor to C++.
Some will view Carbon as an indictment of C++ and those who have dedicated
  their career to developing C++.
Those who work on Carbon, especially those who also currently or previously
  work on C++, may be perceived in a negative light by the C++ community.

We must make every effort to not create a hostile relationship between the Carbon
  and C++ community.
One step that we take to avoid that is to ensure that we clearly acknowledge and
  thank the C++ community and language designers
After all, much of the experience that is driving the creation of Carbon comes
  from C++.

### Risk: Perception of Ownership by a Single Organization

Carbon may be perceived as being owned and pushed by a single organization.
This may lead to a variety of different claims of nefarious intent in the
  development of Carbon.

## Leak Contigencies

We should strive to control the narrative and timeline of launching Carbon.
Having the decision of going public forced upon us by leaks would be unfortunate.
We should err on the side of not letting leaks influence our launch.

Over time, some leaks are going to be inevitable.
The closer that we get to our planned launch, the greater the leak risk will be.
It is important for us to distinguish between minor leaks which do not warrant
  going public and major leaks which do.

### Minor Leak

A minor leak is an unintentional disclosure of Carbon which does not have the
  potential for exponential growth in the number of people aware of Carbon.
In a minor leak, most or all parties are friendly to us and are unlikely to
  intentionally spread information about Carbon against our wishes.

Examples:
* Accidentally mentioning Carbon to someone you believed was already aware of it.
* Accidentally sending an email about Carbon to C++ committee list.
* Accidentally mentioning Carbon on social media and then deleting the mention
    shortly thereafter.

Minor leaks should not force us to prematurely go public.

### Major Leak

A major leak is an disclosure of Carbon which has the potential to exponentially
  grow the number of people aware of Carbon.
Major leaks will typically involve either substantial social media or tech
  press exposure outside of our control.

Examples:
* A tech press site learns about Carbon and decides to publish a news article
    about on it.
* Someone aware of Carbon discloses information about it on social media and
    draws substantial attention.

In the event of a major leak, we should consider the efficacy of going public
  prematurely so that we can control the narrative.
