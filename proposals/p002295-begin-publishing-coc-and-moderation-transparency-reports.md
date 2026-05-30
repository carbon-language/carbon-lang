# Begin publishing CoC and moderation transparency reports

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2295)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Automated dashboards](#automated-dashboards)
    -   [Different cadence](#different-cadence)
    -   [Different publishing options](#different-publishing-options)

<!-- tocstop -->

## Abstract

Proposes a documented cadence, publishing target, and template for Code of
Conduct and moderation transparency reports.

## Problem

The Carbon community wants to be an example in terms of diversity, equity and
inclusion. One of the best practices that supports such ambition is publishing
transparency reports about misconduct within the community and actions taken in
response. There is no previous experience with reporting back to the community
about this so far.

## Background

Transparency reports are not very common practice, but already around. So there
are some ideas to grab from other communities, with their permission.

## Proposal

We will benchmark relevant transparency reports to find inspiration to create
our own. We will look at their scope, their format, their regularity and
possibly community feedback we have access to.

We will create a template for Carbon's transparency reports and maintain that as
part of the project. Our Code of Conduct team will then create and publish
regular reports in the GitHub discussion forum based on that template.

## Details

Some examples of transparency reports in which conduct incidents had been
reported:

Conferences:

-   PyCon US 2022:
    https://pycon.blogspot.com/2022/06/pycon-us-2022-transparency-report.html
-   CppCon 2021: https://cppcon.org/cppcon-2021-transparency-report/
-   PyConDE 2019:
    https://2019.pycon.de/blog/code-of-conduct-transparency-report/

Other:

-   LLVM CoC July 15, 2022: https://llvm.org/coc-reports/2022-07-15-report.html
-   Linux Foundation 2021:
    https://www.linuxfoundation.org/blog/blog/linux-foundation-events-code-of-conduct-transparency-report-2021-event-summary

We propose a [template](/docs/project/transparency_reports.md#template) for our
own CoC transparency report, which has been greatly inspired by transparency
reports issued by the Python community worldwide. Thank you, Pythonistas!

We also document there the
[cadence and target of publishing](/docs/project/transparency_reports.md#cadence-and-publishing).

## Rationale

Transparent moderation and handling of conduct issues is essential to sustaining
a healthy [community and culture](/docs/project/goals.md#community-and-culture),
one of Carbon's primary goals. We need to clearly and transparently surface the
conduct of our community and how we respond to it.

## Alternatives considered

### Automated dashboards

An automated dashboard (for example “X days since AutoMod detected a “guys” in
our Discord”, “Y incidents of harmful language”) would be great, but we are
still figuring out the basics and principles together before we can automate
more.

Such a dashboard could be an attractive alternative for the future.

### Different cadence

Publishing quarterly may result in some quarters with very little or even
nothing to publish. That seems fine. We considered a slower cadence of annually,
but we worry that would be too slow for folks in the community to hold the
conduct team and leadership accountable as things start to be forgotten.

We also thought about how to respond rapidly when needed but would prefer to
publish out-of-band when needed rather than run an even more rapid cadence.

### Different publishing options

We considered different places to publish the individual transparency reports.

They don't seem to belong as part of our version controlled repository as they
aren't something that should be persistent -- they reflect a report at a moment
in time. And while we hope to never need it, we should retain the ability to
easily redact information or make permanent edits to them if necessary. GitHub
discussions seem like a good fit overall compared to alternatives like the main
repository, the wiki, etc.

Using GitHub discussions also allows comments and discussion where folks could
for example ask questions or get more information about how and why we are
taking our moderation approaches. While there is some risk of these discussions
being unproductive, we hope to be able to moderate them as well and still
provide a place where productive and on-topic discussion can still take place.
