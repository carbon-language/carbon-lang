# Guidance on AI coding tools

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5670)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Place more restrictions around AI-based tool usage when contributing](#place-more-restrictions-around-ai-based-tool-usage-when-contributing)

<!-- tocstop -->

## Abstract

Establish some guidance on using AI coding tools when contributing to the Carbon
Language project. These tools have growing popularity and interest, and it would
be good to have a clear and actively documented set of guidance for folks
interested or already using them.

## Problem

AI-based coding tools are wildly popular at this point and we should have active
guidance about how and when to use them when contributing to Carbon rather than
being reactive.

## Background

-   [LLVM Developer Policy guidance on AI generated code](https://llvm.org/docs/DeveloperPolicy.html#ai-generated-contributions)

## Proposal

All submissions to Carbon need to follow our Contributor License Agreement
(CLA), in which contributors agree that their contribution is an original work
of authorship. This doesn’t prohibit the use of coding assistance tools, but
what’s submitted does need to be a contributor’s original creation.

Carbon's license was also selected specifically to maintain full compatibility
between any contributions to Carbon and contributions to LLVM so that we can
potentially move things between these projects easily. We want the same to be
true regarding the use of AI-based coding tools, and so any contributions to
Carbon should also abide by the guidance in the
[LLVM Developer Policy around AI generated code](https://llvm.org/docs/DeveloperPolicy.html#ai-generated-contributions)

This proposal updates our contributing documentation to contain both these
points.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Explicitly documenting what is and isn't required when contributing to
        Carbon makes the project more open and welcoming to new contributors.

## Alternatives considered

### Place more restrictions around AI-based tool usage when contributing

We could consider adopting more restrictions on how AI-based tools can be used
as part of contributing to Carbon. However, at this point we don't have a
compelling rationale for this.

There are real concerns around the quality of code output from AI-based tools in
some cases. However, we should not rely on avoiding the tools to provide
protection from low-quality code. Instead, as we already do, we have a system of
code review, coding standards, and extensive testing to ensure high-quality
contributions regardless of the tools used. **The quality, correctness, and
utility expectations of contributions are true regardless of which tools are
used as part of authoring the contribution.**

We have also seen abuse where large volumes of automatically generated
"contributions" have been sent to projects, overwhelming their community. Again,
this is never OK, regardless of which tools are used to achieve it. We hold the
contributors responsible for using any and all tools responsibly and making
useful and constructive contributions with them.

Last but not least, we also understand that there may contributors who do not
wish to use these tools. Currently, we are not proposing any tools in the
required workflow of the project.
