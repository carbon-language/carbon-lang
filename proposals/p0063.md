# Criteria for Carbon to go public

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/63)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Alternatives considered](#alternatives-considered)
-   [Rationale](#rationale)

<!-- tocstop -->

## Problem

Carbon is currently private, and should have a clear plan for going public.

## Background

Some of this is explained to people when they're asked if they'd like to
contribute to Carbon; the purpose of this proposal is to provide documentation.

## Proposal

See /docs/project/going_public.md.

## Alternatives considered

We have also considered going public immediately. We believe the noted criteria
are important to address before proceeding.

## Rationale

"An open, inclusive process for Carbon changes" requires us to make the decision
to go public in a clear and unsurprising way, with criteria that are written
down.

Going public too early introduces risks to the long-term evolution and
maintenance of the language by increasing the costs of the community members
developing it.

Ensuring that Carbon meets its functional goals, especially that of
interoperability and migration, will inherently require large scale
experimentation that is infeasible to do without becoming public at some point.

The core team is aligned on the core policy decisions, which are:

-   The decision to go public will go through the usual proposal process.
-   The structure of the proposal gives the things that we are looking for prior
    to going public.
-   We expect to delay announcing to the extent we can.
-   We do not expect that to be so late that everything is done and we are ready
    to ship. That is, we are not going to wait until version 1.0 is ready.
    -   We don't expect this to be so late that Carbon is no longer an
        experiment.

Changes to the text and wording that align with these items should be submitted
as code reviews. The core team members chandlerc and zygloid will both approve
each code review.

An example change that would be covered by a code review: there will be no
automatic going public just because the criteria are met -- it will be a
decision of the core team.
