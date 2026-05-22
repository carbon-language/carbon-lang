---
name: Proposals
description:
    Instructions for writing, submitting, and managing Carbon evolution
    proposals.
---

# Proposals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Overview

This skill provides instructions and best practices for working with proposals.
Only create a proposal when explicitly directed as part of the task.

Make sure to confirm the desired title for the proposal as that will govern the
filename.

## Create a new proposal

1.  **Use the helper script**: Run `./proposals/scripts/new_proposal.py "Title"`
    to create a templated file and instructions for setting up the PR.
2.  **Proposal file**: The file will be named `proposals/p######-title.md`,
    where `######` is the 6-digit GitHub pull request number and `title` is a
    slugified version of the proposal title.
3.  **Template**: Follow the structure in `proposals/scripts/template.md`,
    noting the specific `TODO` instructions in each section for the content that
    should be included there.
4.  **PR description**: Update the pull request description to match the
    abstract section in the proposal document.

> [!IMPORTANT] Do _not_ mark the PR ready for review. The user must have the
> opportunity to review the proposal produced before asking for any review.

## Writing style and best practices

-   **Skimmable**: Use
    [BLUF](<https://en.wikipedia.org/wiki/BLUF_(communication)>) (Bottom Line Up
    Front) or
    [Inverted Pyramid](<https://en.wikipedia.org/wiki/Inverted_pyramid_(journalism)>)
    style. Keep it brief, focused, and technical.
-   **Match existing proposal style**: Review [existing proposals](/proposals)
    (preferring more recent ones with higher numbers) to understand the expected
    style, wording, and nature of content to include.
-   **Connect to goals**: In the Rationale section, link to specific goals in
    [`/docs/project/goals.md`](/docs/project/goals.md) and principles in
    [`/docs/project/principles`](/docs/project/principles) (e.g.,
    `error_handling.md`, `one_way.md`).
-   **Living design**: If the proposal updates design documentation, include
    those changes in the PR if possible. If deferred, add "TODO" comments
    pointing to the proposal (e.g.,
    `> **TODO:** Document ... adopted in [p######](/proposals/p######-title.md)`).
    For pervasive changes, file a GitHub issue instead of adding many TODOs.

## Alternatives considered and leads decisions

There are always alternatives to a proposal, and the proposal should carefully
include sections describing all of them and the rationale for not selecting
them. Any living design document updates should focus on fully describing the
end-state design, and the key motivating aspects of that design. The main
proposal should focus on _what is changing_ and _why it is changing_, and should
leave detailed description of the resulting design to the living design
document, and _why not_ rationale to the description of each alternative.

-   **Cover all the alternatives**: Make sure to describe any alternatives
    considered, even if minor or rejected early.
-   **Be specific**: Don't be vague about any of the alternatives, or the
    rationale for not choosing them.
-   **Connect to goals or principles**: In addition to the rationale section,
    one of the best rationale structures for rejecting an alternative connects
    that choice back to the goals or principles relevant.
-   **Always frame as a tradeoff**: Selecting the proposed direction instead of
    an alternative is _always_ a tradeoff, with both advantages and
    disadvantages.
-   **Where relevant, cite the leads issue** that decides against an
    alternative, in addition to summarizing the key points, tradeoffs, and
    rationale for the decision.

> [!IMPORTANT] Don't just list the alternatives, create a sub-section for each
> alternative and carefully describe the alternative, the advantages,
> disadvantages and what the core of the decision is to reject each alternative.

> [!IMPORTANT] Carefully research each alternative in the leads issue in order
> to provide this clear and comprehensive explanation.

## Building from a leads issue

Sometimes a proposal is specifically documenting and formalizing a decided leads
issue. When this is the case, carefully research that leads issue, reading the
original issue text and every comment on the issue. Also read any linked Google
documents, linked issues, examples, gists, or other supplemental information
cited.

-   **Summarize the leads issue**: Ensure you provide a high level summary of
    the _decided_ direction of the leads issue as the proposal.
-   **Capture and document** every key aspect of the decision made and factor
    that led to the decision. It is important that the proposal stands alone,
    and the leads issue is merely cited for context and history.
-   **Stay grounded**: Only include alternatives, rationale, and arguments based
    on what you find in the issue and related documents. No new information
    should be in the proposal.

Use the `gh` command line tool to query leads issues in order to carefully
examine all of the comments. Follow any mentioned links to gather more data.
Refer to the [GitHub CLI usage skill](/.agents/skills/github_cli/SKILL.md) for
detailed instructions on using the `gh` tool.

Ask the user to clarify any aspects of the leads issue that are unclear rather
than continuing to edit the proposal. If there are questions that you don't find
an answer to in the issue, ask this to the user and let them provide an answer
that you use as the basis of what to include.

## Examples are golden

Heavily leverage examples to illustrate both the specifics of the design being
proposed, and the nature of the change being proposed. More examples to
illustrate more aspects, corner cases, or provide a more complete understanding
are almost always good. Comments in example code should focus on what that part
of the example illustrates from the proposal.

## Keep the PR description in sync with the abstract

Whenever you edit the abstract or notice differences from the PR description,
you should update the PR description to match the abstract. The only exception
is to retain any "Assisted-by" or other tags at the end of the description that
are only needed there and not in the abstract.

If you are creating or updating a proposal, make sure the PR description in
question contains an `Assisted-by:` tag that is appropriate for describing which
AI tool is being used.
