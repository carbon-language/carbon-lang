# Getting commit access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4246)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Commit access](#commit-access)
    -   [Related policies](#related-policies)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Longer idle times](#longer-idle-times)

<!-- tocstop -->

## Abstract

Establish a process for getting commit access. We will:

-   Grant access based on a developer's commit history.
    -   Someone with commit access should nominate, and a contributor may ask.
    -   A lead will approve nominations. Only one lead is needed.
-   Remove commit access once someone is idle for 6 months.
    -   "Idle" means no significant project activity on any of GitHub, Discord,
        or in meetings.
    -   Access removed due to being idle will be restored on request.

## Problem

Right now, we have an undocumented process for getting commit access, and no
real agreement for when to remove it. Commit access is important, so rather than
just making a documentation edit, I'm submitting this as a proposal.

## Background

### Commit access

When we say "commit access", what we mean is the ability to push commits to the
main `carbon-lang` repository, regardless of branch. Some details about the
implications:

-   Review and approvals would remain required for pushes to `trunk`.
    -   This does grant access to push to other branches, although we will
        continue to encourage fork-based workflows.
-   Pushing PRs becomes easier with commit access.
    -   Action workflows will automatically execute, instead of requiring a
        per-commit approval by someone with commit access.
    -   Modifying PRs after review approval is possible.
        -   This is both good (for small updates such as fixing typos and
            resolving conflicts) and bad (particularly for code security).
    -   Communication delays can make it hard for an author to resolve conflicts
        and reviewer to approve and merge before more conflicts are introduced.
        -   A possible solution to the conflict problem is to have the reviewer
            merge PRs more frequently, and regardless of any decision here, we
            may eventually need to adopt that approach.
-   Commit access functionally means the ability to approve and merge PRs from
    others.
    -   As alternatives, we could use either a separate GitHub team for
        approvals or
        [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).
        We have avoided these so far because we're small, having backup
        approvers can be helpful, and mistakes are easy to undo.
-   This does not include the ability to push to other repositories. While there
    are a few in the `carbon-language` organization, only the `carbon-lang`
    repository is actively used.
-   People with commit access in effect have access to secrets, can make
    releases, and so on.

### Related policies

This proposal does not supersede other project policies, in particular:

-   [CLA](/CONTRIBUTING.md#contributor-license-agreements-clas)
-   [Code of Conduct](/CODE_OF_CONDUCT.md)
-   [Review process](/docs/project/code_review.md)

## Proposal

The key things I think should be covered in this proposal are:

-   Whether we want to require a lead to approve commit access additions.
-   We're choosing to do a 6 month inactive period for removal.

Things I think we should be okay iterating on without going through evolution
include:

-   Exactly how we define non-idle activity beyond merging and approving PRs,
    both of which mechanically require this access.
-   The detailed process for additions, including nominating.
    -   We want a good starting point, but it may not be worth sending all
        changes through the proposal process.
-   The detailed process for removals, such as whether leads are approving
    removals.
    -   This is something it's been suggested to fully automate, preventing
        review of removals.

See the new [Commit access](/docs/project/commit_access.md) document for
details.

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Establishing the processes around commit access, and making sure they're
        reasonable, is important to maintaining the community.

## Alternatives considered

### Longer idle times

We discussed using a longer idle time, like 1, 2, or 3 years. We're leaning
towards the shorter 6 month period because of concerns about forgotten access
causing issues. We're hoping 6 months is a minimum of inconvenience, and want it
to be easy to get access back on request.
