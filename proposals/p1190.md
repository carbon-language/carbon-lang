# Reviewer-merged PRs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1190)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Never merge PRs from developers with merge access](#never-merge-prs-from-developers-with-merge-access)
    -   [Grant all potential contributors merge access](#grant-all-potential-contributors-merge-access)
    -   [Allow reviewers to clean up pull request descriptions](#allow-reviewers-to-clean-up-pull-request-descriptions)
    -   [Only have authors resolve merge conflicts](#only-have-authors-resolve-merge-conflicts)
    -   [Implement a two-person rule for source code changes](#implement-a-two-person-rule-for-source-code-changes)

<!-- tocstop -->

## Problem

We've been having authors merged PRs, but that's not going to work when we get
contributors who don't have merge access. We need a solution.

## Background

It's been mentioned that LLVM favors having authors merge due to the risks of
breaking build bots. The LLVM community's leaning is to favor author merges so
that the author can decide whether to try rolling back or fixing forward.

At present Carbon has essentially been using a
[shared repository model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#shared-repository-model)
where authors merge their own PRs, but that's difficult to extend to a fully
public setup. The
[fork and pull model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model)
is the other main git collaboration model, and is popular with open source
projects.

## Proposal

Encourage reviewers to merge when they feel okay doing so. Let reviewers make
that choice. Let authors say they'll merge themselves.

This is a
[fork and pull model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model)
which encourages reviewer merges.

## Details

See changes to [code review](/docs/project/code_review.md).

## Rationale

-   [Community and culture](/docs/project/goals.md#community-and-culture)
    -   Defines a process for accepting contributions from developers who don't
        have merge access.

## Alternatives considered

### Never merge PRs from developers with merge access

We could tell reviewers to never merge PRs from developers with merge access.
This is also a
[fork and pull model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model),
but minimizing reviewer merges.

Advantages:

-   While this proposal suggest authors could opt out from having the reviewer
    merge, minimizing reviewer merges avoids a gray area when an author forgets
    or the reviewer misses it (even if enforced, the author might do it wrong).

Disadvantages:

-   Relies on the reviewer figuring out whether the author has merge access,
    which they may forget to do. The most likely consequence is that outside
    contributors will need to ping to get PRs merged.
-   Makes reviewer merges less common. This in turn can make them less
    consistently done well due to unfamiliarity.
    -   Contributors without merge access would be following a different
        process, and as a consequence it increases the chance that a new
        contributor would be first to discover a problem, which in turn can
        discourage contributions.

In the future, build bots may cause more issues, and we may lean more in this
direction. It could also be that we end up here through standard practice of
coordinating with the reviewer is concerned the author may need to fix a break
post-commit. However, it doesn't seem necessary to make it a hard rule at
present.

### Grant all potential contributors merge access

We could grant the public merge access; in other words, continue with a
[shared repository model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#shared-repository-model),
but fully public instead of private. This way, reviewers would not need to
consider access.

Advantages:

-   Authors can always merge, removing the burden on reviewers.

Disadvantages:

-   Relies on approvals heavily for code security, which constrains decisions
    [regarding whether to keep CODEOWNERS](https://github.com/carbon-language/carbon-lang/issues/413).
-   This kind of setup is likely atypical for GitHub projects, and so may be
    more surprising than alternatives.
-   Harder for reviewers to discern between frequent contributors and new
    contributors.
    -   GitHub has an option,
        ["Dismiss stale pull request approvals when new commits"](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/managing-a-branch-protection-rule):
        we would likely want to enable it to reduce the weight on reviewers,
        although it means reviewers would need to approve again for any change
        (likely including merge commits).
-   May create issues when a novice contributor breaks something.
    -   There must not be an expectation that novice contributors should
        understand processes, unless demonstrating an understanding becomes a
        prerequisite for review.

### Allow reviewers to clean up pull request descriptions

We could allow reviewers to clean up pull request descriptions, rather than
asking the author to.

Advantages:

-   Decreases the number of review round-trips.

Disadvantages:

-   May lead to PR descriptions not being in the authors voice, frustrating
    authors.

The preference is to let authors control pull request descriptions.

### Only have authors resolve merge conflicts

We could only have authors resolve merge conflicts, instead of having reviewers
do it.

Advantages:

-   Lowers the chance of an incorrect merge, because authors are likely to
    better understand the conflict.
-   Lets ambiguous resolves be handled with the author's voice.
    -   If a bad resolve introduces a bug, it's the author's fault, rather than
        being the reviewer's fault but _attributed_ to the author.

Disadvantages:

-   Increases the number of review round-trips.
    -   Pull requests by authors lacking merge access would have an extra
        round-trip, because the author would resolve conflicts then the reviewer
        would merge the pull request. In a worst case this may bounce back and
        forth due to new conflicts being added before the reviewer merged.

The preference is to minimize review round-trips. However, this could still be a
real-world outcome if reviewers generally only merge when there are no
outstanding merge conflicts.

### Implement a two-person rule for source code changes

We could implement a
[two-person rule](https://en.wikipedia.org/wiki/Two-man_rule) for source code
changes, where both an author _and_ reviewer must see the merged code. GitHub
supports this with
["Dismiss stale pull request approvals when new commits are pushed"](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/managing-a-branch-protection-rule),
although we may also need to require 2 approvers per pull request.

Advantages:

-   Gives stronger code security, eliminating situations where either the author
    or reviewer could merge unreviewed changes.

Disadvantages:

-   Increases the number of review round-trips.
    -   Right now, reviewers can approve with minor comments ("fix typo").
        Fixing those would require a new commit, which in turn would require
        fresh approval.
    -   Truly preventing this issue may require setting 2 approvers per pull
        request, so that a reviewer couldn't push a commit to the pull request
        then approve and merge. Requiring 2 approvers also increases review
        overhead.

The preference is to minimize review round-trips.
