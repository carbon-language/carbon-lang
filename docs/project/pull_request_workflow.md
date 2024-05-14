# Trunk-based pull-request GitHub workflow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Trunk based development](#trunk-based-development)
    -   [Green tests](#green-tests)
-   [Always use pull requests (with review) rather than pushing directly](#always-use-pull-requests-with-review-rather-than-pushing-directly)
-   [Small, incremental changes](#small-incremental-changes)
-   [Linear history](#linear-history)
-   [Merging a pull request](#merging-a-pull-request)

<!-- tocstop -->

## Overview

Carbon repositories follow a few basic principles:

-   Development directly on the `trunk` branch and
    [revert to green](#green-tests).
-   Always use pull requests, rather than pushing directly.
-   Changes should be small, incremental, and review-optimized.
-   Preserve linear history by
    [squashing](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits)
    pull requests rather than using unsquashed merge commits.

These principles try to optimize for several different uses or activities with
version control:

-   Continuous integration and bisection to identify failures and revert to
    green.
-   Code review both at the time of commit and follow-up review after commit.
-   Understanding how things evolve over time, which can manifest in different
    ways:
    -   When were things introduced?
    -   How does the main branch and project evolve over time?
    -   How was a bug or surprising thing introduced?

Note that this document focuses on the mechanical workflow and branch
management. Details of the code review process are in their own
[document](code_review.md).

## Trunk based development

We work in a simple
[trunk-based development](https://trunkbaseddevelopment.com/) model. This means
all development activity takes place on a single common `trunk` branch in the
repository (our default branch). We focus on
[small, incremental changes](#small-incremental-changes) rather than feature
branches or the "scaled" variations of this workflow.

### Green tests

The `trunk` branch should always stay "green". That means that if tests fail or
if we discover bugs or errors, we revert to a "green" state by default, where
the failure or bug is no longer present. Fixing forward is fine if that will be
comparably fast and efficient. The goal isn't to dogmatically avoid fixing
forward, but to prioritize getting back to green quickly. We hope to eventually
tool this through automatic continuous-integration powered submit queues, but
even those can fail and the principle remains.

## Always use pull requests (with review) rather than pushing directly

We want to ensure that changes to Carbon are always reviewed, and the simplest
way to do this is to consistently follow a pull request workflow. Even if the
change seems trivial, still go through a pull request -- it'll likely be trivial
to review. Always wait for someone else to review your pull request rather than
just merging it, even if you have permission to do so.

Our GitHub repositories are configured to require pull requests and review
before they are merged, so this rule is enforced automatically.

## Small, incremental changes

Developing in small, incremental changes improves code review time, continuous
integration, and bisection. This means we typically squash pull requests into a
single commit when landing. We use two fundamental guides for deciding how to
split up pull requests:

1. Ensure that each pull request builds and passes any tests cleanly when you
   request review and when it lands. This will ensure bisection and continuous
   integration can effectively process them.

2. Without violating the first point, try to get each pull request to be "just
   right": not too big, not too small. You don't want to separate a pattern of
   tightly related changes into separate requests when they're easier to review
   as a set or batch, and you don't want to bundle unrelated changes together.
   Typically you should try to keep the pull request as small as you can without
   breaking apart tightly coupled changes. However, listen to your code reviewer
   if they ask to split things up or combine them.

While the default is to squash pull requests into a single commit, _during_ the
review you typically want to leave the development history undisturbed until the
end so that comments on any particular increment aren't lost. We typically use
the GitHub squash-and-merge functionality to land things.

## Linear history

We want the history of the `trunk` branch of each repository to be as simple and
easy to understand as possible. While Git has strong support for managing
complex history and merge patterns, we find understanding and reasoning about
the history -- especially for humans -- to be at least somewhat simplified by
sticking to a linear progression. As a consequence, pull requests are squashed
when merging them.

## Merging a pull request

Once approved, and even if all checks have not finished running yet, a pull
request can be added to a
[merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue#about-merge-queues).
When queued, and after all pull request checks pass, this will automatically
create a squashed version of the commits in a temporary branch on top of `trunk`
and any previously queued merge. Checks run on this commit help to ensure that
the `trunk` stays "green" in presence of conflicting merges.

After the merge queue checks pass, the `trunk` branch pointer is updated to
include the now merged changset. The resulting commit title and description will
match exactly those of the pull request, so it is important to set those
appropriately before merging. Co-authors are preserved by this operation. If a
failure happens at any point, the merge fails, with both `trunk` and the pull
request branch kept in their original state.
