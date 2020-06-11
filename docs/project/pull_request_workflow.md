# Trunk-based pull-request GitHub workflow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Carbon repositories follow a few basic principles:

- Development directly on the `trunk` branch and revert to green.
- Always use pull requests, rather than pushing directly.
- Changes should be small, incremental, and review-optimized.
- Preserve linear history by
  [rebasing](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#rebase-and-merge-your-pull-request-commits)
  or
  [squashing](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits),
  pull requests rather than using unsquashed merge commits.

These principles try to optimize for several different uses or activities with
version control:

- Continuous integration and bisection to identify failures and revert to green
- Code review both at the time of commit and follow-up review after commit
- Understanding how things evolve over time, which can manifest in different
  ways
  - When were things introduced?
  - How does the main branch and project evolve over time?
  - How was a bug or surprising thing introduced?

Note that this isn't a complete guide to doing code reviews, and just focuses on
the mechanical workflow and branch management. TODO: Add an explicit link to
more detailed guidance on managing pull request based code reviews when it is
developed.

## Trunk based development

We work in a simple
[trunk-based development](https://trunkbaseddevelopment.com/) model. This means
all development activity takes place on a single common `trunk` branch in the
repository (our default branch). We focus on
[small, incremental changes](#small_incremental_changes) rather than feature
branches or the "scaled" variations of this workflow.

The `trunk` branch should always stay "green". That means that if tests fail or
if we discover bugs or errors, we revert to a "green" state (where the failure
or bug is no longer present) by default. Fixing forward is fine if that will be
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

Our GitHub repos are configured to require pull requests and review before they
are merged, so this rule is enforced automatically.

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

### Managing pull requests with multiple commits

Sometimes, it will make sense to _land_ a series of separate commits for a
single pull request through rebasing. This can happen when there is important
overarching context that should feed into the review, but the changes can be
usefully decomposed when landing them. When following this model, each commit
you intend to end up on the `trunk` branch needs to follow the same fundamental
rules as the pull request above: they should each build and pass tests when
landed in order, and they should have well written, cohesive commit messages.

Prior to landing the pull request, you are expected to rebase it (interactively
or non-interactively) to produce this final commit sequence. This kind of rebase
rewrites the history in Git, which can make it hard to track the resolution of
code review comments. Typically, only do this as a cleanup step when the review
has finished, or when it won't otherwise disrupt code review. Adding "addressing
review comments" commits during the review, and then squashing them away before
the pull request is merged is an expected and healthy pattern.

## Linear history

We want the history of the `trunk` branch of each repository to be as simple and
easy to understand as possible. While Git has strong support for managing
complex history and merge patterns, we find understanding and reasoning about
the history (especially for humans) to be at least somewhat simplified by
sticking to a linear progression. As a consequence we either squash pull
requests or rebase them when merging them.
