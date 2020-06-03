# Linear pull-request GitHub workflow

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Carbon repositories follow three basic principles:

- Always use pull requests (rather than directly pushing to the main branch)
- Commit small, incremental changes to optimize for review, continuous
  integration, and bisection
- Linear history through
  [rebasing](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#rebase-and-merge-your-pull-request-commits)
  or
  [squashing](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits),
  rather than merge commits from branches or forks

These principles try to optimize for several different uses or activities with
version control:

- Continuous integration and bisection to identify failures and revert to green
- Code review both at the time of commit and follow-up review after commit
- Understanding how things evolve over time, which can manifest in different
  ways
  - When were things introduced?
  - How does the main branch and project evolve over time?
  - How was a bug or surprising thing introduced?

## Always use pull requests (with review) rather than pushing directly

We want to ensure that changes to Carbon are always reviewed, and the simplest
way to do this is to consistently follow a pull request workflow. Even if the
change seems trivial, still go through a pull request -- it'll likely be trivial
to review. Always wait for someone else to review your pull request rather than
just merging it, even if you have permission to do so.

We have set up automation on GitHub both to require pull requests and review
before they are merged so that this doesn't require any effort from
contributors.

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
   as a set or batch. And you don't want to bundle unrelated changes together.
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
you intend to _land_ needs to follow the same fundamental rules as the pull
request above: they should each build and pass tests when landed in order, and
they should have well written, cohesive commit messages.

It may also make sense to rewrite history by interactive or non-interactive
rebasing to arrive at this final commit sequence. Be mindful of on-going code
review in choosing when to do this. Rewriting history in this way can make it
hard to track the resolution of comments. Typically, only do this as a cleanup
step when the review has finished, or when it won't otherwise disrupt code
review. Adding "addressing review comments" commits during the review, and then
rebasing them away before the pull request is merged is an expected and healthy
pattern.

This isn't intended to be full or complete guidance on how to manage code
reviews, just a basic indication of how to end up with a clean linear history on
the main branch. TODO: Add an explicit link to more detailed guidance on
managing pull request based code reviews when it is developed.

## Linear history

We want the history of the main branch of each repository to be as simple and
easy to understand as possible. While Git has strong support for managing
complex history and merge patterns, we find understanding and reasoning about
the history (especially for humans) to be at least somewhat simplified by
sticking to a linear progression. As a consequence we either squash pull
requests or rebase them when merging them.
