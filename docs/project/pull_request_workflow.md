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

### Stacking dependent pull requests

Carbon uses pull requests in the normal, distributed GitHub model where you
first fork the repository (typically into your own private GitHub fork), and
then develop on feature branches in that fork. When a branch is ready for
review, it is turned into a pull request against the main fork. This flow should
always be where you start when contributing to Carbon and scales well even with
many independent changes in flight.

However, a common limitation to hit is when you want to create a _stack_ of
_dependent_ but small and incremental changes and allow them to be reviewed in
parallel. Each of these should be its own pull request to facilitate our desire
for small and incremental changes and review. Unfortunately GitHub has very poor
support for managing the _review_ of these stacked pull requests. Specifically,
one pull request cannot normally serve as the _base_ for another pull request,
so each pull request will include all of the commits and diffs of the preceding
pull requests in the stack.

We suggest a specific workflow to address this:

1. Create your initial pull request exactly like normal from a branch of your
   fork. Let's say you have a branch `feature-123` in your clone of your fork.
   Push it to your fork as normal:

   ```
   > git checkout feature-123
   > git push origin
   ```

   And create a pull request for it like normal:

   ```
   > gh pr create
   ```

   And let's imagine this creates a pull request `N` in the upstream repository.

2. _If_ you end up needing to create a subsequent pull request based on the
   first one, we need to create a _branch_ in the upstream repository that
   tracks the first pull request and serves as the base for the subsequent pull
   request. Assuming your fork `$USER/carbon-lang` is remote `origin` and
   `carbon-language/carbon-lang` is remote `upstream` in your repository:

   ```
   > git checkout feature-123
   > git push upstream HEAD:pull-N-feature-123
   ```

   This will require you to have access to create this branch in the upstream
   repository. If you don't already have this access, feel free to ask someone
   with access to do this step for you (they can). We'll try to make sure anyone
   routinely developing a stack of pull requests has the access they need for
   this (but don't worry, it doesn't impact the access to the main branch).

3. Create your stacked branch on your fork like normal:

   ```
   > git checkout -b next-feature-456
   > git commit -a -m 'Some initial work on the next feature.'
   > git push origin
   ```

4. Now when you create a pull request, use the upstream branch tracking your
   prior pull request as the base:

   ```
   > gh pr create --base pull-N-feature-123
   ```

   This creates a baseline for the new, stacked pull request that you have
   manually synced to your prior pull request.

5. Each time you update the original pull request by pushing more commits to the
   `feature-123` branch on your `origin`, you'll want to re-push to the upstream
   tracking branch as well:

   ```
   > git checkout feature-123
   > git commit -a -m 'Address some code review feedback...'
   > git push
   > git push upstream HEAD:pull-N-feature-123
   ```

   Then _merge_ those changes into your subsequent pull request:

   ```
   > git checkout next-feature-456
   > git merge feature-123
   > git push
   ```

   The merge will prevent disrupting the history of `next-feature-456` where you
   may have code review comments on specific commits, while still allowing the
   pull request diff view to show the new delta after incorporating the new
   baseline.

6. Follow a similar process as in 5 above for merging updates from the main
   branch of `upstream`:

   ```
   > git checkout master
   > git pull --rebase upstream
   > git push                                   # update your fork (optional)
   > git checkout feature-123
   > git merge master                           # merge w/o disrupting history
   > git push                                   # push to first PR on fork
   > git push upstream HEAD:pull-N-feature-123  # sync upstream tracking branch
   > git checkout next-feature-456
   > git merge feature-123                      # merge w/o disrupting history
   > git push                                   # push to second PR on fork
   ```

7. When the first pull requset lands in the main upstream branch, delete the
   upstream tracking branch for it:

   ```
   > git push upstream --delete pull-N-feature-123
   ```

   The second pull request should automatically switch its baseline to the
   `master` branch of the upstream repository. Merge commits into your fork's
   branch for the second pull request can now be done directly from your
   `master` branch after pulling upstream.

8. When landing the a stacked pull request, it will require actively rebasing or
   squashing due to the complex merge history used while updating.

9. If you want to split the two pull requests so they become independent, you
   can explicitly edit the base branch of a pull request in the GitHub UI.
   Alternatively you can simply delete the tracking branch as above. The result
   will be two pull requests with an overlapping initial sequence of commits.
   You can then restructure each one to make sense independently.

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
