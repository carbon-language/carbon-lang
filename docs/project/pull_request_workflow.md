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

### Stacking dependent pull requests

Carbon uses pull requests in the common, distributed GitHub model where you
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
one pull request cannot serve as the _base_ for another pull request, so each
pull request will include all of the commits and diffs of the preceding pull
requests in the stack.

We suggest a specific workflow to address this:

1. Create your initial pull request from a branch of your fork, nothing special
   is needed at this step. Let's say you have a branch `feature-basic` in your
   clone of your fork, and that the `origin` remote is your fork.

   Push the branch to your fork:

   ```
   > git checkout feature-basic
   > git push origin
   ```

   And create a pull request for it using the
   [`gh`](/docs/project/contribution_tools.md#github_commandline_interface)
   tool:

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
   > git checkout feature-basic
   > git push upstream HEAD:pull-N-feature-basic
   ```

   Everyone marked as a contributor to Carbon is allowed to push branches if the
   name matches `pull-*` without pull request or other process. They can be
   force pushed as necessary and deleted. These branch names should only be used
   for this ephemeral purpose. All other branch names are protected.

   If you don't (yet) have this permission, feel free to ask anyone who does --
   anyone can checkout your pull request and push it to a branch with this name
   pattern for use in a stacked series of commits.

3. Create your stacked branch on your fork:

   ```
   > git checkout -b next-feature-extension
   > git commit -a -m 'Some initial work on the next feature.'
   > git push origin
   ```

4. Now when you create a pull request, use the upstream branch tracking your
   prior pull request as the base:

   ```
   > gh pr create --base pull-N-feature-basic
   ```

   This creates a baseline for the new, stacked pull request that you have
   manually synced to your prior pull request.

5. Each time you update the original pull request by pushing more commits to the
   `feature-basic` branch on your `origin`, you'll want to re-push to the
   upstream tracking branch as well:

   ```
   > git checkout feature-basic
   > git commit -a -m 'Address some code review feedback...'
   > git push
   > git push upstream HEAD:pull-N-feature-basic
   ```

   Then _merge_ those changes into your subsequent pull request:

   ```
   > git checkout next-feature-extension
   > git merge feature-basic
   > git push
   ```

   The merge will prevent disrupting the history of `next-feature-extension`
   where you may have code review comments on specific commits, while still
   allowing the pull request diff view to show the new delta after incorporating
   the new baseline.

6. Follow a similar process as in 5 above for merging updates from the main
   branch of `upstream`:

   ```
   > git checkout trunk
   > git pull --rebase upstream
   > git push                                    # update your fork (optional)
   > git checkout feature-basic
   > git merge trunk                             # merge w/o disrupting history
   > git push                                    # push to first PR on fork
   > git push upstream HEAD:pull-N-feature-basic # sync upstream tracking branch
   > git checkout next-feature-extension
   > git merge feature-basic                     # merge w/o disrupting history
   > git push                                    # push to second PR on fork
   ```

7. When the first pull requset lands in the main upstream branch, delete the
   upstream tracking branch for it:

   ```
   > git push upstream --delete pull-N-feature-basic
   ```

   The second pull request should automatically switch its baseline to the
   `trunk` branch of the upstream repository. Merge commits into your fork's
   branch for the second pull request can now be done directly from your `trunk`
   branch after pulling upstream.

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
