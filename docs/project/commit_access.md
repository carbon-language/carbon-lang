# Commit access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Getting access](#getting-access)
-   [Removing access](#removing-access)

<!-- tocstop -->

## Overview

First and foremost, commit access is **not required** for contributing to Carbon! Anyone can send a pull request with a change to Carbon. Very few parts of our process require commit access, and most development activity is exactly the same for folks with or without. This is an important feature of our GitHub workflow for us, and one we plan to keep.

The main thing commit access allows is for developers to merge PRs into the main
[carbon-lang repository](https://github.com/carbon-language/carbon-lang/). They still need review, and often a reviewer with commit access handles the merge. It
also gives the ability to [approve and merge changes](code_review.md).

Developers with commit access are not required to do reviews. It is granted as a
convenience in order to allow fixing PRs after approval, for example because the
review noted a minor typo, or because there is a conflict that the PR author
needs to resolve before merging.

Developers with commit access are expected to make reasonable choices about when
to approve or merge PRs for others -- generally, the change either needs to be trivially understood without context on the code in question (like a cleanup, typo fix, etc), or the developer should have reasonable context on the code in question. If in doubt, feel free to review but leave approving or merging for someone else.

Similarly, developers with commit access are expected to make reasonable choices about what changes to make to their own PRs
without asking for another round of review prior to merge, even after approval. And again, if in doubt, ask for review.

## Getting access

Contributors can request commit access based on their commit history. We want to
make sure that developers with commit access are reasonably familiar with the
style and structure of the codebase. Access will typically be granted when
developers have contributed several PRs and are expecting to continue making
more.

After a few non-trivial PRs are merged, contributors should ask a reviewer to nominate for commit access if they plan to keep contributing. Reviewers can also directly suggest and nominate someone.
Nominations need to be approved by at least one lead.

When approved, an invitation will be sent to join the
[Commit access team](https://github.com/orgs/carbon-language/teams/commit-access)
on GitHub. The invitation should cause a notification from GitHub, but it's also
possible to go to the
[invitation link](https://github.com/orgs/carbon-language/invitation) directly.
Access is granted when the invitation is accepted.

## Removing access

We'll periodically remove commit access from contributors who have been idle for
over 6 months. We'll use a combination of sources to determine what "idle"
means, including whether a developer has been either sending or reviewing PRs. Developers should always ask for it to be restored if they plan to start contributing again or come back from a break. Any relevant past contributions will still apply and allow it to be restored trivially.

For example, for `jonmeow`, GitHub searches (defaulting to PRs for convenience,
but other types are included):

-   [repository:carbon-language/carbon-lang author:jonmeow](https://github.com/search?q=repository%3Acarbon-language%2Fcarbon-lang+author%3Ajonmeow&type=pullrequests)
-   [repository:carbon-language/carbon-lang commenter:jonmeow](https://github.com/search?q=repository%3Acarbon-language%2Fcarbon-lang+commenter%3Ajonmeow&type=pullrequests)
