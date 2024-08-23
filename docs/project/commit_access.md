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

Commit access is what allows developers to merge PRs into the main
[carbon-lang repository](https://github.com/carbon-language/carbon-lang/). It
also gives the ability to [review, approve, and merge changes](code_review.md).

Developers with commit access are not required to do reviews. It is granted as a
convenience in order to allow fixing PRs after approval, for example because the
review noted a minor typo, or because there is a conflict that the PR author
needs to resolve before merging.

Developers with commit access are expected to make reasonable choices about when
to review and approve PRs for others, and what changes to make to their own PRs
without asking for another round of review.

## Getting access

Contributors can request commit access based on their commit history. We want to
make sure that developers with commit access are reasonably familiar with the
style and structure of the codebase. Access will typically be granted when
developers have contributed several PRs and are expecting to continue making
more.

Typically someone who has been doing reviews for a contributor will nominate,
and contributors are encouraged to ask a reviewer to nominate if needed.
Nominations will be reviewed by at least one lead, and a lead's approval will
lead to access.

When approved, an invitation will be sent to join the
[Commit access team](https://github.com/orgs/carbon-language/teams/commit-access)
on GitHub. The invitation should cause a notification from GitHub, but it's also
possible to go to the
[invitation link](https://github.com/orgs/carbon-language/invitation) directly.
Access is granted when the invitation is accepted.

## Removing access

We'll periodically remove commit access from contributors who have been idle for
over 6 months. We'll use a combination of sources to determine what "idle"
means, including whether a developer has been either sending or reviewing PRs.

For example, for `jonmeow`, GitHub searches (defaulting to PRs for convenience,
but other types are included):

-   [repository:carbon-language/carbon-lang author:jonmeow](https://github.com/search?q=repository%3Acarbon-language%2Fcarbon-lang+author%3Ajonmeow&type=pullrequests)
-   [repository:carbon-language/carbon-lang commenter:jonmeow](https://github.com/search?q=repository%3Acarbon-language%2Fcarbon-lang+commenter%3Ajonmeow&type=pullrequests)
