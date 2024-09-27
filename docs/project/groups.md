# Groups

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Linked entities](#linked-entities)
-   [Carbon leads](#carbon-leads)
-   [Conduct team](#conduct-team)
    -   [Moderators](#moderators)
-   [Admins](#admins)
-   [Implementation team](#implementation-team)
-   [Security](#security)
-   [Access groups](#access-groups)
    -   [GitHub commit access](#github-commit-access)
    -   [GitHub label and review access](#github-label-and-review-access)
    -   [Google Drive access](#google-drive-access)

<!-- tocstop -->

## Overview

These are groups used by the Carbon Language project, listed here for central
tracking. Membership will typically be managed by a team owner or admin; see
each group's summary for information on how to join.

Note that some links are admin or member restricted. We're providing public
links where possible.

### Linked entities

Groups are defined by their linked entities in Carbon's various forums:

-   [GitHub teams](https://github.com/orgs/carbon-language/teams): Used
    primarily for GitHub ACLs.
    -   [GitHub organization members](https://github.com/orgs/carbon-language/people):
        This should only contain people in teams.
-   Discord roles: Used to identify team members on Discord, and for ACLs.
-   [Google groups](https://admin.google.com/ac/groups): Used for a mix of
    contact lists and ACLs.

## Carbon leads

See [governance structure](evolution.md#governance-structure) for team
information.

-   GitHub teams: [Leads](https://github.com/orgs/carbon-language/teams/leads)
-   Discord roles: lead
-   Google groups: None

## Conduct team

See [Conduct team](teams/conduct_team.md) for information.

-   GitHub teams: None
-   Discord roles: None
-   Google groups:
    [conduct@carbon-lang.dev](https://groups.google.com/a/carbon-lang.dev/g/conduct/about)
    Used as a contact list.

### Moderators

See [moderators](moderators.md) for information.

-   GitHub teams:
    [Moderators](https://github.com/orgs/carbon-language/teams/moderators)
-   Discord roles: moderator, senior-moderator
-   Google groups:
    [moderators](https://groups.google.com/a/carbon-lang.dev/g/moderators/about):
    Used for Google Drive ACLs.

## Admins

Maintains infrastructure. Membership changes are handled on an as-needed basis
by leads.

Note that while various groups exist, the way admins are actually configured
goes a little beyond this.

-   Github teams: [Admins](https://github.com/orgs/carbon-language/teams/admins)
    -   Canonically, the
        [role:owner](https://github.com/orgs/carbon-language/people?query=role%3Aowner)
        search.
-   Discord roles: admin
-   Google groups:
    [admins](https://groups.google.com/a/carbon-lang.dev/g/conduct/about): Used
    for `carbon-lang.dev` security settings.

## Implementation team

This team is responsible for development of Carbon's primary, reference
implementation and toolchain. It also oversees other related implementation work
within the Carbon project, from tooling of the language spec to test suites.
There may be some overlap with [admins](#admins) -- any issue can be resolved by
escalating to the [Carbon leads](#carbon-leads). Notably, this team is _not_
responsible for the _design_ of the language itself, only for its
implementation.

-   GitHub teams: None
-   Discord role: implementation-team
-   Google groups: None

## Security

Receives GitHub security reports. Membership changes are handled on an as-needed
basis by leads.

-   GitHub teams:
    [Security](https://github.com/orgs/carbon-language/teams/security)
-   Discord roles: None
-   Google groups: None

## Access groups

These groups are defined by the access they grant. They are not directly tied to
any of the above community groups.

### GitHub commit access

Developers who can merge and approve changes on GitHub. See
[commit access](commit_access.md) for information.

-   GitHub teams:
    [Commit access](https://github.com/orgs/carbon-language/teams/commit-access)
-   Discord roles: None
-   Google groups: None

### GitHub label and review access

Developers who can label and assign PRs and issues on GitHub, particularly
proposals. See [CONTRIBUTING.md](/CONTRIBUTING.md#getting-access) for more
information.

-   GitHub teams:
    [Label and review access](https://github.com/orgs/carbon-language/teams/label-and-review-access)
-   Discord roles: None
-   Google groups: None

### Google Drive access

For Google Drive and the contained documents, we have separate groups for
commenting and contributing (modify and create). See
[CONTRIBUTING.md](/CONTRIBUTING.md#getting-access) for more information.

-   GitHub teams: None
-   Discord roles: None
-   Google groups:
    [commenters](https://groups.google.com/a/carbon-lang.dev/g/commenters/about)
    and
    [contributors](https://groups.google.com/a/carbon-lang.dev/g/contributors/about):
    Used for Google Drive ACLs.
