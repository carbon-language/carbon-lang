"""Updates the contributors-with-label-access team.

This team exists because we need a team to manage triage access to repos;
GitHub doesn't allow the org to be set to triage access, only read/write. It
will be updated to include all members of the carbon-language organization.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
from typing import List, Optional, Set

# https://github.com/PyGithub/PyGithub
# GraphQL is preferred, but falling back to pygithub for unsupported mutations.
import github

from github_tools import github_helpers

# The organization to mirror members from.
_ORG = "carbon-language"

# The team to mirror to.
_TEAM = "contributors"

# Accounts in the org to skip mirroring.
_IGNORE_ACCOUNTS = ("CarbonLangInfra", "google-admin", "googlebot")

# Queries organization members.
_ORG_MEMBER_QUERY = """
query {
  organization(login: "%s") {
    membersWithRole(first: 100%%(cursor)s) {
      nodes {
        login
      }
      %%(pagination)s
    }
  }
}
"""

# The path for nodes in _ORG_MEMBER_QUERY.
_ORG_MEMBER_PATH = ("organization", "membersWithRole")

# Queries team members.
_TEAM_MEMBER_QUERY = """
query {
  organization(login: "%s") {
    team(slug: "%s") {
      members(first: 100%%(cursor)s) {
        nodes {
          login
        }
        %%(pagination)s
      }
    }
  }
}
"""

# The path for nodes in _TEAM_MEMBER_QUERY.
_TEAM_MEMBER_PATH = ("organization", "team", "members")


def _parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    github_helpers.add_access_token_arg(parser, "admin:org, repo")
    return parser.parse_args(args=args)


def _load_org_members(client: github_helpers.Client) -> Set[str]:
    """Loads org members."""
    print("Loading %s..." % _ORG)
    org_members = set()
    ignored = set()
    for node in client.execute_and_paginate(
        _ORG_MEMBER_QUERY % _ORG, _ORG_MEMBER_PATH
    ):
        login = node["login"]
        if login not in _IGNORE_ACCOUNTS:
            org_members.add(login)
        else:
            ignored.add(login)
    print(
        "%s has %d non-ignored members, and %d ignored."
        % (_ORG, len(org_members), len(ignored))
    )
    unignored = set(_IGNORE_ACCOUNTS) - ignored
    assert not unignored, "Missing ignored accounts: %s" % unignored
    return org_members


def _load_team_members(client: github_helpers.Client) -> Set[str]:
    """Load team members."""
    print("Loading %s..." % _TEAM)
    team_members = set()
    for node in client.execute_and_paginate(
        _TEAM_MEMBER_QUERY % (_ORG, _TEAM), _TEAM_MEMBER_PATH
    ):
        team_members.add(node["login"])
    print("%s has %d members." % (_ORG, len(team_members)))
    return team_members


def _update_team(
    gh: github.Github, org_members: Set[str], team_members: Set[str]
) -> None:
    """Updates the team if needed.

    This switches to pygithub because GraphQL lacks equivalent mutation support.
    """
    gh_team = gh.get_organization(_ORG).get_team_by_slug(_TEAM)  # type: ignore
    add_members = org_members - team_members
    if add_members:
        print("Adding members: %s" % ", ".join(add_members))
        for member in add_members:
            gh_team.add_membership(gh.get_user(member))  # type: ignore

    remove_members = team_members - org_members
    if remove_members:
        print("Removing members: %s" % ", ".join(remove_members))
        for member in remove_members:
            gh_team.remove_membership(gh.get_user(member))  # type: ignore


def main() -> None:
    parsed_args = _parse_args()
    print("Connecting...")
    client = github_helpers.Client(parsed_args)

    org_members = _load_org_members(client)
    team_members = _load_team_members(client)
    if org_members != team_members:
        gh = github.Github(parsed_args.access_token)  # type: ignore
        _update_team(gh, org_members, team_members)
    print("Done!")


if __name__ == "__main__":
    main()
