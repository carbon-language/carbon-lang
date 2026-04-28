#!/usr/bin/env python3
"""Check if a PR depends on other open PRs based on shared commits.

Usage examples:
  # Check a specific PR in dry-run mode:
  GITHUB_ACCESS_TOKEN=$(gh auth token) \
    python3 github_tools/check_dependent_pr.py --pr-number <PR_NUMBER> --dry-run

  # Scan all dependent PRs in dry-run mode:
  GITHUB_ACCESS_TOKEN=$(gh auth token) \
    python3 github_tools/check_dependent_pr.py --scan --dry-run
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import datetime
import importlib.util
import json
import re
import os
import sys
import requests
from typing import Any, Optional

# Do some extra work to support direct runs.
try:
    from github_tools import github_helpers
except ImportError:
    github_helpers_spec = importlib.util.spec_from_file_location(
        "github_helpers",
        os.path.join(os.path.dirname(__file__), "github_helpers.py"),
    )
    assert github_helpers_spec is not None
    github_helpers = importlib.util.module_from_spec(github_helpers_spec)
    github_helpers_spec.loader.exec_module(github_helpers)  # type: ignore


# Queries
_QUERY_OPEN_PRS = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequests(states: OPEN, first: 100%(cursor)s) {
      nodes {
        number
        commits(first: 100) {
          nodes {
            commit {
              oid
            }
          }
        }
      }
      %(pagination)s
    }
  }
}
"""

_QUERY_DEPENDENT_PRS = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequests(states: OPEN, labels: ["dependent"], first: 100%(cursor)s) {
      nodes {
        number
      }
      %(pagination)s
    }
  }
}
"""

_QUERY_PR_DETAILS = """
query GetPrDetails($prNumber: Int!) {
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequest(number: $prNumber) {
      id
      headRefOid
      labels(first: 100) {
        nodes {
          name
          id
        }
      }
      commits(first: 100) {
        nodes {
          commit {
            oid
          }
        }
      }
      comments(first: 100) {
        nodes {
          id
          body
          isMinimized
        }
      }
    }
  }
}
"""

_QUERY_LABEL = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    label(name: "dependent") {
      id
    }
  }
}
"""

_QUERY_MAX_MERGED_PR = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequests(
      states: MERGED
      orderBy: {field: CREATED_AT, direction: DESC}
      first: 1
    ) {
      nodes {
        number
      }
    }
  }
}
"""

_MUTATION_ADD_LABEL = """
mutation AddLabel($labelableId: ID!, $labelIds: [ID!]!) {
  addLabelsToLabelable(
    input: {labelableId: $labelableId, labelIds: $labelIds}
  ) {
    clientMutationId
  }
}
"""

_MUTATION_REMOVE_LABEL = """
mutation RemoveLabel($labelableId: ID!, $labelIds: [ID!]!) {
  removeLabelsFromLabelable(
    input: {labelableId: $labelableId, labelIds: $labelIds}
  ) {
    clientMutationId
  }
}
"""

_MUTATION_UPDATE_COMMENT = """
mutation UpdateComment($id: ID!, $body: String!) {
  updateIssueComment(input: {id: $id, body: $body}) {
    clientMutationId
  }
}
"""

_MUTATION_ADD_COMMENT = """
mutation AddComment($subjectId: ID!, $body: String!) {
  addComment(input: {subjectId: $subjectId, body: $body}) {
    clientMutationId
  }
}
"""


def _print_err(*args: Any, **kwargs: Any) -> None:
    """Prints to stderr."""
    kwargs["file"] = sys.stderr
    print(*args, **kwargs)


def _parse_pr_number(x: Any) -> Optional[int]:
    """Parses x into a positive integer if possible."""
    if isinstance(x, int):
        return x if x > 0 else None
    if isinstance(x, str) and x.isdigit():
        val = int(x)
        return val if val > 0 else None
    return None


def _parse_and_validate_state(
    json_str: str,
    open_pr_numbers: set[int],
    max_merged_pr: int = 10000,
    pr_number: int = 0,
) -> tuple[list[int], list[int], Optional[str]]:
    """Parses and validates the state from a JSON string."""
    parsed_open: list[int] = []
    parsed_merged: list[int] = []
    first_commit: Optional[str] = None

    raw_state = json.loads(json_str)
    if not isinstance(raw_state, dict):
        raise ValueError(f"PR #{pr_number}: Parsed JSON is not a dictionary.")

    for x in raw_state.get("open", []):
        val = _parse_pr_number(x)
        if val is None:
            raise ValueError(
                f"PR #{pr_number}: Invalid PR number format in 'open': {x}"
            )
        elif val not in open_pr_numbers and val > max_merged_pr:
            raise ValueError(
                f"PR #{pr_number}: Rejecting PR #{val} from 'open' because "
                "it is not an open PR and exceeds maximum merged PR "
                f"#{max_merged_pr}."
            )
        else:
            parsed_open.append(val)
    for x in raw_state.get("merged", []):
        val = _parse_pr_number(x)
        if val is None:
            raise ValueError(
                f"PR #{pr_number}: Invalid PR number format in 'merged': {x}"
            )
        elif val in open_pr_numbers:
            raise ValueError(
                f"PR #{pr_number}: Rejecting PR #{val} from 'merged' "
                "because it is actually open."
            )
        elif val > max_merged_pr:
            raise ValueError(
                f"PR #{pr_number}: Rejecting PR #{val} from 'merged' "
                f"because it exceeds maximum merged PR #{max_merged_pr}."
            )
        else:
            parsed_merged.append(val)
    if "first_commit" in raw_state:
        fc = raw_state["first_commit"]
        if isinstance(fc, str) and re.fullmatch(r"[0-9a-fA-F]{40}", fc):
            first_commit = fc
        else:
            raise ValueError(
                f"PR #{pr_number}: Invalid commit OID format in "
                f"'first_commit': {fc}"
            )
    return parsed_open, parsed_merged, first_commit


def _set_commit_status(
    sha: str,
    state: str,
    description: str,
    token: str,
    dry_run: bool,
) -> None:
    """Sets the commit status via the GitHub REST API."""
    url = (
        "https://api.github.com/repos/carbon-language/carbon-lang/"
        f"statuses/{sha}"
    )
    headers = {
        "Authorization": f"bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "state": state,
        "description": description,
        "context": "PR dependencies check",
    }
    if dry_run:
        _print_err(
            f"[Dry-run] Would set commit status on {sha[:8]} to {state} "
            f"({description})"
        )
        return

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        _print_err(f"Set commit status on {sha[:8]} to {state}")
    except Exception as e:
        _print_err(f"Error setting commit status on {sha[:8]}: {e}")


def _process_pr(
    client: github_helpers.Client,
    pr_number: int,
    pr_to_commits: dict[int, list[str]],
    open_pr_numbers: set[int],
    label_id: str,
    token: str,
    dry_run: bool = False,
    scanning: bool = False,
    max_merged_pr: int = 10000,
) -> None:
    """Processes a single PR to check for dependencies and update comments."""
    current_res = client.execute(
        _QUERY_PR_DETAILS, variable_values={"prNumber": pr_number}
    )
    pr_node = current_res["repository"]["pullRequest"]
    if not pr_node:
        _print_err(f"PR #{pr_number} not found.")
        return

    pr_id = pr_node["id"]
    commits = pr_node["commits"]["nodes"]
    comments = pr_node["comments"]["nodes"]
    labels = pr_node["labels"]["nodes"]

    open_deps: list[int] = []

    if len(commits) <= 1:
        _print_err(
            f"PR #{pr_number} has 1 or fewer commits, skipping overlap check."
        )
        current_oids = [c["commit"]["oid"] for c in commits]
    else:
        current_oids = [c["commit"]["oid"] for c in commits]

        # Dependency Logic: Overlap and Sequence
        #
        # We consider PR B dependent on PR A if:
        # 1. The dependency PR A was created before PR B (A.number < B.number).
        # 2. There is a non-empty overlap of commits between PR A and PR B.
        # 3. PR B has at least one commit not present in PR A.
        #
        # Why this works:
        # - Ensures the dependency direction reflects the creation sequence.
        # - Handles minor fixes or differences by only requiring overlap, not
        #   strict subset inclusion.
        # - Avoids circular dependencies via the sequence check.
        current_oids_set = set(current_oids)
        for other_pr_num, other_oids in pr_to_commits.items():
            if other_pr_num >= pr_number:
                continue
            other_oids_set = set(other_oids)
            if not (other_oids_set & current_oids_set):
                continue
            if not (current_oids_set - other_oids_set):
                continue
            open_deps.append(other_pr_num)

    # Parse existing comment
    marker_prefix = "<!-- check_dependent_pr "
    existing_comment_id = None
    parsed_open_deps: list[int] = []
    parsed_merged_deps: list[int] = []
    previous_first_commit: Optional[str] = None

    matching_comment = None
    for comment in comments:
        # If a marker comment is hidden (minimized), we ignore it and treat
        # the PR as if it never had that comment.
        if marker_prefix in comment["body"] and not comment.get("isMinimized"):
            matching_comment = comment
            break

    if matching_comment:
        existing_comment_id = matching_comment["id"]
        body = matching_comment["body"]
        start = body.find(marker_prefix) + len(marker_prefix)
        end = body.find(" -->", start)
        if end != -1:
            parsed_open_deps, parsed_merged_deps, previous_first_commit = (
                _parse_and_validate_state(
                    body[start:end], open_pr_numbers, max_merged_pr, pr_number
                )
            )

    # Keep tracking previously identified dependencies if they are still open,
    # even if they no longer pass the subset check (e.g. they got new commits).
    for pr in parsed_open_deps:
        if pr in open_pr_numbers and pr not in open_deps:
            open_deps.append(pr)

    # Identify newly merged PRs
    newly_merged_deps = []
    for pr in parsed_open_deps:
        if pr not in open_deps and pr not in open_pr_numbers:
            newly_merged_deps.append(pr)

    merged_deps = list(set(parsed_merged_deps + newly_merged_deps))

    if open_deps:
        state = "pending"
        pr_list_str = ", ".join([f"#{num}" for num in open_deps])
        description = f"This PR has open dependencies: {pr_list_str}"
    else:
        state = "success"
        description = "This PR has no open dependencies"

    _set_commit_status(
        pr_node["headRefOid"], state, description, token, dry_run
    )

    first_independent_commit_oid = None
    if open_deps:
        dependent_oids = set()
        for d in open_deps:
            dependent_oids.update(pr_to_commits[d])

        # previous_first_commit already assigned from comment state.
        if previous_first_commit and previous_first_commit in current_oids:
            start_idx = current_oids.index(previous_first_commit)
        else:
            start_idx = 0

        # Assumes `current_oids` is in chronological order (oldest first).
        # This guarantees we find the first independent commit to start the
        # review.
        for oid in current_oids[start_idx:]:
            if oid not in dependent_oids:
                first_independent_commit_oid = oid
                break

        any_later_dependent_oids = False
        if first_independent_commit_oid:
            idx = current_oids.index(first_independent_commit_oid)
            any_later_dependent_oids = any(
                oid in dependent_oids for oid in current_oids[idx + 1 :]
            )

    if (
        open_deps == parsed_open_deps
        and merged_deps == parsed_merged_deps
        and first_independent_commit_oid == previous_first_commit
    ):
        return

    # Construct new comment
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    new_state: dict[str, Any] = {
        "open": open_deps,
        "merged": merged_deps,
        "first_commit": first_independent_commit_oid,
    }
    state_json = json.dumps(new_state)

    comment_body = f"{marker_prefix}{state_json} -->\n"

    if open_deps:
        pr_list_str = ", ".join([f"#{num}" for num in open_deps])
        if first_independent_commit_oid:
            changes_url = (
                "https://github.com/carbon-language/carbon-lang/pull/"
                f"{pr_number}/changes/{first_independent_commit_oid}..HEAD"
            )
            comment_body += (
                f"Depends on {pr_list_str}, start review with "
                f"[these changes]({changes_url})"
            )
            if any_later_dependent_oids:
                comment_body += (
                    "\n\n> [!WARNING]\n"
                    "> Also contains changes from dependent PRs due to "
                    "non-linear history."
                )
        else:
            comment_body += (
                f"Depends on {pr_list_str}, unable to identify starting review "
                f"commit from simple analysis"
            )
    else:
        comment_body += "All dependent PRs are merged."

    if merged_deps:
        merged_str = ", ".join([f"#{num}" for num in sorted(merged_deps)])
        comment_body += f"\n\nMerged dependent PRs: {merged_str}"

    comment_body += f"\n\n(Last updated: {timestamp})"

    _print_err(f"PR #{pr_number}: Updating comment. New body:\n{comment_body}")

    # Apply mutations
    has_dependent_label = any(label["name"] == "dependent" for label in labels)

    if open_deps and not has_dependent_label and not scanning:
        if dry_run:
            _print_err(
                f"[Dry-run] Would add 'dependent' label to PR #{pr_number}"
            )
        else:
            client.execute(
                _MUTATION_ADD_LABEL,
                variable_values={"labelableId": pr_id, "labelIds": [label_id]},
            )
    elif not open_deps and has_dependent_label:
        if dry_run:
            _print_err(
                f"[Dry-run] Would remove 'dependent' label from PR #{pr_number}"
            )
        else:
            client.execute(
                _MUTATION_REMOVE_LABEL,
                variable_values={"labelableId": pr_id, "labelIds": [label_id]},
            )

    if existing_comment_id:
        if dry_run:
            _print_err(f"[Dry-run] Would update comment {existing_comment_id}")
        else:
            client.execute(
                _MUTATION_UPDATE_COMMENT,
                variable_values={
                    "id": existing_comment_id,
                    "body": comment_body,
                },
            )
    else:
        if scanning:
            _print_err(
                f"PR #{pr_number}: Skipping new comment creation in scan mode."
            )
            return
        if dry_run:
            _print_err(f"[Dry-run] Would add comment to PR #{pr_number}")
        else:
            client.execute(
                _MUTATION_ADD_COMMENT,
                variable_values={"subjectId": pr_id, "body": comment_body},
            )


def _parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pr-number",
        type=int,
        help="The pull request number to check.",
    )
    group.add_argument(
        "--scan",
        action="store_true",
        help="Scan all open PRs with 'dependent' label and update them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mutations without updating GitHub",
    )
    github_helpers.add_access_token_arg(parser, "repo")
    return parser.parse_args(args=args)


def main() -> None:
    parsed_args = _parse_args()
    client = github_helpers.Client(parsed_args)

    _print_err("Loading open PRs ...", end="", flush=True)
    pr_to_commits: dict[int, list[str]] = {}
    open_pr_numbers: set[int] = set()
    for node in client.execute_and_paginate(
        _QUERY_OPEN_PRS, ("repository", "pullRequests")
    ):
        _print_err(".", end="", flush=True)
        other_pr_num = node["number"]
        open_pr_numbers.add(other_pr_num)
        pr_to_commits[other_pr_num] = [
            c["commit"]["oid"] for c in node["commits"]["nodes"]
        ]
    _print_err()

    label_res = client.execute(_QUERY_LABEL)
    label_id = label_res["repository"]["label"]["id"]

    merged_res = client.execute(_QUERY_MAX_MERGED_PR)
    merged_nodes = merged_res["repository"]["pullRequests"]["nodes"]
    max_merged_pr = merged_nodes[0]["number"] if merged_nodes else 0

    if parsed_args.pr_number:
        _process_pr(
            client,
            parsed_args.pr_number,
            pr_to_commits,
            open_pr_numbers,
            label_id,
            parsed_args.access_token,
            dry_run=parsed_args.dry_run,
            max_merged_pr=max_merged_pr,
        )
    elif parsed_args.scan:
        for node in client.execute_and_paginate(
            _QUERY_DEPENDENT_PRS, ("repository", "pullRequests")
        ):
            _process_pr(
                client,
                node["number"],
                pr_to_commits,
                open_pr_numbers,
                label_id,
                parsed_args.access_token,
                dry_run=parsed_args.dry_run,
                scanning=True,
                max_merged_pr=max_merged_pr,
            )


if __name__ == "__main__":
    main()
