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
import os
import sys
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
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequest(number: %d) {
      id
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


def _print_err(*args: Any, **kwargs: Any) -> None:
    """Prints to stderr."""
    kwargs["file"] = sys.stderr
    print(*args, **kwargs)


def _process_pr(
    client: github_helpers.Client,
    pr_number: int,
    commit_to_prs: dict[str, set[int]],
    open_pr_numbers: set[int],
    label_id: str,
    dry_run: bool,
    scanning: bool = False,
) -> None:
    """Processes a single PR to check for dependencies and update comments."""
    current_res = client.execute(_QUERY_PR_DETAILS % pr_number)
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
        commits_to_check = current_oids[:-1]

        for oid in commits_to_check:
            for other_pr in commit_to_prs.get(oid, ()):
                if other_pr not in open_deps and other_pr != pr_number:
                    open_deps.append(other_pr)

    # Parse existing comment
    marker_prefix = "<!-- check_dependent_pr "
    existing_comment_id = None
    state: dict[str, list[int]] = {"open": [], "merged": []}

    for comment in comments:
        body = comment["body"]
        if marker_prefix in body:
            existing_comment_id = comment["id"]
            try:
                start = body.find(marker_prefix) + len(marker_prefix)
                end = body.find(" -->", start)
                state = json.loads(body[start:end])
            except Exception as e:
                _print_err(
                    f"Error parsing marker JSON in PR #{pr_number} "
                    f"comment {existing_comment_id}: {e}"
                )
                return
            break

    if not open_deps and not existing_comment_id:
        return

    # Identify newly merged PRs
    newly_merged_deps = []
    for pr in state.get("open", []):
        if pr not in open_deps and pr not in open_pr_numbers:
            newly_merged_deps.append(pr)

    merged_deps = list(set(state.get("merged", []) + newly_merged_deps))

    if open_deps == state.get("open") and merged_deps == state.get("merged"):
        return

    # Construct new comment
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    new_state = {"open": open_deps, "merged": merged_deps}
    state_json = json.dumps(new_state)

    comment_body = f"{marker_prefix}{state_json} -->\n"

    if open_deps:
        first_independent_commit_oid = None
        for oid in current_oids:
            if oid not in commit_to_prs:
                first_independent_commit_oid = oid
                break
        if not first_independent_commit_oid:
            first_independent_commit_oid = current_oids[-1]

        short_hash = first_independent_commit_oid[:8]
        pr_list_str = ", ".join([f"#{num}" for num in open_deps])
        first_commit_linked = (
            f"[{short_hash}]({pr_number}/commits/{short_hash})"
        )
        comment_body += (
            f"Depends on {pr_list_str}, start review at {first_commit_linked}"
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
        mutation_label = """
        mutation {
          addLabelsToLabelable(input: {labelableId: "%s", labelIds: ["%s"]}) {
            clientMutationId
          }
        }
        """ % (
            pr_id,
            label_id,
        )
        if dry_run:
            _print_err(
                f"[Dry-run] Would add 'dependent' label to PR #{pr_number}"
            )
        else:
            client.execute(mutation_label)
    elif not open_deps and has_dependent_label:
        mutation_remove_label = """
        mutation {
          removeLabelsFromLabelable(
            input: {labelableId: "%s", labelIds: ["%s"]}
          ) {
            clientMutationId
          }
        }
        """ % (
            pr_id,
            label_id,
        )
        if dry_run:
            _print_err(
                f"[Dry-run] Would remove 'dependent' label from PR #{pr_number}"
            )
        else:
            client.execute(mutation_remove_label)

    safe_comment_body = comment_body.replace('"', '\\"')
    if existing_comment_id:
        mutation_comment = """
        mutation {
          updateIssueComment(input: {id: "%s", body: "%s"}) {
            clientMutationId
          }
        }
        """ % (
            existing_comment_id,
            safe_comment_body,
        )
        if dry_run:
            _print_err(f"[Dry-run] Would update comment {existing_comment_id}")
        else:
            client.execute(mutation_comment)
    else:
        if scanning:
            _print_err(
                f"PR #{pr_number}: Skipping new comment creation in scan mode."
            )
            return
        mutation_comment = """
        mutation {
          addComment(input: {subjectId: "%s", body: "%s"}) {
            clientMutationId
          }
        }
        """ % (
            pr_id,
            safe_comment_body,
        )
        if dry_run:
            _print_err(f"[Dry-run] Would add comment to PR #{pr_number}")
        else:
            client.execute(mutation_comment)


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
    commit_to_prs: dict[str, set[int]] = {}
    open_pr_numbers: set[int] = set()
    for node in client.execute_and_paginate(
        _QUERY_OPEN_PRS, ("repository", "pullRequests")
    ):
        _print_err(".", end="", flush=True)
        other_pr_num = node["number"]
        open_pr_numbers.add(other_pr_num)
        for c in node["commits"]["nodes"]:
            oid = c["commit"]["oid"]
            if oid not in commit_to_prs:
                commit_to_prs[oid] = set()
            commit_to_prs[oid].add(other_pr_num)
    _print_err()

    label_res = client.execute(_QUERY_LABEL)
    label_id = label_res["repository"]["label"]["id"]

    if parsed_args.pr_number:
        _process_pr(
            client,
            parsed_args.pr_number,
            commit_to_prs,
            open_pr_numbers,
            label_id,
            parsed_args.dry_run,
        )
    elif parsed_args.scan:
        for node in client.execute_and_paginate(
            _QUERY_DEPENDENT_PRS, ("repository", "pullRequests")
        ):
            _process_pr(
                client,
                node["number"],
                commit_to_prs,
                open_pr_numbers,
                label_id,
                parsed_args.dry_run,
                scanning=True,
            )


if __name__ == "__main__":
    main()
