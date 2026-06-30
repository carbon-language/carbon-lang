#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gql>=2.0.0,<3.0.0",
# ]
# ///


"""Gather Carbon review stats, prints to CSV."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import datetime
import os
import sys
from typing import Any, Optional

# https://pypi.org/project/gql/
import gql
from gql.transport.requests import RequestsHTTPTransport

# The main query. We request all pullRequests in the repository, along with a
# number of different events on each pullRequest. The request is paginated, and
# we use the pageInfo to determine what the `cursor` should be for the next
# request.
_QUERY = """
{
  repository(owner:"carbon-language", name:"carbon-lang") {
    pullRequests(
      first:100,
      states: MERGED,
      labels:"toolchain"
      %(cursor)s
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      edges {
        node {
          ... on PullRequest {
            number
            author {
              login
            }
            createdAt
            mergedAt
            timelineItems(first: 250) {
              nodes {
                __typename
                ... on ReadyForReviewEvent {
                  actor {
                    login
                  }
                  createdAt
                }
                ... on ReviewRequestedEvent {
                  actor {
                    login
                  }
                  createdAt
                  requestedReviewer {
                    ... on User {
                      login
                    }
                  }
                }
                ... on IssueComment {
                  author {
                    login
                  }
                  createdAt
                }
                ... on PullRequestReview {
                  author {
                    login
                  }
                  createdAt
                  state
                }
              }
            }
          }
        }
      }
    }
  }
}"""


_ENV_TOKEN = "GITHUB_ACCESS_TOKEN"


class PRInfo:
    number: int
    author: str
    created_at: datetime.datetime
    ready_for_review_at: datetime.datetime
    merged_at: datetime.datetime
    autoassignee: Optional[str] = None
    first_comment_user: Optional[str] = None
    first_comment_at: Optional[datetime.datetime] = None
    first_approval_user: Optional[str] = None
    first_approval_at: Optional[datetime.datetime] = None

    def __init__(
        self,
        number: int,
        author: str,
        created_at: datetime.datetime,
        merged_at: datetime.datetime,
    ):
        self.number = number
        self.author = author
        self.created_at = created_at
        self.ready_for_review_at = created_at
        self.merged_at = merged_at


def _parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    access_token = os.environ.get(_ENV_TOKEN, default=None)
    parser.add_argument(
        "--access-token",
        metavar="ACCESS_TOKEN",
        default=access_token,
        required=not access_token,
        help="The access token for use with GitHub. May also be specified in "
        f"the environment as {_ENV_TOKEN}.",
    )
    return parser.parse_args(args=args)


def _time(timestamp: str) -> datetime.datetime:
    return datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parsed_args = _parse_args()

    # TODO: This is a gql.transport.aiohttp.AIOHTTPTransport in gql>=3
    transport = RequestsHTTPTransport(
        url="https://api.github.com/graphql",
        headers={"Authorization": "bearer %s" % parsed_args.access_token},
        # TODO: `ssl=True` in gql>=3
    )
    client = gql.Client(
        transport=transport,
        # TODO: `execute_timeout=60` in gql>=3
    )

    cursor = ""

    prs = []

    print("querying PRs...", file=sys.stderr)
    while True:
        retries = 3
        while True:
            try:
                raw_result = client.execute(
                    gql.gql(request_string=_QUERY % {"cursor": cursor})
                )
                break
            # TODO: This is a gql.transport.TransportServerError in gql>=3
            except Exception:
                if retries == 0:
                    raise
                retries -= 1
                print(f"retrying ({retries} more)...", file=sys.stderr)
        result = raw_result["repository"]["pullRequests"]

        last = None
        for pr_edge in result["edges"]:
            pr = pr_edge["node"]
            last = pr["number"]

            pr_info = PRInfo(
                number=int(pr["number"]),
                author=pr["author"]["login"],
                created_at=_time(pr["createdAt"]),
                merged_at=_time(pr["mergedAt"]),
            )

            for event in pr["timelineItems"]["nodes"]:
                if (
                    not pr_info.first_comment_user
                    and event["__typename"]
                    in ("IssueComment", "PullRequestReview")
                    and event["author"]
                    and event["author"]["login"]
                    not in (
                        pr_info.author,
                        "CarbonInfraBot",
                        "github-actions",
                        "gemini-code-assist",
                    )
                ):
                    pr_info.first_comment_user = event["author"]["login"]
                    pr_info.first_comment_at = _time(event["createdAt"])

                if event["__typename"] == "ReadyForReviewEvent":
                    pr_info.ready_for_review_at = _time(event["createdAt"])

                if (
                    not pr_info.first_approval_user
                    and event["__typename"] == "PullRequestReview"
                    and event["state"] == "APPROVED"
                ):
                    pr_info.first_approval_user = event["author"]["login"]
                    pr_info.first_approval_at = _time(event["createdAt"])

                if (
                    not pr_info.autoassignee
                    and event["__typename"] == "ReviewRequestedEvent"
                    and event["actor"]["login"] == "github-actions"
                ):
                    pr_info.autoassignee = event["requestedReviewer"]["login"]
                    continue

            prs.append(pr_info)

        page_info = result["pageInfo"]
        if page_info["hasNextPage"]:
            cursor = f'after: "{page_info["endCursor"]}"'
            print(f"continuing from {last}...", file=sys.stderr)
        else:
            break

    def for_csv(x: Any) -> str:
        return str(x) if x else ""

    print(
        "PR#,Author,Created At,Ready For Review At,Merged At,Autoassignee,"
        "First Comment User,First Comment At,First Approval User,"
        "First Approval At"
    )
    for pr in prs:
        print(
            f"{pr.number},{pr.author},{pr.created_at},"
            f"{pr.ready_for_review_at},{pr.merged_at},"
            f"{for_csv(pr.autoassignee)},{for_csv(pr.first_comment_user)},"
            f"{for_csv(pr.first_comment_at)},{for_csv(pr.first_approval_user)},"
            f"{for_csv(pr.first_approval_at)}"
        )


if __name__ == "__main__":
    main()
