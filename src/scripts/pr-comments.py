#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

import argparse
import gql
import gql.transport.requests
import os
import sys
import textwrap

# Use https://developer.github.com/v4/explorer/ to help with edits.
_QUERY = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequest(number: %d) {
      reviewThreads(first: 100) {
        totalCount
        nodes {
          comments(first: 100) {
            nodes {
              body
              author {
                login
              }
              url
            }
          }
          isResolved
          resolvedBy {
            login
          }
        }
      }
      author {
        login
      }
      title
    }
  }
}
"""


def parse_args():
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description="Lists comments on a PR.")
    parser.add_argument(
        "pr_num",
        metavar="PR#",
        type=int,
        nargs=1,
        help="The pull request to fetch comments from.",
    )
    parser.add_argument(
        "--github-access-token",
        metavar="ACCESS_TOKEN",
        help="The access token for use with GitHub. May also be specified in "
        "the environment as GITHUB_ACCESS_TOKEN.",
    )
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="Whether to include resolved review threads. By default, only "
        "unresolved threads will be shown.",
    )
    return parser.parse_args()


def rewrap(content):
    indent = "    "
    lines = []
    for line in content.split("\n"):
        for x in textwrap.wrap(
            line, width=80, initial_indent=indent, subsequent_indent=indent
        ):
            lines.append(x)
    return "\n".join(lines)


def main():
    parsed_args = parse_args()
    pr_num = parsed_args.pr_num[0]
    access_token = parsed_args.github_access_token
    include_resolved = parsed_args.include_resolved
    if not access_token:
        if "GITHUB_ACCESS_TOKEN" not in os.environ:
            sys.exit(
                "Missing github access token. This must be provided through "
                "either --github-access-token or GITHUB_ACCESS_TOKEN."
            )
        access_token = os.environ["GITHUB_ACCESS_TOKEN"]

    transport = gql.transport.requests.RequestsHTTPTransport(
        url="https://api.github.com/graphql",
        headers={"Authorization": "bearer %s" % access_token},
    )
    client = gql.Client(transport=transport, fetch_schema_from_transport=True)
    threads_result = client.execute(gql.gql(_QUERY % pr_num))
    pull_request = threads_result["repository"]["pullRequest"]
    print(
        "'%s' (%d) by %s"
        % (pull_request["title"], pr_num, pull_request["author"]["login"])
    )

    for thread in pull_request["reviewThreads"]["nodes"]:
        resolved = thread["isResolved"]
        if resolved and not include_resolved:
            continue
        print("\nThread (%s)" % ("resolved" if resolved else "unresolved"))
        print("    %s" % thread["comments"]["nodes"][0]["url"])
        for comment in thread["comments"]["nodes"]:
            print("  %s:" % comment["author"]["login"])
            print(rewrap(comment["body"]))
        if resolved:
            print("  %s:\n    RESOLVED" % thread["resolvedBy"]["login"])


if __name__ == "__main__":
    main()
