#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

import argparse
import os
import sys
import textwrap

# https://pypi.org/project/gql/
import gql
import gql.transport.requests

# Use https://developer.github.com/v4/explorer/ to help with edits.
_QUERY = """
{
  repository(owner: "carbon-language", name: "carbon-lang") {
    pullRequest(number: %d) {
      reviewThreads(first: 100%s) {
        nodes {
          comments(first: 100) {
            nodes {
              body
              author {
                login
              }
              path
              originalPosition
              url
            }
          }
          isResolved
          resolvedBy {
            login
          }
        }
        pageInfo {
          endCursor
          hasNextPage
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
    # TODO: Add flag to filter for review threads including a specific user.
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


def accumulate_threads(threads_by_path, review_threads, include_resolved):
    """Adds threads to threads_by_path for later sorting."""
    for thread in review_threads["nodes"]:
        if thread["isResolved"] and not include_resolved:
            continue

        first_comment = thread["comments"]["nodes"][0]
        path = first_comment["path"]
        line = first_comment["originalPosition"]
        if path not in threads_by_path:
            threads_by_path[path] = []
        threads_by_path[path].append((line, thread))


def rewrap(content):
    """Rewraps a comment to fit in 80 columns with a 4-space indent."""
    indent = "    "
    lines = []
    for line in content.split("\n"):
        for x in textwrap.wrap(
            line, width=80, initial_indent=indent, subsequent_indent=indent
        ):
            lines.append(x)
    return "\n".join(lines)


def main():
    # Parse command-line flags.
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

    # Prepare the GraphQL client.
    transport = gql.transport.requests.RequestsHTTPTransport(
        url="https://api.github.com/graphql",
        headers={"Authorization": "bearer %s" % access_token},
    )
    client = gql.Client(transport=transport, fetch_schema_from_transport=True)

    # Get the initial set of review threads, and print the PR summary.
    threads_result = client.execute(gql.gql(_QUERY % (pr_num, "")))
    pull_request = threads_result["repository"]["pullRequest"]
    print(
        "'%s' (%d) by %s"
        % (pull_request["title"], pr_num, pull_request["author"]["login"])
    )

    # Paginate through the review threads.
    threads_by_path = {}
    while True:
        # Accumulate the review threads.
        review_threads = pull_request["reviewThreads"]
        accumulate_threads(threads_by_path, review_threads, include_resolved)
        if not review_threads["pageInfo"]["hasNextPage"]:
            break
        # There are more review threads, so fetch them.
        threads_result = client.execute(
            gql.gql(
                _QUERY
                % (
                    pr_num,
                    ', after: "%s"' % review_threads["pageInfo"]["endCursor"],
                )
            )
        )
        pull_request = threads_result["repository"]["pullRequest"]

    # Print threads sorted by path and line.
    for path in sorted(threads_by_path.keys()):
        # Print a header for each path.
        print()
        print("=" * 80)
        print(path)
        print("=" * 80)

        for line, thread in sorted(threads_by_path[path], key=lambda x: x[0]):
            resolved = thread["isResolved"]
            # Print a header for each thread.
            # TODO: Add flag to fetch/print diffHunk for more context.
            print()
            print(
                "line %d; %s"
                % (line, ("resolved" if resolved else "unresolved"))
            )
            # TODO: Try to link to the review thread with an appropriate diff.
            # Ideally comment-to-present, worst case original-to-comment (to see
            # comment). Possibly both.
            print("    %s" % thread["comments"]["nodes"][0]["url"])
            # TODO: Add a short comment mode that does comment-per-line.
            # TODO: Timestamps would be nice.
            for comment in thread["comments"]["nodes"]:
                print("  %s:" % comment["author"]["login"])
                print(rewrap(comment["body"]))
            if resolved:
                print("  %s:\n    RESOLVED" % thread["resolvedBy"]["login"])


if __name__ == "__main__":
    main()
