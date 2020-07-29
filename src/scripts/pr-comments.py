#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

import argparse
import datetime
import os
import re
import sys
import textwrap

# https://pypi.org/project/gql/
import gql
import gql.transport.requests

# Use https://developer.github.com/v4/explorer/ to help with edits.
_QUERY = """
{
  repository(owner: "carbon-language", name: "%(repo)s") {
    pullRequest(number: %(pr_num)d) {
      reviewThreads(first: 100%(review_threads_cursor)s) {
        nodes {
          comments(first: 100) {
            nodes {
              author {
                login
              }
              body
              createdAt
              originalPosition
              path
              url
            }
          }
          isResolved
          resolvedBy {
            createdAt
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


def _parse_args():
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description="Lists comments on a PR.")
    parser.add_argument(
        "pr_num",
        metavar="PR#",
        type=int,
        nargs=1,
        help="The pull request to fetch comments from.",
    )
    env_token = "GITHUB_ACCESS_TOKEN"
    parser.add_argument(
        "--access-token",
        metavar="ACCESS_TOKEN",
        default=os.environ.get(env_token, default=None),
        help="The access token for use with GitHub. May also be specified in "
        "the environment as %s." % env_token,
    )
    parser.add_argument(
        "--comments-after",
        metavar="LOGIN",
        help="Only print threads where the final comment is not from the given "
        "user. For example, use when looking for threads that you still need "
        "to respond to.",
    )
    parser.add_argument(
        "--comments-from",
        metavar="LOGIN",
        help="Only print threads with comments from the given user. For "
        "example, use when looking for threads that you've commented on.",
    )
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="Whether to include resolved review threads. By default, only "
        "unresolved threads will be shown.",
    )
    parser.add_argument(
        "--repo",
        choices=["carbon-lang", "carbon-toolchain"],
        default="carbon-lang",
        help="The Carbon repo to query. Defaults to carbon-toolchain.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Prints long output, with the full comment.",
    )
    parsed_args = parser.parse_args()
    if not parsed_args.access_token:
        sys.exit(
            "Missing github access token. This must be provided through "
            "either --github-access-token or %s." % env_token
        )
    return parsed_args


def _query(parsed_args, client, review_threads=None):
    """Queries for comments.

    review_threads may be specified if a cursor is present.
    """
    print(".", end="", flush=True)
    fields = {
        "repo": parsed_args.repo,
        "pr_num": parsed_args.pr_num[0],
        "review_threads_cursor": "",
    }
    if review_threads:
        fields["review_threads_cursor"] = (
            ', after: "%s"' % review_threads["pageInfo"]["endCursor"]
        )
    return client.execute(gql.gql(_QUERY % fields))


def _has_comment_from(comments, comments_from):
    """Returns true if comments has a comment from comments_from."""
    for comment in comments:
        if comment["author"]["login"] == comments_from:
            return True
    return False


def _accumulate_threads(parsed_args, threads_by_path, review_threads):
    """Adds threads to threads_by_path for later sorting."""
    for thread in review_threads["nodes"]:
        # Optionally skip resolved threads.
        if not parsed_args.include_resolved and thread["isResolved"]:
            continue

        comments = thread["comments"]["nodes"]

        # Optionally skip threads where the given user isn't the last commenter.
        if (
            parsed_args.comments_after
            and comments[-1]["author"]["login"] == parsed_args.comments_after
        ):
            continue

        # Optionally skip threads where the given user hasn't commented.
        if parsed_args.comments_from and not _has_comment_from(
            comments, parsed_args.comments_from
        ):
            continue

        first_comment = comments[0]
        path = first_comment["path"]
        line = first_comment["originalPosition"]
        if path not in threads_by_path:
            threads_by_path[path] = []
        threads_by_path[path].append((line, thread))


def _fetch_comments(parsed_args):
    """Fetches comments from GitHub."""
    # Each _query call will print a '.' for progress.
    print(
        "Loading https://github.com/carbon-language/%s/pull/%d ..."
        % (parsed_args.repo, parsed_args.pr_num[0]),
        end="",
        flush=True,
    )

    # Prepare the GraphQL client.
    transport = gql.transport.requests.RequestsHTTPTransport(
        url="https://api.github.com/graphql",
        headers={"Authorization": "bearer %s" % parsed_args.access_token},
    )
    client = gql.Client(transport=transport, fetch_schema_from_transport=True)

    # Get the initial set of review threads, and print the PR summary.
    threads_result = _query(parsed_args, client)
    pull_request = threads_result["repository"]["pullRequest"]

    # Paginate through the review threads.
    threads_by_path = {}
    while True:
        # Accumulate the review threads.
        review_threads = pull_request["reviewThreads"]
        _accumulate_threads(parsed_args, threads_by_path, review_threads)
        if not review_threads["pageInfo"]["hasNextPage"]:
            break
        # There are more review threads, so fetch them.
        threads_result = _query(
            parsed_args, client, review_threads=review_threads
        )
        pull_request = threads_result["repository"]["pullRequest"]

    # Now that loading is done (no more progress indicators), print the header.
    print(
        "\n'%s' by %s"
        % (pull_request["title"], pull_request["author"]["login"])
    )
    return threads_by_path


def _rewrap(content):
    """Rewraps a comment to fit in 80 columns with a 4-space indent."""
    indent = "    "
    lines = []
    for line in content.split("\n"):
        for x in textwrap.wrap(
            line, width=80, initial_indent=indent, subsequent_indent=indent
        ):
            lines.append(x)
    return "\n".join(lines)


def _print_comment(parsed_args, author, created_at, body):
    """Prints a given comment."""
    if parsed_args.long:
        time = datetime.datetime.strptime(
            created_at["createdAt"], "%Y-%m-%dT%H:%M:%SZ"
        )
        print(
            "  %s at %s:" % (author["login"], time.strftime("%Y-%m-%d %H:%M"))
        )
        print(_rewrap(body))
    else:
        # Compact newlines down into pilcrows, leaving a space after.
        body = body.replace("\r", "").replace("\n", "¶ ")
        while "¶ ¶" in body:
            body = body.replace("¶ ¶", "¶¶")
        line = "  %s: %s" % (author["login"], body)
        print(line if len(line) <= 80 else line[:77] + "...")


def main():
    parsed_args = _parse_args()
    threads_by_path = _fetch_comments(parsed_args)

    # TODO: PR-level comments

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
            for comment in thread["comments"]["nodes"]:
                _print_comment(
                    parsed_args, comment["author"], comment, comment["body"]
                )
            if resolved:
                _print_comment(
                    parsed_args,
                    thread["resolvedBy"],
                    thread["resolvedBy"],
                    "RESOLVED",
                )


if __name__ == "__main__":
    main()
