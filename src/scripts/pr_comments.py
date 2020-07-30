#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

import argparse
import datetime
import os
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
      author {
        login
      }
      title

      %(comments)s
      %(review_threads)s
    }
  }
}
"""

_QUERY_COMMENTS = """
      comments(first: 100%s) {
        nodes {
          author {
            login
          }
          body
          createdAt
          url
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
"""

_QUERY_REVIEW_THREADS = """
      reviewThreads(first: 100%s) {
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
"""


class _Comment(object):
    """A comment, either on a review thread or top-level on the PR."""

    def __init__(self, author, timestamp, body):
        self.author = author
        self.timestamp = datetime.datetime.strptime(
            timestamp, "%Y-%m-%dT%H:%M:%SZ"
        )
        self.body = body

    @staticmethod
    def from_raw_comment(raw_comment):
        """Creates the comment from a raw comment dict."""
        return _Comment(
            raw_comment["author"]["login"],
            raw_comment["createdAt"],
            raw_comment["body"],
        )

    @staticmethod
    def _rewrap(content, indent):
        """Rewraps a comment to fit in 80 columns with a 4-space indent."""
        lines = []
        for line in content.split("\n"):
            lines.extend(
                [
                    x
                    for x in textwrap.wrap(
                        line,
                        width=80,
                        initial_indent=" " * indent,
                        subsequent_indent=" " * indent,
                    )
                ]
            )
        return "\n".join(lines)

    def format(self, long, indent):
        """Formats the comment."""
        if long:
            return "%s%s at %s:\n%s" % (
                " " * indent,
                self.author,
                self.timestamp.strftime("%Y-%m-%d %H:%M"),
                self._rewrap(self.body, indent + 2),
            )
        else:
            # Compact newlines down into pilcrows, leaving a space after.
            body = self.body.replace("\r", "").replace("\n", "¶ ")
            while "¶ ¶" in body:
                body = body.replace("¶ ¶", "¶¶")
            line = "%s%s: %s" % (" " * indent, self.author, body)
            return line if len(line) <= 80 else line[:77] + "..."


class _Thread(object):
    """A review thread on a line of code."""

    def __init__(self, thread):
        self.is_resolved = thread["isResolved"]

        comments = thread["comments"]["nodes"]
        self.line = comments[0]["originalPosition"]
        self.path = comments[0]["path"]
        self.url = comments[0]["url"]

        self.comments = [
            _Comment.from_raw_comment(comment)
            for comment in thread["comments"]["nodes"]
        ]
        if self.is_resolved:
            self.comments.append(
                _Comment(
                    thread["resolvedBy"]["login"],
                    thread["resolvedBy"]["createdAt"],
                    "<resolved>",
                )
            )

    def __lt__(self, other):
        """Sort threads by line then timestamp."""
        if self.line != other.line:
            return self.line < other.line
        return self.comments[0].timestamp < other.comments[0].timestamp

    def format(self, long):
        """Formats the review thread with comments."""
        lines = []
        # TODO: Add flag to fetch/print diffHunk for more context.
        lines.append(
            "line %d; %s"
            % (self.line, ("resolved" if self.is_resolved else "unresolved"),)
        )
        # TODO: Try to link to the review thread with an appropriate diff.
        # Ideally comment-to-present, worst case original-to-comment (to see
        # comment). Possibly both.
        lines.append("    %s" % self.url)
        for comment in self.comments:
            lines.append(comment.format(long, 2))
        return "\n".join(lines)

    def has_comment_from(self, comments_from):
        """Returns true if comments has a comment from comments_from."""
        for comment in self.comments:
            if comment.author == comments_from:
                return True
        return False


def _parse_args(args=None):
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
        help="The Carbon repo to query. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Prints long output, with the full comment.",
    )
    parsed_args = parser.parse_args(args=args)
    if not parsed_args.access_token:
        sys.exit(
            "Missing github access token. This must be provided through "
            "either --github-access-token or %s." % env_token
        )
    return parsed_args


def _query(parsed_args, client, field_name=None, field=None):
    """Queries for comments.

    review_threads may be specified if a cursor is present.
    """
    print(".", end="", flush=True)
    fields = {
        "pr_num": parsed_args.pr_num[0],
        "repo": parsed_args.repo,
        "comments": "",
        "review_threads": "",
    }
    if field:
        # Use a cursor for pagination of the field.
        cursor = ', after: "%s"' % field["pageInfo"]["endCursor"]
        if field_name == "comments":
            fields["comments"] = _QUERY_COMMENTS % cursor
        elif field_name == "reviewThreads":
            fields["review_threads"] = _QUERY_REVIEW_THREADS % cursor
        else:
            raise ValueError("Unexpected field_name: %s" % field_name)
    else:
        # Fetch the first page of both fields.
        fields["comments"] = _QUERY_COMMENTS % ""
        fields["review_threads"] = _QUERY_REVIEW_THREADS % ""
    return client.execute(gql.gql(_QUERY % fields))


def _accumulate_comments(parsed_args, comments, raw_comments):
    """Collects top-level comments."""
    for raw_comment in raw_comments:
        comments.append(_Comment.from_raw_comment(raw_comment))


def _accumulate_threads(parsed_args, threads_by_path, raw_threads):
    """Adds threads to threads_by_path for later sorting."""
    for raw_thread in raw_threads:
        thread = _Thread(raw_thread)

        # Optionally skip resolved threads.
        if not parsed_args.include_resolved and thread.is_resolved:
            continue

        # Optionally skip threads where the given user isn't the last commenter.
        if (
            parsed_args.comments_after
            and thread.comments[-1].author == parsed_args.comments_after
        ):
            continue

        # Optionally skip threads where the given user hasn't commented.
        if parsed_args.comments_from and not thread.has_comment_from(
            parsed_args.comments_from
        ):
            continue

        if thread.path not in threads_by_path:
            threads_by_path[thread.path] = []
        threads_by_path[thread.path].append(thread)


def _paginate(
    field_name, accumulator, parsed_args, client, pull_request, output
):
    """Paginates through the given field_name, accumulating results."""
    while True:
        # Accumulate the review threads.
        field = pull_request[field_name]
        accumulator(parsed_args, output, field["nodes"])
        if not field["pageInfo"]["hasNextPage"]:
            break
        # There are more review threads, so fetch them.
        next_page = _query(
            parsed_args, client, field_name=field_name, field=field
        )
        pull_request = next_page["repository"]["pullRequest"]


def _fetch_comments(parsed_args):
    """Fetches comments and review threads from GitHub."""
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

    # Paginate comments and review threads.
    comments = []
    _paginate(
        "comments",
        _accumulate_comments,
        parsed_args,
        client,
        pull_request,
        comments,
    )
    threads_by_path = {}
    _paginate(
        "reviewThreads",
        _accumulate_threads,
        parsed_args,
        client,
        pull_request,
        threads_by_path,
    )

    # Now that loading is done (no more progress indicators), print the header.
    print(
        "\n  '%s' by %s"
        % (pull_request["title"], pull_request["author"]["login"])
    )
    return comments, threads_by_path


def main():
    parsed_args = _parse_args()
    comments, threads_by_path = _fetch_comments(parsed_args)

    print()
    for comment in comments:
        print(comment.format(parsed_args.long, 0))

    for path, threads in sorted(threads_by_path.items()):
        # Print a header for each path.
        print()
        print("=" * 80)
        print(path)
        print("=" * 80)

        for thread in sorted(threads):
            print()
            print(thread.format(parsed_args.long))


if __name__ == "__main__":
    main()
