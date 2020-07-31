#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

import argparse
import datetime
import hashlib
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
      %(reviews)s
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

_QUERY_REVIEWS = """
      reviews(first: 100%s) {
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
              originalCommit {
                abbreviatedOid
              }
              path
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
    def _rewrap(content):
        """Rewraps a comment to fit in 80 columns with an indent."""
        lines = []
        for line in content.split("\n"):
            lines.extend(
                [
                    x
                    for x in textwrap.wrap(
                        line,
                        width=80,
                        initial_indent=" " * 4,
                        subsequent_indent=" " * 4,
                    )
                ]
            )
        return "\n".join(lines)

    def format(self, long):
        """Formats the comment."""
        if long:
            return "%s%s at %s:\n%s" % (
                " " * 2,
                self.author,
                self.timestamp.strftime("%Y-%m-%d %H:%M"),
                self._rewrap(self.body),
            )
        else:
            # Compact newlines down into pilcrows, leaving a space after.
            body = self.body.replace("\r", "").replace("\n", "¶ ")
            while "¶ ¶" in body:
                body = body.replace("¶ ¶", "¶¶")
            line = "%s%s: %s" % (" " * 2, self.author, body)
            return line if len(line) <= 80 else line[:77] + "..."


class _PRComment(_Comment):
    """A comment on the top-level PR."""

    def __init__(self, raw_comment):
        super().__init__(
            raw_comment["author"]["login"],
            raw_comment["createdAt"],
            raw_comment["body"],
        )
        self.url = raw_comment["url"]

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def format(self, long):
        return "%s\n%s" % (self.url, super().format(long))


class _Thread(object):
    """A review thread on a line of code."""

    def __init__(self, parsed_args, thread):
        self.is_resolved = thread["isResolved"]

        comments = thread["comments"]["nodes"]
        first_comment = comments[0]
        self.line = first_comment["originalPosition"]
        self.path = first_comment["path"]

        # Link to the comment in the commit; GitHub features work better there
        # than in the conversation view. The diff_url allows viewing changes
        # since the comment, although the comment won't be visible there.
        template = (
            "https://github.com/carbon-language/%(repo)s/pull/%(pr_num)s/"
            "files/%(oid)s%(head)s#diff-%(path_md5)s%(line_side)s%(line)s"
        )
        # GitHub uses an md5 of the file's path for the link.
        path_md5 = hashlib.md5()
        path_md5.update(bytearray(self.path, "utf-8"))
        format_dict = {
            "head": "",
            "line_side": "R",
            "line": self.line,
            "oid": first_comment["originalCommit"]["abbreviatedOid"],
            "path_md5": path_md5.hexdigest(),
            "pr_num": parsed_args.pr_num,
            "repo": parsed_args.repo,
        }
        self.url = template % format_dict
        format_dict["head"] = "..HEAD"
        format_dict["line_side"] = "L"
        self.diff_url = template % format_dict

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
        lines.append(
            "%s\n  - line %d; %s"
            % (
                self.url,
                self.line,
                ("resolved" if self.is_resolved else "unresolved"),
            )
        )
        if self.diff_url:
            lines.append("  - diff: %s" % self.diff_url)
        for comment in self.comments:
            lines.append(comment.format(long))
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

    field_name and field should be specified for cursor-based queries.
    """
    print(".", end="", flush=True)
    format_inputs = {
        "pr_num": parsed_args.pr_num,
        "repo": parsed_args.repo,
        "comments": "",
        "review_threads": "",
        "reviews": "",
    }
    if field:
        # Use a cursor for pagination of the field.
        cursor = ', after: "%s"' % field["pageInfo"]["endCursor"]
        if field_name == "comments":
            format_inputs["comments"] = _QUERY_COMMENTS % cursor
        elif field_name == "reviewThreads":
            format_inputs["review_threads"] = _QUERY_REVIEW_THREADS % cursor
        elif field_name == "reviews":
            format_inputs["reviews"] = _QUERY_REVIEWS % cursor
        else:
            raise ValueError("Unexpected field_name: %s" % field_name)
    else:
        # Fetch the first page of all fields.
        format_inputs["comments"] = _QUERY_COMMENTS % ""
        format_inputs["review_threads"] = _QUERY_REVIEW_THREADS % ""
        format_inputs["reviews"] = _QUERY_REVIEWS % ""
    return client.execute(gql.gql(_QUERY % format_inputs))


def _accumulate_pr_comments(parsed_args, comments, raw_comments):
    """Collects top-level comments and reviews."""
    for raw_comment in raw_comments:
        # Elide reviews that have no top-level comment body.
        if not raw_comment["body"]:
            continue
        comments.append(_PRComment(raw_comment))


def _accumulate_threads(parsed_args, threads_by_path, raw_threads):
    """Adds threads to threads_by_path for later sorting."""
    for raw_thread in raw_threads:
        thread = _Thread(parsed_args, raw_thread)

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
        % (parsed_args.repo, parsed_args.pr_num),
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

    # Paginate comments, reviews, and review threads.
    comments = []
    _paginate(
        "comments",
        _accumulate_pr_comments,
        parsed_args,
        client,
        pull_request,
        comments,
    )
    # Combine reviews into comments for interleaving.
    _paginate(
        "reviews",
        _accumulate_pr_comments,
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

    for comment in sorted(comments):
        print()
        print(comment.format(parsed_args.long))

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
