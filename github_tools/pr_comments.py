#!/usr/bin/env python3

"""Figure out comments on a GitHub PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import datetime
import hashlib
import os
import importlib.util
import textwrap
from typing import Any, Dict, Callable, List, Optional, Tuple


# Do some extra work to support direct runs.
try:
    from github_tools import github_helpers
except ImportError:
    github_helpers_spec = importlib.util.spec_from_file_location(
        "github_helpers",
        os.path.join(os.path.dirname(__file__), "github_helpers.py"),
    )
    github_helpers = importlib.util.module_from_spec(github_helpers_spec)
    github_helpers_spec.loader.exec_module(github_helpers)  # type: ignore


# The main query, into which other queries are composed.
_QUERY = """
{
  repository(owner: "carbon-language", name: "%(repo)s") {
    pullRequest(number: %(pr_num)d) {
      author {
        login
      }
      createdAt
      title

      %(comments)s
      %(reviews)s
      %(review_threads)s
    }
  }
}
"""

# Queries for comments on the PR. These are direct, non-review comments on the
# PR.
_QUERY_COMMENTS = """
      comments(first: 100%(cursor)s) {
        nodes {
          author {
            login
          }
          body
          createdAt
          url
        }
        %(pagination)s
      }
"""

# Queries for reviews on the PR, which have a non-empty body if a review has
# a summary comment.
_QUERY_REVIEWS = """
      reviews(first: 100%(cursor)s) {
        nodes {
          author {
            login
          }
          body
          createdAt
          url
        }
        %(pagination)s
      }
"""

# Queries for review threads on the PR.
_QUERY_REVIEW_THREADS = """
      reviewThreads(first: 100%(cursor)s) {
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
        %(pagination)s
      }
"""


class _Comment(object):
    """A comment, either on a review thread or top-level on the PR."""

    def __init__(self, author: str, timestamp: str, body: str):
        self.author = author
        self.timestamp = datetime.datetime.strptime(
            timestamp, "%Y-%m-%dT%H:%M:%SZ"
        )
        self.body = body

    @staticmethod
    def from_raw_comment(raw_comment: Dict) -> "_Comment":
        """Creates the comment from a raw comment dict."""
        return _Comment(
            raw_comment["author"]["login"],
            raw_comment["createdAt"],
            raw_comment["body"],
        )

    @staticmethod
    def _rewrap(content: str) -> str:
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

    def format(self, long: bool) -> str:
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

    def __init__(self, raw_comment: Dict):
        super().__init__(
            raw_comment["author"]["login"],
            raw_comment["createdAt"],
            raw_comment["body"],
        )
        self.url = raw_comment["url"]

    def __lt__(self, other: "_PRComment") -> bool:
        return self.timestamp < other.timestamp

    def format(self, long: bool) -> str:
        return "%s\n%s" % (self.url, super().format(long))


class _Thread(object):
    """A review thread on a line of code."""

    def __init__(self, parsed_args: argparse.Namespace, thread: Dict):
        self.is_resolved: bool = thread["isResolved"]

        comments = thread["comments"]["nodes"]
        first_comment = comments[0]
        self.line: int = first_comment["originalPosition"]
        self.path: str = first_comment["path"]

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
        self.url: str = template % format_dict
        format_dict["head"] = "..HEAD"
        format_dict["line_side"] = "L"
        self.diff_url: str = template % format_dict

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

    def __lt__(self, other: "_Thread") -> bool:
        """Sort threads by line then timestamp."""
        if self.line != other.line:
            return bool(self.line < other.line)
        return self.comments[0].timestamp < other.comments[0].timestamp

    def format(self, long: bool) -> str:
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

    def has_comment_from(self, comments_from: str) -> bool:
        """Returns true if comments has a comment from comments_from."""
        for comment in self.comments:
            if comment.author == comments_from:
                return True
        return False


def _parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description="Lists comments on a PR.")
    parser.add_argument(
        "pr_num",
        metavar="PR#",
        type=int,
        help="The pull request to fetch comments from.",
    )
    github_helpers.add_access_token_arg(parser, "repo")
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
        choices=["carbon-lang"],
        default="carbon-lang",
        help="The Carbon repo to query. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Prints long output, with the full comment.",
    )
    return parser.parse_args(args=args)


def _query(
    parsed_args: argparse.Namespace, field_name: Optional[str] = None
) -> str:
    """Returns a query for the passed field_name, or all by default."""
    print(".", end="", flush=True)
    format = {
        "pr_num": parsed_args.pr_num,
        "repo": parsed_args.repo,
        "comments": "",
        "review_threads": "",
        "reviews": "",
    }
    if field_name:
        # Use a cursor for pagination of the field.
        if field_name == "comments":
            format["comments"] = _QUERY_COMMENTS
        elif field_name == "reviewThreads":
            format["review_threads"] = _QUERY_REVIEW_THREADS
        elif field_name == "reviews":
            format["reviews"] = _QUERY_REVIEWS
        else:
            raise ValueError("Unexpected field_name: %s" % field_name)
    else:
        # Fetch the first page of all fields.
        subformat = {"cursor": "", "pagination": github_helpers.PAGINATION}
        format["comments"] = _QUERY_COMMENTS % subformat
        format["review_threads"] = _QUERY_REVIEW_THREADS % subformat
        format["reviews"] = _QUERY_REVIEWS % subformat
    return _QUERY % format


def _accumulate_pr_comment(
    parsed_args: argparse.Namespace,
    comments: List[_PRComment],
    raw_comment: Dict,
) -> None:
    """Collects top-level comments and reviews."""
    # Elide reviews that have no top-level comment body.
    if raw_comment["body"]:
        comments.append(_PRComment(raw_comment))


def _accumulate_thread(
    parsed_args: argparse.Namespace,
    threads_by_path: Dict[str, List[_Thread]],
    raw_thread: Dict,
) -> None:
    """Adds threads to threads_by_path for later sorting."""
    thread = _Thread(parsed_args, raw_thread)

    # Optionally skip resolved threads.
    if not parsed_args.include_resolved and thread.is_resolved:
        return

    # Optionally skip threads where the given user isn't the last commenter.
    if (
        parsed_args.comments_after
        and thread.comments[-1].author == parsed_args.comments_after
    ):
        return

    # Optionally skip threads where the given user hasn't commented.
    if parsed_args.comments_from and not thread.has_comment_from(
        parsed_args.comments_from
    ):
        return

    if thread.path not in threads_by_path:
        threads_by_path[thread.path] = []
    threads_by_path[thread.path].append(thread)


def _paginate(
    field_name: str,
    accumulator: Callable[[argparse.Namespace, Any, Dict], None],
    parsed_args: argparse.Namespace,
    client: github_helpers.Client,
    main_result: Dict,
    output: Any,
) -> None:
    """Paginates through the given field_name, accumulating results."""
    query = _query(parsed_args, field_name=field_name)
    path = ("repository", "pullRequest", field_name)
    for node in client.execute_and_paginate(
        query, path, first_page=main_result
    ):
        accumulator(parsed_args, output, node)


def _fetch_comments(
    parsed_args: argparse.Namespace,
) -> Tuple[List[_PRComment], Dict[str, List[_Thread]]]:
    """Fetches comments and review threads from GitHub."""
    # Each _query call will print a '.' for progress.
    print(
        "Loading https://github.com/carbon-language/%s/pull/%d ..."
        % (parsed_args.repo, parsed_args.pr_num),
        end="",
        flush=True,
    )

    client = github_helpers.Client(parsed_args)

    # Get the initial set of review threads, and print the PR summary.
    main_result = client.execute(_query(parsed_args))
    pull_request = main_result["repository"]["pullRequest"]

    # Paginate comments, reviews, and review threads.
    comments: List[_PRComment] = []
    _paginate(
        "comments",
        _accumulate_pr_comment,
        parsed_args,
        client,
        main_result,
        comments,
    )
    # Combine reviews into comments for interleaving.
    _paginate(
        "reviews",
        _accumulate_pr_comment,
        parsed_args,
        client,
        main_result,
        comments,
    )
    threads_by_path: Dict[str, List[_Thread]] = {}
    _paginate(
        "reviewThreads",
        _accumulate_thread,
        parsed_args,
        client,
        main_result,
        threads_by_path,
    )

    # Now that loading is done (no more progress indicators), print the header.
    print()
    pr_desc = _Comment(
        pull_request["author"]["login"],
        pull_request["createdAt"],
        pull_request["title"],
    )
    print(pr_desc.format(parsed_args.long))
    return comments, threads_by_path


def main() -> None:
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
