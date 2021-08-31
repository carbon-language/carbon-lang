"""Tests for pr_comments.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import unittest
from unittest import mock

from github_tools import github_helpers
from github_tools import pr_comments


class TestPRComments(unittest.TestCase):
    def setUp(self):
        # Stub out the access token.
        os.environ[github_helpers._ENV_TOKEN] = "unused"

    def test_format_comment_short(self):
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(False),
            "  author: brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                False
            ),
            "  author: brief¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author", created_at, "brief\n\n\nwrap"
            ).format(False),
            "  author: brief¶¶¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmo",
            ).format(False),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiusmo",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmod",
            ).format(False),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiu...",
        )

    def test_format_comment_long(self):
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(True),
            "  author at 2001-02-03 04:05:\n    brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                True
            ),
            "  author at 2001-02-03 04:05:\n    brief\n    wrap",
        )

        body = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
            "do eiusmod tempor incididunt ut labore et dolore magna "
            "aliqua.\n"
            "Ut enim ad minim veniam,"
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, body).format(True),
            "  author at 2001-02-03 04:05:\n"
            "    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
            "do eiusmod\n"
            "    tempor incididunt ut labore et dolore magna aliqua.\n"
            "    Ut enim ad minim veniam,",
        )

    @staticmethod
    def fake_thread(**kwargs):
        with mock.patch.dict(os.environ, {}):
            parsed_args = pr_comments._parse_args(["83"])
        return pr_comments._Thread(
            parsed_args, TestPRComments.fake_thread_dict(**kwargs)
        )

    @staticmethod
    def fake_thread_dict(
        is_resolved=False,
        path="foo.md",
        line=3,
        created_at="2001-02-03T04:05:06Z",
    ):
        thread_dict = {
            "isResolved": is_resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "author"},
                        "body": "comment",
                        "createdAt": created_at,
                        "originalCommit": {"abbreviatedOid": "abcdef"},
                        "originalPosition": line,
                        "path": path,
                        "url": "http://xyz",
                    },
                    {
                        "author": {"login": "other"},
                        "body": "reply",
                        "createdAt": "2001-02-03T04:15:16Z",
                    },
                ],
            },
        }
        if is_resolved:
            thread_dict["resolvedBy"] = {
                "login": "resolver",
                "createdAt": "2001-02-03T04:25:26Z",
            }
        return thread_dict

    def test_thread_format(self):
        self.assertEqual(
            self.fake_thread().format(False),
            "https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef#diff-d8ca3b3d314d8209367af0eea2373b6fR3\n"
            "  - line 3; unresolved\n"
            "  - diff: https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef..HEAD#diff-d8ca3b3d314d8209367af0eea2373b6fL3\n"
            "  author: comment\n"
            "  other: reply",
        )
        self.assertEqual(
            self.fake_thread().format(True),
            "https://github.com/carbon-language/carbon-lang/pull/83/files/"
            "abcdef#diff-d8ca3b3d314d8209367af0eea2373b6fR3\n"
            "  - line 3; unresolved\n"
            "  - diff: https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef..HEAD#diff-d8ca3b3d314d8209367af0eea2373b6fL3\n"
            "  author at 2001-02-03 04:05:\n"
            "    comment\n"
            "  other at 2001-02-03 04:15:\n"
            "    reply",
        )

        self.assertEqual(
            self.fake_thread(is_resolved=True).format(False),
            "https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef#diff-d8ca3b3d314d8209367af0eea2373b6fR3\n"
            "  - line 3; resolved\n"
            "  - diff: https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef..HEAD#diff-d8ca3b3d314d8209367af0eea2373b6fL3\n"
            "  author: comment\n"
            "  other: reply\n"
            "  resolver: <resolved>",
        )
        self.assertEqual(
            self.fake_thread(is_resolved=True).format(True),
            "https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef#diff-d8ca3b3d314d8209367af0eea2373b6fR3\n"
            "  - line 3; resolved\n"
            "  - diff: https://github.com/carbon-language/carbon-lang/pull/83/"
            "files/abcdef..HEAD#diff-d8ca3b3d314d8209367af0eea2373b6fL3\n"
            "  author at 2001-02-03 04:05:\n"
            "    comment\n"
            "  other at 2001-02-03 04:15:\n"
            "    reply\n"
            "  resolver at 2001-02-03 04:25:\n"
            "    <resolved>",
        )

    def test_thread_lt(self):
        thread1 = self.fake_thread(line=2)
        thread2 = self.fake_thread()
        thread3 = self.fake_thread(created_at="2002-02-03T04:05:06Z")

        self.assertTrue(thread1 < thread2)
        self.assertFalse(thread2 < thread1)

        self.assertFalse(thread2 < thread2)

        self.assertTrue(thread2 < thread3)
        self.assertFalse(thread3 < thread2)

    def test_accumulate_thread(self):
        with mock.patch.dict(os.environ, {}):
            parsed_args = pr_comments._parse_args(["83"])
        threads_by_path = {}
        review_threads = [
            self.fake_thread_dict(line=2),
            self.fake_thread_dict(line=4),
            self.fake_thread_dict(path="other.md"),
            self.fake_thread_dict(),
        ]
        for thread in review_threads:
            pr_comments._accumulate_thread(
                parsed_args,
                threads_by_path,
                thread,
            )
        self.assertEqual(sorted(threads_by_path.keys()), ["foo.md", "other.md"])
        threads = sorted(threads_by_path["foo.md"])
        self.assertEqual(len(threads), 3)
        self.assertEqual(threads[0].line, 2)
        self.assertEqual(threads[1].line, 3)
        self.assertEqual(threads[2].line, 4)
        self.assertEqual(len(threads_by_path["other.md"]), 1)

    @staticmethod
    def fake_pr_comment(**kwargs):
        return pr_comments._PRComment(
            TestPRComments.fake_pr_comment_dict(**kwargs)
        )

    @staticmethod
    def fake_pr_comment_dict(
        body="comment",
        created_at="2001-02-03T04:05:06Z",
    ):
        pr_comment_dict = {
            "author": {"login": "author"},
            "body": body,
            "createdAt": created_at,
            "url": "http://xyz",
        }
        return pr_comment_dict

    def test_pr_comment_format(self):
        self.assertEqual(
            self.fake_pr_comment().format(False),
            "http://xyz\n  author: comment",
        )
        self.assertEqual(
            self.fake_pr_comment().format(True),
            "http://xyz\n  author at 2001-02-03 04:05:\n    comment",
        )

    def test_pr_comment_lt(self):
        pr_comment1 = self.fake_pr_comment()
        pr_comment2 = self.fake_pr_comment(created_at="2002-02-03T04:05:06Z")

        self.assertTrue(pr_comment1 < pr_comment2)
        self.assertFalse(pr_comment2 < pr_comment1)

        self.assertFalse(pr_comment2 < pr_comment2)

    def test_accumulate_pr_comment(self):
        with mock.patch.dict(os.environ, {}):
            parsed_args = pr_comments._parse_args(["83"])
        raw_comments = [
            self.fake_pr_comment_dict(body="x"),
            self.fake_pr_comment_dict(body=""),
            self.fake_pr_comment_dict(
                body="y", created_at="2000-02-03T04:05:06Z"
            ),
            self.fake_pr_comment_dict(
                body="z", created_at="2002-02-03T04:05:06Z"
            ),
        ]
        comments = []
        for raw_comment in raw_comments:
            pr_comments._accumulate_pr_comment(
                parsed_args, comments, raw_comment
            )
        comments.sort()
        self.assertEqual(len(comments), 3)
        self.assertEqual(comments[0].body, "y")
        self.assertEqual(comments[1].body, "x")
        self.assertEqual(comments[2].body, "z")


if __name__ == "__main__":
    unittest.main()
