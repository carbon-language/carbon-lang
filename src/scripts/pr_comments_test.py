#!/usr/bin/env python3

"""Tests for pr-comments.py."""

import os
import pr_comments
import unittest
from unittest import mock


class TestPRComments(unittest.TestCase):
    def test_format_comment_short(self):
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(
                False, 0
            ),
            "author: brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(
                False, 2
            ),
            "  author: brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                False, 2
            ),
            "  author: brief¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author", created_at, "brief\n\n\nwrap"
            ).format(False, 2),
            "  author: brief¶¶¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmo",
            ).format(False, 2),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiusmo",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmod",
            ).format(False, 2),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiu...",
        )

    def test_format_comment_long(self):
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(True, 0),
            "author at 2001-02-03 04:05:\n  brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(True, 2),
            "  author at 2001-02-03 04:05:\n    brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                True, 2
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
            pr_comments._Comment("author", created_at, body).format(True, 2),
            "  author at 2001-02-03 04:05:\n"
            "    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
            "do eiusmod\n"
            "    tempor incididunt ut labore et dolore magna aliqua.\n"
            "    Ut enim ad minim veniam,",
        )

    @staticmethod
    def fake_thread(**kwargs):
        return pr_comments._Thread(TestPRComments.fake_thread_dict(**kwargs))

    @staticmethod
    def fake_thread_dict(
        is_resolved=False,
        path="foo.md",
        line=3,
        created_at="2001-02-03T04:05:06Z",
    ):
        return {
            "isResolved": is_resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "author"},
                        "body": "comment",
                        "createdAt": created_at,
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
            "resolvedBy": {
                "login": "resolver",
                "createdAt": "2001-02-03T04:25:26Z",
            },
        }

    def test_thread_format(self):
        self.assertEqual(
            self.fake_thread().format(False),
            "line 3; unresolved\n"
            "    http://xyz\n"
            "  author: comment\n"
            "  other: reply",
        )
        self.assertEqual(
            self.fake_thread().format(True),
            "line 3; unresolved\n"
            "    http://xyz\n"
            "  author at 2001-02-03 04:05:\n"
            "    comment\n"
            "  other at 2001-02-03 04:15:\n"
            "    reply",
        )

        self.assertEqual(
            self.fake_thread(is_resolved=True).format(False),
            "line 3; resolved\n"
            "    http://xyz\n"
            "  author: comment\n"
            "  other: reply\n"
            "  resolver: <resolved>",
        )
        self.assertEqual(
            self.fake_thread(is_resolved=True).format(True),
            "line 3; resolved\n"
            "    http://xyz\n"
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

    def test_accumulate_threads(self):
        with mock.patch.dict(os.environ, {}):
            parsed_args = pr_comments._parse_args(["83"])
        threads_by_path = {}
        review_threads = [
            self.fake_thread_dict(line=2),
            self.fake_thread_dict(line=4),
            self.fake_thread_dict(path="other.md"),
            self.fake_thread_dict(),
        ]
        pr_comments._accumulate_threads(
            parsed_args, threads_by_path, review_threads
        )
        self.assertEqual(sorted(threads_by_path.keys()), ["foo.md", "other.md"])
        threads = sorted(threads_by_path["foo.md"])
        self.assertEqual(len(threads), 3)
        self.assertEqual(threads[0].line, 2)
        self.assertEqual(threads[1].line, 3)
        self.assertEqual(threads[2].line, 4)
        self.assertEqual(len(threads_by_path["other.md"]), 1)


if __name__ == "__main__":
    unittest.main()
