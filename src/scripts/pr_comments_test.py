#!/usr/bin/env python3

"""Tests for pr-comments.py."""

import os
import pr_comments
import unittest
import unittest.mock


class TestPRComments(unittest.TestCase):
    def test_format_comment_short(self):
        with unittest.mock.patch.dict("os.environ", {}):
            parsed_args = pr_comments._parse_args(args=["123"])
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(
                parsed_args
            ),
            "  author: brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                parsed_args
            ),
            "  author: brief¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author", created_at, "brief\n\n\nwrap"
            ).format(parsed_args),
            "  author: brief¶¶¶ wrap",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmo",
            ).format(parsed_args),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiusmo",
        )
        self.assertEqual(
            pr_comments._Comment(
                "author",
                created_at,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
                "do eiusmod",
            ).format(parsed_args),
            "  author: Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiu...",
        )

    def test_format_comment_long(self):
        with unittest.mock.patch.dict("os.environ", {}):
            parsed_args = pr_comments._parse_args(args=["--long", "123"])
        created_at = "2001-02-03T04:05:06Z"
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief").format(
                parsed_args
            ),
            "  author at 2001-02-03 04:05:\n    brief",
        )
        self.assertEqual(
            pr_comments._Comment("author", created_at, "brief\nwrap").format(
                parsed_args
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
            pr_comments._Comment("author", created_at, body).format(
                parsed_args
            ),
            "  author at 2001-02-03 04:05:\n"
            "    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
            "do eiusmod\n"
            "    tempor incididunt ut labore et dolore magna aliqua.\n"
            "    Ut enim ad minim veniam,",
        )


if __name__ == "__main__":
    unittest.main()
