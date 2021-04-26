"""Tests for gen_sidebar.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

from carbon.website.jekyll import gen_sidebar


class TestGenSidebar(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 9999

    def test_get_title(self):
        self.assertEqual(
            gen_sidebar._get_title(gen_sidebar._TOP_DIR, "README.md"),
            "Carbon language",
        )

    def _assert_and_get_subdir(self, dirs, rel_dir, title, readme_path):
        """Validates a subdir entry and returns it for further checks."""
        subdir = None
        for d in dirs:
            if d.rel_dir == rel_dir:
                subdir = d
                break
        self.assertIsNotNone(subdir, rel_dir)
        self.assertEqual(subdir.title, title)
        self.assertIn((gen_sidebar._README_TITLE, readme_path), subdir.files)
        return subdir

    def test_crawl(self):
        dirs = gen_sidebar.crawl(gen_sidebar._TOP_DIR)

        # Essential ordering.
        self.assertEqual(dirs[0].rel_dir, ".")
        self.assertEqual(dirs[-1].rel_dir, "proposals")

        # Check root structure and edge cases.
        root = self._assert_and_get_subdir(
            dirs, ".", "Carbon language", "/README.md"
        )
        self.assertIn(("Code of conduct", "/CODE_OF_CONDUCT.md"), root.files)
        self.assertIn((gen_sidebar._LICENSE_TITLE, "/LICENSE"), root.files)

        # Check nesting of design, which should elide docs.
        design = self._assert_and_get_subdir(
            dirs, "docs/design", "Language design", "/docs/design/README.md"
        )
        self._assert_and_get_subdir(
            design.subdirs,
            "docs/design/code_and_name_organization",
            "Code and name organization",
            "/docs/design/code_and_name_organization/README.md",
        )

        # Check a little proposal structure.
        proposals = self._assert_and_get_subdir(
            dirs, "proposals", "Proposals", "/proposals/README.md"
        )
        self.assertEqual(
            proposals.files[1],
            (
                "0024 - Generics goals",
                "/proposals/p0024.md",
            ),
        )
        self.assertEqual(
            proposals.files[2],
            (
                "0029 - Linear, rebase, and pull-request GitHub workflow",
                "/proposals/p0029.md",
            ),
        )

    def test_format(self):
        d = [
            gen_sidebar._Dir(
                "Test",
                "unused",
                [
                    (gen_sidebar._README_TITLE, "/README.md"),
                    (gen_sidebar._LICENSE_TITLE, "/LICENSE"),
                    ("A", "/A.md"),
                ],
            )
        ]
        d[0].subdirs.append(
            gen_sidebar._Dir(
                "Child",
                "unused",
                [(gen_sidebar._README_TITLE, "/child/README.md")],
            )
        )
        format_args = (
            gen_sidebar._HEADER,
            gen_sidebar._LINK_TEMPLATE
            % {
                "indent": 6 * " ",
                "url": "/",
                "title": gen_sidebar._README_TITLE,
            },
            gen_sidebar._LINK_TEMPLATE
            % {
                "indent": 6 * " ",
                "url": "/%s" % gen_sidebar._LICENSE_HTML,
                "title": gen_sidebar._LICENSE_TITLE,
            },
            gen_sidebar._LINK_TEMPLATE
            % {"indent": 6 * " ", "url": "/A.html", "title": "A"},
            gen_sidebar._LINK_TEMPLATE
            % {
                "indent": 10 * " ",
                "url": "/child/",
                "title": gen_sidebar._README_TITLE,
            },
        )
        self.assertEqual(
            gen_sidebar._format(d),
            "%s"
            "\n  <li>"
            '\n    <a href="#">Test</a>'
            "\n    <ul>"
            "%s"
            "%s"
            "%s"
            "\n      <li>"
            '\n        <a href="#">Child</a>'
            "\n        <ul>"
            "%s"
            "\n        </ul>"
            "\n      </li>"
            "\n    </ul>"
            "\n  </li>" % format_args,
        )


if __name__ == "__main__":
    unittest.main()
