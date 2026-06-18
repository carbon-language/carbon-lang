#!/usr/bin/env python3

"""Tests for new_proposal.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import unittest

from proposals.scripts import new_proposal


class TestNewProposal(unittest.TestCase):
    def test_calculate_branch_short(self):
        parsed_args = new_proposal._parse_args(["foo bar"])
        self.assertEqual(
            new_proposal._calculate_branch(parsed_args), "proposal-foo-bar"
        )

    def test_calculate_branch_long(self):
        parsed_args = new_proposal._parse_args(
            ["A really long long long title"]
        )
        self.assertEqual(
            new_proposal._calculate_branch(parsed_args),
            "proposal-a-really-long-long-long-title",
        )

    def test_calculate_branch_special(self):
        parsed_args = new_proposal._parse_args(
            ["A title with #*%$ special `char`s"]
        )
        self.assertEqual(
            new_proposal._calculate_branch(parsed_args),
            "proposal-a-title-with-special-char-s",
        )

    def test_calculate_branch_flag(self):
        parsed_args = new_proposal._parse_args(["--branch=wiz", "foo"])
        self.assertEqual(new_proposal._calculate_branch(parsed_args), "wiz")

    def test_fill_template(self):
        # Switch directories for the test.
        parsed_args = new_proposal._parse_args(["foo"])
        os.chdir(parsed_args.repo_root)

        content = new_proposal._fill_template(
            "proposals/scripts/template.md",
            "TITLE",
            123,
        )
        self.assertTrue(content.startswith("# TITLE\n\n"), content)
        self.assertTrue(
            "[Pull request](https://github.com/carbon-language/carbon-lang/"
            "pull/123)" in content,
            content,
        )
        self.assertTrue(
            "<!-- tocstop -->\n\n## Abstract\n\n" in content, content
        )

    def test_run_success(self):
        new_proposal._run(["true"])

    def test_run_failure(self):
        self.assertRaises(SystemExit, new_proposal._run, ["false"])


if __name__ == "__main__":
    unittest.main()
