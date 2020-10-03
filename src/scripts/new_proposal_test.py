#!/usr/bin/env python3

"""Tests for new_proposal.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest
from unittest import mock

import new_proposal


class FakeExitError(Exception):
    pass


def _fake_exit(message):
    raise FakeExitError(message)


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
            "proposal-a-really-long-long-l",
        )

    def test_calculate_branch_flag(self):
        parsed_args = new_proposal._parse_args(["--branch=wiz", "foo"])
        self.assertEqual(new_proposal._calculate_branch(parsed_args), "wiz")

    def test_fill_template(self):
        content = new_proposal._fill_template(
            "../../proposals/template.md", "TITLE", 123
        )
        self.assertTrue(content.startswith("# TITLE\n\n"), content)
        self.assertTrue(
            "[Pull request](https://github.com/carbon-language/carbon-lang/"
            "pull/123)" in content,
            content,
        )

    def test_run_success(self):
        new_proposal._run(["true"])

    def test_run_failure(self):
        with mock.patch(
            "new_proposal._exit", side_effect=_fake_exit
        ) as mock_exit:
            self.assertRaises(FakeExitError, new_proposal._run, ["false"])


if __name__ == "__main__":
    unittest.main()
