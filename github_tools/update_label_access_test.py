"""Tests for update_label_access.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import unittest
from unittest import mock

import github  # type: ignore

from github_tools import github_helpers
from github_tools import update_label_access


class TestUpdateLabelAccess(unittest.TestCase):
    def setUp(self):
        # Stub out the access token.
        os.environ[github_helpers._ENV_TOKEN] = "unused"

        self.client = mock.create_autospec(github_helpers.Client, instance=True)
        self.gh = mock.create_autospec(github.Github, instance=True)
        self.gh_org = mock.create_autospec(github.Organization, instance=True)
        self.gh_team = mock.create_autospec(github.Team, instance=True)

        self.gh.get_organization = mock.MagicMock(return_value=self.gh_org)
        self.gh_org.get_team_by_slug = mock.MagicMock(return_value=self.gh_team)

    def _mock_nodes(self, logins):
        self.client.execute_and_paginate.return_value = [
            {"login": login} for login in logins
        ]

    def test_load_org_members_empty(self):
        self._mock_nodes([])
        self.assertRaises(
            AssertionError, update_label_access._load_org_members, self.client
        )

    def test_load_org_members_missing_ignored(self):
        self._mock_nodes(["foo", "bar"])
        self.assertRaises(
            AssertionError, update_label_access._load_org_members, self.client
        )

    def test_load_org_members_ignored_only(self):
        self._mock_nodes(update_label_access._IGNORE_ACCOUNTS)
        self.assertEqual(
            update_label_access._load_org_members(self.client), set()
        )

    def test_load_org_members_found(self):
        self._mock_nodes(
            ["foo", "bar"] + list(update_label_access._IGNORE_ACCOUNTS)
        )
        self.assertEqual(
            update_label_access._load_org_members(self.client),
            set(["foo", "bar"]),
        )

    def test_load_team_members_empty(self):
        self._mock_nodes([])
        self.assertEqual(
            update_label_access._load_team_members(self.client), set()
        )

    def test_load_team_members_found(self):
        self._mock_nodes(["foo", "bar"])
        self.assertEqual(
            update_label_access._load_team_members(self.client),
            set(["foo", "bar"]),
        )

    def test_update_team_empty(self):
        update_label_access._update_team(self.gh, set(), set())

    def test_update_team_equal(self):
        update_label_access._update_team(
            self.gh, set(["foo", "bar"]), set(["foo", "bar"])
        )

    def test_update_team_add(self):
        self.gh.get_user = mock.MagicMock(return_value="bar-user")
        self.gh_team.add_membership = mock.MagicMock()
        update_label_access._update_team(
            self.gh, set(["foo", "bar"]), set(["foo"])
        )
        self.gh.get_user.assert_called_once_with("bar")
        self.gh_team.add_membership.assert_called_once_with("bar-user")

    def test_update_team_remove(self):
        self.gh.get_user = mock.MagicMock(return_value="bar-user")
        self.gh_team.remove_membership = mock.MagicMock()
        update_label_access._update_team(
            self.gh, set(["foo"]), set(["foo", "bar"])
        )
        self.gh.get_user.assert_called_once_with("bar")
        self.gh_team.remove_membership.assert_called_once_with("bar-user")


if __name__ == "__main__":
    unittest.main()
