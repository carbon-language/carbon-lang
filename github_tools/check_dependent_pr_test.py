"""Tests for check_dependent_pr.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest
from unittest import mock

import check_dependent_pr
import github_helpers

_OID1 = "1" * 40
_OID2 = "2" * 40


class TestCheckDependentPR(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = mock.MagicMock(spec=github_helpers.Client)

    def test_process_pr_no_overlap(self) -> None:
        # 1 commit, no overlap, no existing comment
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_1",
                    "labels": {"nodes": []},
                    "commits": {"nodes": [{"commit": {"oid": _OID1}}]},
                    "comments": {"nodes": []},
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=1,
            commit_to_prs={_OID1: {1}},
            open_pr_numbers={1},
            label_id="label_id",
            dry_run=False,
        )
        # Should not call execute again (no mutations)
        self.assertEqual(self.mock_client.execute.call_count, 1)

    def test_process_pr_with_overlap(self) -> None:
        # 2 commits, overlap with PR 7087
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_2",
                    "labels": {"nodes": []},
                    "commits": {
                        "nodes": [
                            {"commit": {"oid": _OID1}},
                            {"commit": {"oid": _OID2}},
                        ]
                    },
                    "comments": {"nodes": []},
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            commit_to_prs={_OID1: {1, 2}, _OID2: {2}},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            dry_run=False,
        )
        # Should call execute 3 times: 1 for details, 1 for label, 1 for comment
        self.assertEqual(self.mock_client.execute.call_count, 3)

        calls = self.mock_client.execute.call_args_list
        self.assertIn("addLabelsToLabelable", calls[1][0][0])
        self.assertIn("addComment", calls[2][0][0])

    def test_process_pr_dependencies_merged(self) -> None:
        # Existing comment says it depends on 7087, but 7087 is closed
        # (not in open_pr_numbers)
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_3",
                    "labels": {
                        "nodes": [
                            {"name": "dependent", "id": "label_dependent"}
                        ]
                    },
                    "commits": {
                        "nodes": [
                            {"commit": {"oid": _OID1}},
                            {"commit": {"oid": _OID2}},
                        ]
                    },
                    "comments": {
                        "nodes": [
                            {
                                "id": "comment_id",
                                "body": (
                                    '<!-- check_dependent_pr {"open": [7087], '
                                    '"merged": []} -->\nDepends on #7087'
                                ),
                            }
                        ]
                    },
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=3,
            commit_to_prs={_OID1: {3}, _OID2: {3}},
            open_pr_numbers={3},
            label_id="label_dependent",
            dry_run=False,
        )

        calls = self.mock_client.execute.call_args_list
        # Should remove label
        self.assertIn("removeLabelsFromLabelable", calls[1][0][0])
        # Should update comment
        self.assertIn("updateIssueComment", calls[2][0][0])

    def test_process_pr_scanning_no_add(self) -> None:
        # Overlap found, but scanning=True, so no comment or label should be
        # added.
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_7",
                    "labels": {"nodes": []},
                    "commits": {
                        "nodes": [
                            {"commit": {"oid": _OID1}},
                            {"commit": {"oid": _OID2}},
                        ]
                    },
                    "comments": {"nodes": []},
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=7,
            commit_to_prs={_OID1: {1, 7}, _OID2: {7}},
            open_pr_numbers={1, 7},
            label_id="label_dependent",
            dry_run=False,
            scanning=True,
        )
        # Should not call execute again (no mutations)
        self.assertEqual(self.mock_client.execute.call_count, 1)

    def test_process_pr_no_changes_needed(self) -> None:
        # Overlap with PR 1, existing comment already lists it, label
        # present.
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_6",
                    "labels": {
                        "nodes": [
                            {"name": "dependent", "id": "label_dependent"}
                        ]
                    },
                    "commits": {
                        "nodes": [
                            {"commit": {"oid": _OID1}},
                            {"commit": {"oid": _OID2}},
                        ]
                    },
                    "comments": {
                        "nodes": [
                            {
                                "id": "comment_id",
                                "body": (
                                    '<!-- check_dependent_pr {"open": [1], '
                                    '"merged": []} -->\nDepends on #1'
                                ),
                            }
                        ]
                    },
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=6,
            commit_to_prs={_OID1: {1, 6}, _OID2: {6}},
            open_pr_numbers={1, 6},
            label_id="label_dependent",
            dry_run=False,
        )
        # Should not call execute again (no mutations)
        self.assertEqual(self.mock_client.execute.call_count, 1)

    def test_process_pr_invalid_marker(self) -> None:
        self.mock_client.execute.return_value = {
            "repository": {
                "pullRequest": {
                    "id": "pr_5",
                    "labels": {"nodes": []},
                    "commits": {"nodes": [{"commit": {"oid": _OID1}}]},
                    "comments": {
                        "nodes": [
                            {
                                "id": "comment_id",
                                "body": (
                                    "<!-- check_dependent_pr {invalid_json} "
                                    "-->"
                                ),
                            }
                        ]
                    },
                }
            }
        }
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=5,
            commit_to_prs={_OID1: {5}},
            open_pr_numbers={5},
            label_id="label_dependent",
            dry_run=False,
        )
        # Should not call execute again (no mutations)
        self.assertEqual(self.mock_client.execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
