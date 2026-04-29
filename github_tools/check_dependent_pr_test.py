"""Tests for check_dependent_pr.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import unittest
from unittest import mock
from typing import Any

import check_dependent_pr
import github_helpers

_OID1 = "1" * 40
_OID2 = "2" * 40
_OID3 = "3" * 40
_OID4 = "4" * 40
_OID9 = "9" * 40


class TestCheckDependentPR(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = mock.MagicMock(spec=github_helpers.Client)
        # Mock requests.post to avoid network calls and track status updates.
        self.requests_post_patcher = mock.patch("requests.post")
        self.mock_post = self.requests_post_patcher.start()

    def tearDown(self) -> None:
        self.requests_post_patcher.stop()

    def _assert_status(self, sha: str, state: str, description: str) -> None:
        """Validates that requests.post was called to set the commit status."""
        self.mock_post.assert_called_once()
        args, kwargs = self.mock_post.call_args
        self.assertIn(f"statuses/{sha}", args[0])
        self.assertEqual(kwargs["json"]["state"], state)
        self.assertEqual(kwargs["json"]["context"], "PR dependencies check")
        self.assertEqual(kwargs["json"]["description"], description)

    def _make_comment(
        self,
        open_deps: list[int],
        merged_deps: list[int] = None,
        first_commit: str = None,
        comment_id: str = "comment_id",
    ) -> dict[str, str]:
        """Builds a boilerplate PR comment."""
        state: dict[str, Any] = {
            "open": open_deps,
            "merged": merged_deps if merged_deps else [],
        }
        if first_commit:
            state["first_commit"] = first_commit
        return {
            "id": comment_id,
            "body": f"<!-- check_dependent_pr {json.dumps(state)} -->",
        }

    def _make_pr_response(
        self,
        pr_id: str,
        head_ref_oid: str,
        commits: list[str],
        comments: list[dict[str, str]] = None,
        has_dependent_label: bool = False,
    ) -> dict[str, Any]:
        """Builds a boilerplate GitHub response for a PR."""
        labels = (
            [{"name": "dependent", "id": "label_dependent"}]
            if has_dependent_label
            else []
        )
        return {
            "repository": {
                "pullRequest": {
                    "id": pr_id,
                    "headRefOid": head_ref_oid,
                    "labels": {"nodes": labels},
                    "commits": {
                        "nodes": [{"commit": {"oid": oid}} for oid in commits]
                    },
                    "comments": {"nodes": comments if comments else []},
                }
            }
        }

    def test_process_pr_no_overlap(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_1",
            head_ref_oid=_OID1,
            commits=[_OID1],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=1,
            pr_to_commits={1: [_OID1]},
            open_pr_numbers={1},
            label_id="label_id",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID1, "success", "This PR has no open dependencies"
        )

    def test_process_pr_with_overlap(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_2",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            pr_to_commits={1: [_OID1], 2: [_OID1, _OID2]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 3)
        calls = self.mock_client.execute.call_args_list
        self.assertIn("addLabelsToLabelable", calls[1][0][0])
        self.assertIn("addComment", calls[2][0][0])
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_dependencies_merged(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_3",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[self._make_comment(open_deps=[1])],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=3,
            pr_to_commits={3: [_OID1, _OID2]},
            open_pr_numbers={3},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        self.assertIn("removeLabelsFromLabelable", calls[1][0][0])
        self.assertIn("updateIssueComment", calls[2][0][0])
        self._assert_status(
            _OID2, "success", "This PR has no open dependencies"
        )

    def test_process_pr_dependency_got_new_commits(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_3",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[self._make_comment(open_deps=[1, 2])],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=3,
            pr_to_commits={1: [_OID1, _OID4], 3: [_OID1, _OID2]},
            open_pr_numbers={1, 3},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        update_mutation = calls[1][0][0]
        self.assertIn("updateIssueComment", update_mutation)
        variable_values = calls[1][1]["variable_values"]
        self.assertIn('"open": [1]', variable_values["body"])
        self.assertIn('"merged": [2]', variable_values["body"])
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_non_coherent_prefix(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_10",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=10,
            pr_to_commits={10: [_OID1, _OID2], 11: [_OID1, _OID3]},
            open_pr_numbers={10, 11},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID2, "success", "This PR has no open dependencies"
        )

    def test_process_pr_overlap_only_on_head_ref(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_9",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=9,
            pr_to_commits={1: [_OID2], 9: [_OID1, _OID2]},
            open_pr_numbers={1, 9},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 3)
        calls = self.mock_client.execute.call_args_list
        self.assertIn("addLabelsToLabelable", calls[1][0][0])
        self.assertIn("addComment", calls[2][0][0])
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_scanning_no_add(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_7",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=7,
            pr_to_commits={1: [_OID1], 7: [_OID1, _OID2]},
            open_pr_numbers={1, 7},
            label_id="label_dependent",
            token="test_token",
            scanning=True,
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_no_changes_needed(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_6",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[self._make_comment(open_deps=[1], first_commit=_OID2)],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=6,
            pr_to_commits={1: [_OID1], 6: [_OID1, _OID2]},
            open_pr_numbers={1, 6},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_invalid_marker(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_5",
            head_ref_oid=_OID1,
            commits=[_OID1],
            comments=[
                {
                    "id": "comment_id",
                    "body": "<!-- check_dependent_pr {invalid_json} -->",
                }
            ],
        )
        import json

        self.assertRaises(
            json.decoder.JSONDecodeError,
            check_dependent_pr._process_pr,
            self.mock_client,
            pr_number=5,
            pr_to_commits={5: [_OID1]},
            open_pr_numbers={5},
            label_id="label_dependent",
            token="test_token",
        )

    def test_process_pr_hidden_comment(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_14",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[
                {
                    "id": "hidden_comment_id",
                    "body": '<!-- check_dependent_pr {"open": [1]} -->',
                    "isMinimized": True,
                }
            ],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=14,
            pr_to_commits={1: [_OID1], 14: [_OID1, _OID2]},
            open_pr_numbers={1, 14},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        self.assertEqual(self.mock_client.execute.call_count, 2)
        self.assertIn("addComment", calls[1][0][0])
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_sticky_first_commit(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_11",
            head_ref_oid=_OID3,
            commits=[_OID1, _OID2, _OID3],
            comments=[self._make_comment(open_deps=[1, 2], first_commit=_OID2)],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=11,
            pr_to_commits={1: [_OID1], 11: [_OID1, _OID2, _OID3]},
            open_pr_numbers={1, 11},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        variable_values = calls[1][1]["variable_values"]
        self.assertIn(_OID2[:8], variable_values["body"])
        self.assertNotIn(_OID1[:8], variable_values["body"])
        self._assert_status(
            _OID3, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_rebase_first_commit(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_12",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[self._make_comment(open_deps=[1, 2])],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=12,
            pr_to_commits={1: [_OID9], 12: [_OID1, _OID2]},
            open_pr_numbers={1, 12},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        variable_values = calls[1][1]["variable_values"]
        self.assertIn(_OID1[:8], variable_values["body"])
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_fallback_no_independent_commit(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_13",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
            comments=[self._make_comment(open_deps=[1, 2])],
            has_dependent_label=True,
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=13,
            pr_to_commits={1: [_OID1, _OID2], 13: [_OID1, _OID2]},
            open_pr_numbers={1, 13},
            label_id="label_dependent",
            token="test_token",
        )
        calls = self.mock_client.execute.call_args_list
        variable_values = calls[1][1]["variable_values"]
        self.assertIn(
            "unable to identify starting review commit", variable_values["body"]
        )
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_process_pr_sequence_failure(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_1",
            head_ref_oid=_OID1,
            commits=[_OID1],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=1,
            pr_to_commits={1: [_OID1], 2: [_OID1, _OID2]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID1, "success", "This PR has no open dependencies"
        )

    def test_process_pr_no_overlap_different_commits(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_2",
            head_ref_oid=_OID2,
            commits=[_OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            pr_to_commits={1: [_OID1], 2: [_OID2]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID2, "success", "This PR has no open dependencies"
        )

    def test_process_pr_no_unique_commit(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_2",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            pr_to_commits={1: [_OID1, _OID2, _OID3], 2: [_OID1, _OID2]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 1)
        self._assert_status(
            _OID2, "success", "This PR has no open dependencies"
        )

    def test_process_pr_multiple_non_overlapping_commits(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_2",
            head_ref_oid=_OID4,
            commits=[_OID1, _OID3, _OID4],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            pr_to_commits={1: [_OID1, _OID2], 2: [_OID1, _OID3, _OID4]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self.assertEqual(self.mock_client.execute.call_count, 3)
        calls = self.mock_client.execute.call_args_list
        self.assertIn("addLabelsToLabelable", calls[1][0][0])
        self._assert_status(
            _OID4, "pending", "This PR has open dependencies: #1"
        )

    def test_always_sets_status_check_success(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_1",
            head_ref_oid=_OID1,
            commits=[_OID1],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=1,
            pr_to_commits={1: [_OID1]},
            open_pr_numbers={1},
            label_id="label_id",
            token="test_token",
        )
        self._assert_status(
            _OID1, "success", "This PR has no open dependencies"
        )

    def test_always_sets_status_check_pending(self) -> None:
        self.mock_client.execute.return_value = self._make_pr_response(
            pr_id="pr_2",
            head_ref_oid=_OID2,
            commits=[_OID1, _OID2],
        )
        check_dependent_pr._process_pr(
            self.mock_client,
            pr_number=2,
            pr_to_commits={1: [_OID1], 2: [_OID1, _OID2]},
            open_pr_numbers={1, 2},
            label_id="label_dependent",
            token="test_token",
        )
        self._assert_status(
            _OID2, "pending", "This PR has open dependencies: #1"
        )

    def test_query_max_merged_pr_explicit_orderBy_and_first_one(self) -> None:
        self.assertIn(
            "orderBy: {field: CREATED_AT, direction: DESC}",
            check_dependent_pr._QUERY_MAX_MERGED_PR,
        )
        self.assertIn(
            "first: 1",
            check_dependent_pr._QUERY_MAX_MERGED_PR,
        )


if __name__ == "__main__":
    unittest.main()
