"""Tests for github_helpers.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest
from unittest import mock

from github_tools import github_helpers


_TEST_QUERY = """
query {
  top(login: "foo") {
    child(first: 100%(cursor)s) {
      nodes {
        login
      }
      %(pagination)s
    }
  }
}
"""

_TEST_QUERY_PATH = ("top", "child")

_EXP_QUERY_FIRST_PAGE = """
query {
  top(login: "foo") {
    child(first: 100) {
      nodes {
        login
      }
      pageInfo {
  hasNextPage
  endCursor
}
totalCount
    }
  }
}
"""

_EXP_QUERY_SECOND_PAGE = """
query {
  top(login: "foo") {
    child(first: 100 after: "CURSOR") {
      nodes {
        login
      }
      pageInfo {
  hasNextPage
  endCursor
}
totalCount
    }
  }
}
"""


class TestGithubHelpers(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            github_helpers.Client, "__init__", lambda self, parsed_args: None
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.client = github_helpers.Client(None)

    @staticmethod
    def mock_result(nodes, total_count=None, has_next_page=False):
        if total_count is None:
            total_count = len(nodes)
        end_cursor = None
        if has_next_page:
            end_cursor = "CURSOR"
        return {
            "top": {
                "child": {
                    "nodes": nodes,
                    "pageInfo": {
                        "hasNextPage": has_next_page,
                        "endCursor": end_cursor,
                    },
                    "totalCount": total_count,
                }
            }
        }

    def test_execute_and_paginate_empty(self):
        self.client.execute = mock.MagicMock(return_value=self.mock_result([]))
        self.assertEqual(
            list(
                self.client.execute_and_paginate(_TEST_QUERY, _TEST_QUERY_PATH)
            ),
            [],
        )
        self.client.execute.assert_called_once_with(_EXP_QUERY_FIRST_PAGE)

    def test_execute_and_paginate_one_page(self):
        self.client.execute = mock.MagicMock(
            return_value=self.mock_result(["foo", "bar", "baz"])
        )
        self.assertEqual(
            list(
                self.client.execute_and_paginate(_TEST_QUERY, _TEST_QUERY_PATH)
            ),
            ["foo", "bar", "baz"],
        )
        self.client.execute.assert_called_once_with(_EXP_QUERY_FIRST_PAGE)

    def test_execute_and_paginate_one_page_count_mismatch(self):
        self.client.execute = mock.MagicMock(
            return_value=self.mock_result(["foo"], total_count=2),
        )
        self.assertRaises(
            AssertionError,
            list,
            self.client.execute_and_paginate(_TEST_QUERY, _TEST_QUERY_PATH),
        )
        self.client.execute.assert_called_once_with(_EXP_QUERY_FIRST_PAGE)

    def test_execute_and_paginate_two_page(self):
        def paging(query):
            if query == _EXP_QUERY_FIRST_PAGE:
                return self.mock_result(
                    ["foo", "bar"], total_count=3, has_next_page=True
                )
            elif query == _EXP_QUERY_SECOND_PAGE:
                return self.mock_result(["baz"], total_count="unused")
            else:
                raise ValueError("Bad query: %s" % query)

        self.client.execute = mock.MagicMock(side_effect=paging)
        self.assertEqual(
            list(
                self.client.execute_and_paginate(_TEST_QUERY, _TEST_QUERY_PATH)
            ),
            ["foo", "bar", "baz"],
        )
        self.assertEqual(self.client.execute.call_count, 2)

    def test_execute_and_paginate_first_page_done(self):
        self.client.execute = mock.MagicMock()
        self.assertEqual(
            list(
                self.client.execute_and_paginate(
                    _TEST_QUERY,
                    _TEST_QUERY_PATH,
                    first_page=self.mock_result(["foo"]),
                )
            ),
            ["foo"],
        )
        self.assertEqual(self.client.execute.call_count, 0)

    def test_execute_and_paginate_first_page_continue(self):
        self.client.execute = mock.MagicMock(
            return_value=self.mock_result(["bar"], total_count="unused")
        )
        self.assertEqual(
            list(
                self.client.execute_and_paginate(
                    _TEST_QUERY,
                    _TEST_QUERY_PATH,
                    first_page=self.mock_result(
                        ["foo"], total_count=2, has_next_page=True
                    ),
                )
            ),
            ["foo", "bar"],
        )
        self.client.execute.assert_called_once_with(_EXP_QUERY_SECOND_PAGE)


if __name__ == "__main__":
    unittest.main()
