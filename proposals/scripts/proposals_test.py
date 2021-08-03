"""Tests for proposals.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

from carbon.proposals.scripts import proposals


class TestProposal(unittest.TestCase):
    def test_get_path(self):
        proposals_path = proposals.get_path()
        p = proposals.get_list(proposals_path)
        self.assertEqual(
            p[0],
            (
                "0024 - Generics goals",
                "p0024.md",
            ),
        )
        self.assertEqual(
            p[1],
            (
                "0029 - Linear, rebase, and pull-request GitHub workflow",
                "p0029.md",
            ),
        )


if __name__ == "__main__":
    unittest.main()
