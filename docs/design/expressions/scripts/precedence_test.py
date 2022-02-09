"""Tests for precedence.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

from carbon.docs.design.expressions.scripts import precedence


class TestGithubHelpers(unittest.TestCase):
    def test_golden_dot(self):
        with open("docs/design/expressions/scripts/precedence.dot") as f:
            expected = f.read()
        self.assertEqual(expected, precedence.Graph().source())
