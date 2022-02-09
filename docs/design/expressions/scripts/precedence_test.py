"""Tests for precedence.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import io
import unittest

from carbon.docs.design.expressions.scripts import precedence


class PrecedenceTest(unittest.TestCase):
    def test_golden(self):
        with open("docs/design/expressions/scripts/precedence.dot") as f:
            expected = f.read()

        with io.StringIO() as buffer:
            precedence.Graph(buffer).write()
            actual = buffer.getvalue()

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
