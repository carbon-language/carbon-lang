#!/usr/bin/env python3

"""Tests for utils.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

from proposals.scripts import utils


class TestUtils(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(utils.slugify("Hello World"), "hello-world")
        self.assertEqual(utils.slugify("Hello   World"), "hello-world")
        self.assertEqual(utils.slugify("Hello World!"), "hello-world")
        self.assertEqual(utils.slugify("123 Hello"), "123-hello")
        self.assertEqual(utils.slugify("Hello-World"), "hello-world")
        self.assertEqual(utils.slugify("  Hello World  "), "hello-world")
        self.assertEqual(utils.slugify("Hello_World"), "hello-world")
        self.assertEqual(utils.slugify("Hello 🚀 World"), "hello-world")
        self.assertEqual(utils.slugify("🚀 Hello"), "hello")
        self.assertEqual(utils.slugify("Hello 🚀"), "hello")
        self.assertEqual(utils.slugify("Hello\nWorld"), "hello-world")
        self.assertEqual(utils.slugify("Hello\x01World"), "hello-world")
        self.assertEqual(utils.slugify("Hello\tWorld"), "hello-world")
        self.assertEqual(utils.slugify("Hello  \n  World"), "hello-world")


if __name__ == "__main__":
    unittest.main()
