"""Tests for format_grammar.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

from executable_semantics.syntax import format_grammar


class TestFormatGrammar(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments("", False), ([""], {})
        )

    def test_text(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments("text", False), (["text"], {})
        )

    def test_cpp(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments("{ code; }", False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, "code;", 0, 0, False)]},
            ),
        )

    def test_word_in_braces(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments("{AND}", False),
            (["{AND}"], {}),
        )

    def test_cpp_str(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments('{ "\\x {"; }', False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, '"\\x {";', 0, 0, False)]},
            ),
        )

    def test_brace_in_str(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments('"{" not code }', False),
            (['"{" not code }'], {}),
        )

    def test_quote_regex(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments('\\"', False),
            (['\\"'], {}),
        )

    def test_block_comment_quote(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments('/* " */', False),
            (['/* " */'], {}),
        )

    def test_cpp_after_block_comment(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments("/* */{ code; }", False),
            (
                ["/* */", None, ""],
                {0: [format_grammar._CppCode(1, "code;", 5, 0, False)]},
            ),
        )

    def test_line_comment_quote(self):
        self.assertEqual(
            format_grammar._parse_cpp_segments('{\n// "\n}', False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, '// "', 0, 0, False)]},
            ),
        )


if __name__ == "__main__":
    unittest.main()
