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
            format_grammar._parse_segments("", False), ([""], {}, [])
        )

    def test_text(self):
        self.assertEqual(
            format_grammar._parse_segments("text", False),
            (["text"], {}, []),
        )

    def test_cpp(self):
        self.assertEqual(
            format_grammar._parse_segments("{ code; }", False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, "code;", 0, 0, False)]},
                [],
            ),
        )

    def test_word_in_braces(self):
        self.assertEqual(
            format_grammar._parse_segments("{AND}", False),
            (["{AND}"], {}, []),
        )

    def test_cpp_str(self):
        self.assertEqual(
            format_grammar._parse_segments('{ "\\x {"; }', False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, '"\\x {";', 0, 0, False)]},
                [],
            ),
        )

    def test_brace_in_str(self):
        self.assertEqual(
            format_grammar._parse_segments('"{" not code }', False),
            (['"{" not code }'], {}, []),
        )

    def test_quote_regex(self):
        self.assertEqual(
            format_grammar._parse_segments('\\"', False),
            (['\\"'], {}, []),
        )

    def test_block_comment_quote(self):
        self.assertEqual(
            format_grammar._parse_segments('/* " */', False),
            (['/* " */'], {}, []),
        )

    def test_cpp_after_block_comment(self):
        self.assertEqual(
            format_grammar._parse_segments("/* */{ code; }", False),
            (
                ["/* */", None, ""],
                {0: [format_grammar._CppCode(1, "code;", 5, 0, False)]},
                [],
            ),
        )

    def test_line_comment_quote(self):
        self.assertEqual(
            format_grammar._parse_segments('{\n// "\n}', False),
            (
                ["", None, ""],
                {0: [format_grammar._CppCode(1, '// "', 0, 0, False)]},
                [],
            ),
        )

    def test_table(self):
        self.assertEqual(
            format_grammar._parse_segments(
                "content\n"
                "/* table-begin */\n"
                "{VAR} { return SIMPLE_TOKEN(VAR); }\n"
                "{WHILE} { return SIMPLE_TOKEN(WHILE); }\n"
                "/* table-end */\n"
                "more content\n",
                False,
            ),
            (
                [
                    "content\n" "/* table-begin */\n",
                    None,
                    "\n" "/* table-end */\n" "more content\n",
                ],
                {},
                [
                    format_grammar._Table(
                        1,
                        "{VAR} { return SIMPLE_TOKEN(VAR); }\n"
                        "{WHILE} { return SIMPLE_TOKEN(WHILE); }",
                    )
                ],
            ),
        )

    def test_table_with_space(self):
        self.assertEqual(
            format_grammar._parse_segments(
                "content\n"
                " /* table-begin */\n"
                "{VAR} { return SIMPLE_TOKEN(VAR); }\n"
                "{WHILE} { return SIMPLE_TOKEN(WHILE); }\n"
                " /* table-end */\n"
                "more content\n",
                False,
            ),
            (
                [
                    "content\n /* table-begin */\n",
                    None,
                    "\n /* table-end */\nmore content\n",
                ],
                {},
                [
                    format_grammar._Table(
                        1,
                        "{VAR} { return SIMPLE_TOKEN(VAR); }\n"
                        "{WHILE} { return SIMPLE_TOKEN(WHILE); }",
                    )
                ],
            ),
        )

    def test_table_tokens(self):
        self.assertEqual(
            format_grammar._parse_segments(
                "%tokens\n"
                "  // Comment\n"
                "  // table-begin\n"
                "  VAR\n"
                "  WHILE\n"
                "  // table-end\n"
                "  MORE\n",
                False,
            ),
            (
                [
                    "%tokens\n" "  // Comment\n" "  // table-begin\n",
                    None,
                    "\n" "  // table-end\n" "  MORE\n",
                ],
                {},
                [format_grammar._Table(1, "  VAR\n" "  WHILE")],
            ),
        )

    def test_format_table_defines(self):
        text_segments = [None]
        format_grammar._format_table_segments(
            text_segments,
            [
                format_grammar._Table(
                    0,
                    'DEFAULT "default"\n'
                    'CONTINUE "continue"\n'
                    'DOUBLE_ARROW "=>"',
                )
            ],
            False,
        )
        self.assertEqual(
            text_segments,
            [
                'CONTINUE     "continue"\n'
                'DEFAULT      "default"\n'
                'DOUBLE_ARROW "=>"'
            ],
        )

    def test_format_table_returns(self):
        text_segments = [None]
        format_grammar._format_table_segments(
            text_segments,
            [
                format_grammar._Table(
                    0,
                    "{VAR} { return SIMPLE_TOKEN(VAR); }\n"
                    "{WHILE} { return SIMPLE_TOKEN(WHILE); }",
                )
            ],
            False,
        )
        self.assertEqual(
            text_segments,
            [
                "{VAR}   { return SIMPLE_TOKEN(VAR);   }\n"
                "{WHILE} { return SIMPLE_TOKEN(WHILE); }"
            ],
        )

    def test_format_table_tokens(self):
        text_segments = [None]
        format_grammar._format_table_segments(
            text_segments,
            [
                format_grammar._Table(
                    0,
                    "  AND\n" "  CONTINUE\n" "  BREAK",
                )
            ],
            False,
        )
        self.assertEqual(
            text_segments,
            [
                "  AND\n" "  BREAK\n" "  CONTINUE",
            ],
        )


if __name__ == "__main__":
    unittest.main()
