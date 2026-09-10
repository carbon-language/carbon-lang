#!/usr/bin/env python3

"""Structural checks on the TextMate grammar.

None of these compare against golden output, so a grammar change never
requires regenerating them.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import unittest
from pathlib import Path

import tmlanguage

# The grammar gives these no scope, so the symbol check skips them.
_BRACKETS = frozenset("()[]{}")

# Scope prefixes that count as keyword highlighting. Which one a word gets
# depends on context; the check only requires that it not match an identifier
# rule instead.
_KEYWORD_SCOPES = (
    "storage.type",
    "storage.modifier",
    "keyword.",
    "variable.language",
    "constant.language",
    "support.type.builtin",
    "support.class",
)

_OPERATOR_SCOPES = ("keyword.operator", "punctuation.")

# Every `token_kind.def` macro that carries a spelling. A new one has to be
# classified here, rather than silently going untested.
_SPELLING_MACROS = frozenset(
    {
        "CARBON_KEYWORD_TOKEN",
        "CARBON_DECL_INTRODUCER_TOKEN",
        "CARBON_SYMBOL_TOKEN",
        "CARBON_ONE_CHAR_SYMBOL_TOKEN",
        "CARBON_OPENING_GROUP_SYMBOL_TOKEN",
        "CARBON_CLOSING_GROUP_SYMBOL_TOKEN",
    }
)


def _root() -> Path:
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        return Path(srcdir) / (os.environ.get("TEST_WORKSPACE") or "_main")
    return Path(__file__).resolve().parents[2]


_ROOT = _root()
_GRAMMAR = tmlanguage.Grammar.load(
    _ROOT / "utils/vscode/carbon.tmLanguage.json"
)
_TOKEN_KIND_DEF = (_ROOT / "toolchain/lex/token_kind.def").read_text(
    encoding="utf-8"
)


def _lexer_spellings() -> tuple[list[str], list[str]]:
    """Returns the keyword and symbol spellings from `token_kind.def`."""
    keywords = re.findall(
        r'CARBON_(?:DECL_INTRODUCER|KEYWORD)_TOKEN\(\s*\w+,\s*"([^"]+)"',
        _TOKEN_KIND_DEF,
    )
    symbols = re.findall(
        r"CARBON_(?:ONE_CHAR_|OPENING_GROUP_|CLOSING_GROUP_)?SYMBOL_TOKEN"
        r'\(\s*\w+,\s*"((?:[^"\\]|\\.)+)"',
        _TOKEN_KIND_DEF,
    )
    return keywords, [s.replace("\\\\", "\\") for s in symbols]


_KEYWORDS, _SYMBOLS = _lexer_spellings()


def _first_token(source: str) -> tmlanguage.Token:
    """Returns the token covering the start of a single-line `source`."""
    tokens, _ = tmlanguage.tokenize_line(
        _GRAMMAR, 0, source + "\n", _GRAMMAR.initial_stack()
    )
    return tokens[0]


def _region_open_at_eof(text: str) -> bool:
    """Whether a `begin`/`end` region is still open at the end of `text`.

    A region that never closes recolors the rest of the file.
    """
    stack = _GRAMMAR.initial_stack()
    for linenum, line in enumerate(tmlanguage.split_lines(text)):
        _, stack = tmlanguage.tokenize_line(
            _GRAMMAR, linenum, line + "\n", stack
        )
    return len(stack) > 1


def _corpus() -> list[Path]:
    """The Carbon files this runs the grammar over.

    The nine samples exercise constructs deliberately; `examples/` adds 51
    whole programs.
    """
    return sorted(
        list(_ROOT.glob("examples/**/*.carbon"))
        + list(_ROOT.glob("utils/textmate/Samples/*.carbon"))
    )


class GrammarTest(unittest.TestCase):
    def test_spellings_were_read(self):
        # Every check below reads these, and `re.findall` returns an empty
        # list rather than failing, so a macro rename would leave them
        # iterating nothing.
        self.assertGreater(len(_KEYWORDS), 60, "no keywords in token_kind.def")
        self.assertGreater(len(_SYMBOLS), 40, "no symbols in token_kind.def")
        spelled = set(
            re.findall(r'(CARBON_[A-Z_]*TOKEN)\(\s*\w+,\s*"', _TOKEN_KIND_DEF)
        )
        self.assertEqual(
            sorted(spelled - _SPELLING_MACROS),
            [],
            "token_kind.def has a spelling macro this does not classify",
        )

    def test_regex_dialect(self):
        # Passing alone does not prove `re` and Oniguruma agree, because they
        # can interpret shared syntax differently; only `conformance_test.py`
        # rules that out. This is the part that runs hermetically.
        self.assertEqual(tmlanguage.check_regex_dialect(_GRAMMAR), [])

    def test_includes_resolve(self):
        repository = _GRAMMAR.raw.get("repository", {})
        unresolved = []

        def walk(node):
            if isinstance(node, dict):
                target = node.get("include")
                if isinstance(target, str):
                    # `resolve` understands `#name`, `$self`, and another
                    # grammar's scope name, and drops anything else, so a typo
                    # loses rules with no other symptom.
                    if target.startswith("#"):
                        if target[1:] not in repository:
                            unresolved.append(target)
                    elif target != "$self" and "." not in target:
                        unresolved.append(target)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(_GRAMMAR.raw)
        self.assertEqual(
            unresolved, [], "`include` targets tmlanguage cannot resolve"
        )

    def test_keyword_coverage(self):
        for keyword in _KEYWORDS:
            # One per line, so a declaration rule cannot match the next
            # keyword as the name being declared.
            token = _first_token(keyword)
            with self.subTest(keyword=keyword):
                self.assertTrue(
                    token.scopes[-1].startswith(_KEYWORD_SCOPES),
                    f"`{keyword}` scopes as {token.scopes[-1]}, "
                    "not as a keyword",
                )
                self.assertEqual(
                    token.end,
                    len(keyword),
                    f"`{keyword}` is split at offset {token.end}",
                )

    def test_symbol_coverage(self):
        for symbol in _SYMBOLS:
            if symbol in _BRACKETS:
                continue
            token = _first_token(symbol)
            with self.subTest(symbol=symbol):
                self.assertTrue(
                    token.scopes[-1].startswith(_OPERATOR_SCOPES),
                    f"`{symbol}` scopes as {token.scopes[-1]}, "
                    "not as an operator",
                )
                # A shorter rule listed first would split a longer symbol.
                self.assertEqual(
                    token.end,
                    len(symbol),
                    f"`{symbol}` is split at offset {token.end}",
                )

    def test_no_stale_keywords(self):
        # A keyword list is a leading pure word alternation, `\b(a|b|c)`,
        # followed by `\b` or by `\s` where the rule goes on to match a name.
        # Requiring the alternation to contain only words keeps another
        # regex's alternation out of the match.
        known = set(_KEYWORDS)
        stale = []
        lists = 0
        for _, regex in _GRAMMAR.all_regexes():
            listed = re.match(r"\\b\(([A-Za-z_|]+)\)(?:\\b|\\s)", regex)
            if listed is None:
                continue
            lists += 1
            stale.extend(
                word for word in listed.group(1).split("|") if word not in known
            )
        # Only lists written in that shape are reached, so this covers most
        # of the grammar's keywords rather than all of them; the guard is
        # against reaching none.
        self.assertGreater(lists, 15, "no keyword lists recognized")
        self.assertEqual(
            stale,
            [],
            "the grammar treats these as keywords but token_kind.def "
            "no longer does",
        )

    def test_no_open_region_at_eof(self):
        corpus = _corpus()
        # Guards against the globs resolving to nothing under Bazel.
        self.assertGreater(
            len(corpus), 40, f"expected about 60 files, found {len(corpus)}"
        )
        open_at_eof = [
            str(path.relative_to(_ROOT))
            for path in corpus
            if _region_open_at_eof(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            open_at_eof,
            [],
            "a region left open at end of file recolors the rest of it",
        )


if __name__ == "__main__":
    unittest.main()
