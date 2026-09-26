#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///

"""Compares the scopes given to the TextMate samples against checked-in goldens.

The nine samples in `utils/textmate/Samples` are what the bundle's renderings
are made from, so a golden of their scopes shows exactly which construct
changed color. Regenerate with

    ./utils/textmate/golden_test.py --update

which regenerates the renderings from the same grammar, so the goldens and the
renderings cannot disagree.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import difflib
import os
import sys
import unittest
from pathlib import Path

import render_sample
import tmlanguage

_HEADER = """\
# Golden scopes for the TextMate grammar. Regenerate with:
#   ./utils/textmate/golden_test.py --update
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""


def _root() -> Path:
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        return Path(srcdir) / (os.environ.get("TEST_WORKSPACE") or "_main")
    return Path(__file__).resolve().parents[2]


_ROOT = _root()
_SAMPLES = _ROOT / "utils/textmate/Samples"
_GOLDENS = _ROOT / "utils/textmate/testdata/highlight"
_GRAMMAR_PATH = _ROOT / "utils/vscode/carbon.tmLanguage.json"

# Width of the text column. `repr` escapes, so the escaped form is what has to
# fit.
_TEXT_WIDTH = 26


def _format_scopes(grammar: tmlanguage.Grammar, scopes: list[str]) -> str:
    """Formats a scope stack, dropping the root scope every token carries."""
    return " ".join(s for s in scopes if s != grammar.scope_name) or "-"


def _render_scopes(grammar: tmlanguage.Grammar, source: str) -> str:
    """Renders every token under the source line it came from."""
    lines = tmlanguage.split_lines(source)
    by_line: dict[int, list[tmlanguage.Token]] = {}
    for token in tmlanguage.tokenize(grammar, source):
        by_line.setdefault(token.line, []).append(token)
    out = []
    for number, text in enumerate(lines):
        if not text.strip():
            continue
        out.append(f"{number + 1:4} | {text}")
        for token in by_line.get(number, []):
            piece = text[token.start : token.end]
            # Whitespace a rule scopes is worth showing; whitespace no rule
            # touched is noise.
            if not piece.strip() and len(token.scopes) <= 1:
                continue
            # The span already gives the boundaries, so long text can be
            # truncated.
            quoted = repr(piece)
            if len(quoted) > _TEXT_WIDTH:
                quoted = quoted[: _TEXT_WIDTH - 4] + "..." + quoted[-1]
            span = f"{token.start}-{token.end}"
            out.append(
                f"     | {span:>9} {quoted:<{_TEXT_WIDTH}} "
                f"{_format_scopes(grammar, token.scopes)}"
            )
    return "\n".join(out) + "\n"


def _cases() -> list[tuple[Path, Path]]:
    return [
        (path, _GOLDENS / (path.stem + ".scopes"))
        for path in sorted(_SAMPLES.glob("*.carbon"))
    ]


class GoldenTest(unittest.TestCase):
    def test_scopes_match_goldens(self):
        grammar = tmlanguage.Grammar.load(_GRAMMAR_PATH)
        cases = _cases()
        self.assertTrue(cases, f"no samples found in {_SAMPLES}")
        for source_path, golden_path in cases:
            with self.subTest(sample=source_path.name):
                actual = _HEADER + _render_scopes(
                    grammar, source_path.read_text(encoding="utf-8")
                )
                self.assertTrue(
                    golden_path.exists(),
                    f"missing golden {golden_path.name}: run "
                    "./utils/textmate/golden_test.py --update",
                )
                golden = golden_path.read_text(encoding="utf-8")
                if golden == actual:
                    continue
                # unittest elides a diff this large, so show a readable one.
                delta = list(
                    difflib.unified_diff(
                        golden.splitlines(),
                        actual.splitlines(),
                        fromfile=f"{golden_path.name} (checked in)",
                        tofile=f"{golden_path.name} (current)",
                        lineterm="",
                        n=1,
                    )
                )
                shown = "\n".join(delta[:40])
                if len(delta) > 40:
                    shown += f"\n... and {len(delta) - 40} more lines"
                self.fail(
                    f"highlighting changed for {source_path.name}. If this is "
                    f"intended, regenerate with\n"
                    f"  ./utils/textmate/golden_test.py --update\n\n{shown}"
                )

    def test_renderings_are_current(self):
        # The renderings are how a reviewer sees a highlighting change, so a
        # stale one misrepresents the grammar. Checking only that the file
        # exists would let a hand-edited SVG through.
        grammar = tmlanguage.Grammar.load(_GRAMMAR_PATH)
        stale = []
        for source_path, _ in _cases():
            svg_path = source_path.with_suffix(".svg")
            expected = render_sample.render(
                grammar, source_path.read_text(encoding="utf-8")
            )
            if (
                not svg_path.exists()
                or svg_path.read_text(encoding="utf-8") != expected
            ):
                stale.append(svg_path.name)
        self.assertEqual(
            stale,
            [],
            "renderings are stale: run ./utils/textmate/golden_test.py "
            "--update",
        )

    def test_no_orphaned_goldens(self):
        # A golden whose sample is gone is never regenerated and never
        # compared, so it sits stale forever.
        stems = {source.stem for source, _ in _cases()}
        orphans = sorted(
            path.name
            for path in _GOLDENS.glob("*.scopes")
            if path.stem not in stems
        )
        self.assertEqual(orphans, [])


def _update() -> int:
    grammar = tmlanguage.Grammar.load(_GRAMMAR_PATH)
    _GOLDENS.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    stems = {source.stem for source, _ in cases}
    for path in sorted(_GOLDENS.glob("*.scopes")):
        if path.stem not in stems:
            path.unlink()
            print(f"removed {path.relative_to(_ROOT)}")
    for source_path, golden_path in cases:
        golden_path.write_text(
            _HEADER
            + _render_scopes(grammar, source_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        print(f"wrote {golden_path.relative_to(_ROOT)}")

    # Regenerate the renderings from this same grammar, so they cannot
    # disagree with the goldens.
    return render_sample.main(
        ["--grammar", str(_GRAMMAR_PATH)] + [str(source) for source, _ in cases]
    )


if __name__ == "__main__":
    # The only flag of our own is `--update`; the rest belongs to unittest.
    if "--update" in sys.argv:
        sys.exit(_update())
    unittest.main()
