#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///

"""Checks `tmlanguage.py` against the real VS Code tokenizer.

Nothing else checks that `re` and Oniguruma agree on this grammar, so run it
after any change to `tmlanguage.py` or to the regex syntax the grammar uses.
It needs node and two npm packages, so it is not hermetic and has no Bazel
target; run it directly. See README.md.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import hashlib
import shutil
import subprocess
import unittest
from pathlib import Path

import tmlanguage

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_GRAMMAR_PATH = _ROOT / "utils/vscode/carbon.tmLanguage.json"

_SETUP = (
    "Install vscode-textmate, which this compares against:\n"
    "  cd utils/textmate && npm install --no-save "
    "vscode-textmate vscode-oniguruma"
)


def _corpus() -> list[Path]:
    patterns = (
        "core/**/*.carbon",
        "examples/**/*.carbon",
        "toolchain/**/testdata/**/*.carbon",
        "utils/textmate/Samples/*.carbon",
    )
    return sorted(path for pattern in patterns for path in _ROOT.glob(pattern))


def _python_digest(path: Path, grammar: tmlanguage.Grammar) -> str:
    """Digests the scope stack of every character, as conformance.mjs does."""
    stack = grammar.initial_stack()
    digest = hashlib.sha256()
    lines = tmlanguage.split_lines(path.read_text(encoding="utf-8"))
    for number, line in enumerate(lines):
        tokens, stack = tmlanguage.tokenize_line(
            grammar, number, line + "\n", stack
        )
        per_char = ["-"] * len(line)
        for token in tokens:
            for i in range(token.start, min(token.end, len(line))):
                per_char[i] = "/".join(token.scopes)
        digest.update((",".join(per_char) + ";").encode("utf-8"))
    return digest.hexdigest()


class ConformanceTest(unittest.TestCase):
    def test_matches_vscode_textmate(self):
        if shutil.which("node") is None:
            self.fail(f"node is not installed.\n{_SETUP}")
        if not (_HERE / "node_modules" / "vscode-textmate").is_dir():
            self.fail(f"vscode-textmate is not installed.\n{_SETUP}")

        corpus = _corpus()
        self.assertTrue(corpus, "no Carbon files found")
        grammar = tmlanguage.Grammar.load(_GRAMMAR_PATH)

        result = subprocess.run(
            ["node", str(_HERE / "conformance.mjs"), str(_GRAMMAR_PATH)],
            input="\n".join(str(path) for path in corpus),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0, f"conformance.mjs failed:\n{result.stderr}"
        )
        expected = {}
        for line in result.stdout.splitlines():
            digest, _, name = line.partition(" ")
            expected[name] = digest

        # A short read means node stopped early without failing, which would
        # otherwise surface as a KeyError below.
        self.assertCountEqual(
            expected,
            [str(path) for path in corpus],
            "conformance.mjs did not report a digest for every file",
        )

        mismatched = [
            str(path.relative_to(_ROOT))
            for path in corpus
            if _python_digest(path, grammar) != expected[str(path)]
        ]
        self.assertEqual(
            mismatched,
            [],
            "tmlanguage.py disagrees with vscode-textmate; `re` and "
            "Oniguruma no longer agree on this grammar",
        )


if __name__ == "__main__":
    unittest.main()
