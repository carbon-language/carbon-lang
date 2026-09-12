#!/usr/bin/env python3

"""Renders Carbon sources as highlighted SVG.

This regenerates the renderings next to the TextMate samples, so they show what
the grammar in this repository actually produces rather than whatever an editor
looked like when someone last took a screenshot by hand.

Output is SVG, drawn by whatever displays it, so this needs nothing outside the
standard library. Colors are VS Code's Dark+; a scope the theme does not style
resolves outward through the scope stack, which is what makes a string's quotes
take the string color and a comment's `//` take the comment color.

    ./utils/vscode/render_sample.py utils/textmate/Samples/*.carbon
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import tmlanguage

# VS Code's Dark+, keyed by the selectors the theme itself uses. Matching is by
# longest dotted prefix, as a real theme does, so this stays correct for any
# grammar rather than only the scope names in use today.
_THEME = {
    "comment": "#6a9955",
    "constant.character.escape": "#d7ba7d",
    "constant.language": "#569cd6",
    "constant.numeric": "#b5cea8",
    "entity.name.function": "#dcdcaa",
    "entity.name.namespace": "#4ec9b0",
    "entity.name.tag": "#569cd6",
    "entity.name.type": "#4ec9b0",
    "keyword.control": "#c586c0",
    "keyword.operator": "#d4d4d4",
    "keyword.other": "#569cd6",
    "meta.embedded": "#d4d4d4",
    "storage.modifier": "#569cd6",
    "storage.type": "#569cd6",
    "string": "#ce9178",
    "support.class": "#4ec9b0",
    "support.function": "#dcdcaa",
    "support.type": "#4ec9b0",
    "support.type.property-name": "#9cdcfe",
    "support.variable": "#9cdcfe",
    "variable.language": "#569cd6",
    "variable.other": "#9cdcfe",
    "variable.other.enummember": "#4fc1ff",
    "variable.parameter": "#9cdcfe",
}

_BACKGROUND = "#1f1f1f"
_FOREGROUND = "#d4d4d4"
_GUTTER = "#6e7681"
# Whatever monospace font the viewer has: nothing can be fetched, because
# GitHub serves SVG under `default-src 'none'` and a web font would be blocked.
# The text simply flows, so the font's own metrics lay each line out.
_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
_FONT_SIZE = 14
# Only used to size the canvas. Monospace advances cluster near 0.6em, so a
# font a little wider than this just runs closer to the right edge.
_CHAR_WIDTH = 8.5
_LINE_HEIGHT = 19
_PAD = 12


def _color_for(scopes: list[str]) -> str:
    """Resolves a scope stack to a color, innermost scope first.

    Within a scope the longest matching prefix wins, so that a selector such as
    `variable.other.enummember` beats `variable.other`. A scope the theme does
    not style resolves outward to its enclosing scope, which is what gives a
    string's quotes the string color.
    """
    for scope in reversed(scopes):
        parts = scope.split(".")
        for end in range(len(parts), 0, -1):
            color = _THEME.get(".".join(parts[:end]))
            if color is not None:
                return color
    return _FOREGROUND


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs(colors: list[str]) -> list[tuple[int, int, str]]:
    """Merges a per-character color list into `(start, end, color)` runs."""
    runs: list[tuple[int, int, str]] = []
    for index, color in enumerate(colors):
        if runs and runs[-1][2] == color:
            runs[-1] = (runs[-1][0], index + 1, color)
        else:
            runs.append((index, index + 1, color))
    return runs


def render(grammar: tmlanguage.Grammar, source: str) -> str:
    """Renders a Carbon source as a standalone SVG document."""
    lines = source.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()

    colored = [[_FOREGROUND] * len(line) for line in lines]
    for token in tmlanguage.tokenize(grammar, source):
        if token.line >= len(colored):
            continue
        color = _color_for(token.scopes)
        row = colored[token.line]
        for column in range(token.start, min(token.end, len(row))):
            row[column] = color

    digits = len(str(len(lines))) if lines else 1
    longest = max((len(line) for line in lines), default=0)
    width = round(_PAD * 2 + (digits + 1 + longest) * _CHAR_WIDTH)
    height = _PAD * 2 + len(lines) * _LINE_HEIGHT

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"'
        f' height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{_BACKGROUND}"/>',
        # Indentation has to survive two eras of the spec. SVG 1.1 renderers
        # read `xml:space`, and honor it here on the group. Browsers follow
        # SVG 2, where whitespace is a CSS property and neither form is
        # inherited into text from an ancestor, so each `text` repeats it.
        f'<g xml:space="preserve" font-family="{_FONT}"'
        f' font-size="{_FONT_SIZE}">',
    ]
    for number, line in enumerate(lines):
        baseline = _PAD + (number + 1) * _LINE_HEIGHT - 5
        spans = [
            f'<tspan fill="{_GUTTER}">{str(number + 1).rjust(digits)} </tspan>'
        ]
        spans += [
            f'<tspan fill="{color}">{_escape(line[start:end])}</tspan>'
            for start, end, color in _runs(colored[number])
        ]
        out.append(
            f'<text style="white-space:pre" x="{_PAD}" y="{baseline}">'
            f"{''.join(spans)}</text>"
        )
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument(
        "--grammar",
        type=Path,
        default=Path(__file__).resolve().parent / "carbon.tmLanguage.json",
    )
    args = parser.parse_args(argv)

    grammar = tmlanguage.Grammar.load(args.grammar)
    for source_path in args.sources:
        out_path = source_path.with_suffix(".svg")
        out_path.write_text(
            render(grammar, source_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
