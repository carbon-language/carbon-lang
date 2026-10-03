#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///

"""Renders Carbon sources as highlighted SVG.

This regenerates the renderings next to the TextMate samples, so they show what
the grammar in this repository actually produces rather than whatever an editor
looked like when someone last took a screenshot by hand.

The SVG holds the source as text rather than as outlines, so whatever displays
it lays the text out and draws the glyphs; nothing here rasterizes.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import html
import itertools
import sys
from pathlib import Path
from typing import Iterator, Optional

import tmlanguage

# The grammar is part of the VS Code extension, which must all be contained
# under the extension's root directory.
_GRAMMAR_PATH = (
    Path(__file__).resolve().parents[1] / "vscode" / "carbon.tmLanguage.json"
)

# A theme built for this grammar rather than taken from an editor. Each family
# of related scopes gets one color, and bold and italic distinguish the scopes
# within a family, so every scope the grammar emits renders differently and a
# change to what it scopes is visible in the rendering. A scope with no entry of
# its own takes its family's. The colors are Catppuccin Mocha's.
_COLORS = {
    "comment": "#7f849c",
    "punctuation.definition.comment": "#7f849c",
    "keyword.other.directive": "#7f849c",
    "storage": "#cba6f7",
    "constant.language": "#cba6f7",
    "keyword.control": "#f38ba8",
    "keyword.operator": "#94e2d5",
    "keyword.other": "#94e2d5",
    "punctuation": "#74c7ec",
    "entity.name.type": "#f9e2af",
    "support.type": "#f9e2af",
    "support.class": "#f9e2af",
    "entity.name.namespace": "#f5e0dc",
    "entity.name.tag": "#f5e0dc",
    "entity.name.function": "#89b4fa",
    "variable": "#cdd6f4",
    "string": "#a6e3a1",
    "constant.character.escape": "#f5c2e7",
    "constant.numeric": "#fab387",
    # Inline C++ is highlighted by the C++ grammar, so here it gets the color of
    # unscoped text.
    "meta.embedded": "#a6adc8",
}

# A theme resolves the font style separately from the color, so a scope takes
# its color from its family and its style from its own entry. An empty style
# clears one an enclosing scope set, which keeps an escape non-italic inside an
# italic single-quoted string.
_STYLES = {
    "comment": "italic",
    "punctuation.definition.comment": "",
    "keyword.other.directive": "bold",
    "storage.modifier": "italic",
    "constant.language": "bold",
    "keyword.other": "italic",
    "punctuation.terminator": "bold",
    "punctuation.definition.string": "italic",
    "punctuation.definition.raw-identifier": "bold italic",
    "support.type": "italic",
    "support.class": "bold",
    "entity.name.tag": "italic",
    "variable.parameter": "italic",
    "variable.language": "bold",
    "string.quoted.single": "italic",
    "string.quoted.triple": "bold",
    "constant.character.escape": "",
    "constant.numeric.hex": "italic",
    "constant.numeric.binary": "bold",
    "constant.numeric.octal": "bold italic",
    "meta.embedded": "italic",
}

# The four font styles a theme can name, as SVG presentation attributes.
_STYLE_ATTRS = {
    "italic": ' font-style="italic"',
    "bold": ' font-weight="bold"',
    "underline": ' text-decoration="underline"',
    "strikethrough": ' text-decoration="line-through"',
}

_BACKGROUND = "#1e1e2e"
# Unscoped text. It is dimmer than any scope's color, so text the grammar misses
# stands out.
_FOREGROUND = "#a6adc8"
_GUTTER = "#6c7086"
# Whatever monospace font the viewer has: nothing can be fetched, because
# GitHub serves SVG under `default-src 'none'` and a web font would be blocked.
# The text simply flows, so the font's own metrics lay each line out.
_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
_FONT_SIZE = 14
# Only used to size the canvas. Monospace advances cluster near 0.6em, so a
# font a little wider than this just runs closer to the right edge.
_CHAR_WIDTH = 8.5
_LINE_HEIGHT = 19
# How far above the bottom of its line the baseline sits, leaving room for the
# descenders of `g` and `y` at this line height.
_DESCENT = 5
_PAD = 12


def _select(table: dict[str, str], scopes: list[str], default: str) -> str:
    """Resolves a scope stack against one table, innermost scope first.

    Within a scope the longest matching prefix wins, so a selector such as
    `keyword.other.directive` beats `keyword.other`. A scope the table names at
    no prefix falls back to its enclosing scope, as a theme does.
    """
    for scope in reversed(scopes):
        while scope:
            if scope in table:
                return table[scope]
            scope = scope.rpartition(".")[0]
    return default


def _span_for(scopes: list[str]) -> str:
    """Resolves a scope stack to the attributes its `tspan` carries."""
    attrs = "".join(
        _STYLE_ATTRS[word]
        for word in _select(_STYLES, scopes, "").split()
        if word in _STYLE_ATTRS
    )
    return f'fill="{_select(_COLORS, scopes, _FOREGROUND)}"{attrs}'


def _runs(spans: list[str]) -> Iterator[tuple[int, int, str]]:
    """Merges a per-character attribute list into `(start, end, attrs)` runs."""
    end = 0
    for attrs, run in itertools.groupby(spans):
        start, end = end, end + len(list(run))
        yield start, end, attrs


def render(grammar: tmlanguage.Grammar, source: str) -> str:
    """Renders a Carbon source as a standalone SVG document."""
    lines = tmlanguage.split_lines(source)
    plain = f'fill="{_FOREGROUND}"'
    styled = [[plain] * len(line) for line in lines]
    for token in tmlanguage.tokenize(grammar, source):
        attrs = _span_for(token.scopes)
        row = styled[token.line]
        # A token runs one past the line, over the newline it was tokenized
        # with, and there is no column there to style.
        for column in range(token.start, min(token.end, len(row))):
            row[column] = attrs

    # Trailing blank lines would only pad the bottom of the image.
    while lines and not lines[-1].strip():
        lines.pop()
        styled.pop()

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
    for linenum, linetext in enumerate(lines):
        baseline = _PAD + (linenum + 1) * _LINE_HEIGHT - _DESCENT
        spans = [
            f'<tspan fill="{_GUTTER}">{str(linenum + 1).rjust(digits)} </tspan>'
        ] + [
            f"<tspan {attrs}>"
            f"{html.escape(linetext[start:end], quote=False)}</tspan>"
            for start, end, attrs in _runs(styled[linenum])
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
    parser.add_argument("--grammar", type=Path, default=_GRAMMAR_PATH)
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
