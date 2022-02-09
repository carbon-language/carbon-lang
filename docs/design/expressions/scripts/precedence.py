"""Generates the predence dot and svg files.

Using this requires `dot` be locally installed.

New operators should be added to Graph.write.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
from enum import Enum
import html
from pathlib import Path
import subprocess
import textwrap
from typing import TextIO, List

DOT_HEADER = """
# Auto-generated using precedence.sh.
digraph {
  layout = dot
  rankdir = TB
  rank = "min"
  node [shape="none" fontsize="12" height="0"
        fontname="BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"]
  edge [dir="none"]

"""


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dot_path",
        metavar="PATH",
        help="The path for the output .dot file.",
    )
    parser.add_argument(
        "--svg_path",
        metavar="PATH",
        help="The path for the output .svg file.",
    )
    return parser.parse_args()


class Assoc(Enum):
    LEFT = "rarrow"
    RIGHT = "larrow"
    NONE = "rect"


class Graph(object):

    _counter = 0

    def __init__(self, dot_file: TextIO) -> None:
        self._dot_file = dot_file

    def _make_op(self, ops: List[str], assoc: Assoc) -> str:
        """Makes a node for an operator, or cluster of operators."""
        self._counter = self._counter + 1
        name = f"op{self._counter}"
        label = "<br/>".join([html.escape(op) for op in ops])
        self._dot_file.write(
            f'  {name} [label=<{label}> shape="{assoc.value}"]\n'
        )
        return name

    def _make_edge(self, higher: str, lower: str) -> None:
        """Makes an edge from a high predence op to a low precedence op."""
        self._dot_file.write(f"  {higher} -> {lower}\n")

    def write(self) -> None:
        """Generates the actual precedence content."""
        self._dot_file.write(textwrap.indent(__copyright__.lstrip("\n"), "# "))
        self._dot_file.write(DOT_HEADER)

        parens = self._make_op(["(...)"], Assoc.NONE)
        op_as = self._make_op(["x as T"], Assoc.NONE)
        ops_and_or = self._make_op(["x and y", "x or y"], Assoc.LEFT)
        op_not = self._make_op(["not x"], Assoc.NONE)
        ops_comp = self._make_op(
            ["x == y", "x != y", "x < y", "x <= y", "x > y", "x >= y"],
            Assoc.NONE,
        )

        self._make_edge(parens, op_as)
        self._make_edge(parens, op_not)
        self._make_edge(op_not, ops_and_or)
        self._make_edge(op_as, ops_comp)
        self._make_edge(ops_comp, ops_and_or)

        self._dot_file.write("}\n")


def main() -> None:
    parsed_args = parse_args()

    dot_path = Path(parsed_args.dot_path)
    print(f"Writing {dot_path}...")
    with dot_path.open("w") as dot_file:
        Graph(dot_file).write()

    image_path = Path(parsed_args.svg_path)
    print(f"Writing {image_path}...")
    subprocess.check_call(
        ["dot", "-Tsvg", f"-o{image_path}", dot_path],
    )
    print("Done!")


if __name__ == "__main__":
    main()
