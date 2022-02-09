"""Generates the predence dot and svg files.

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
from typing import List

import graphviz  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dot_path",
        metavar="PATH",
        help="The path for the output .dot file.",
        required=True,
    )
    parser.add_argument(
        "--svg_path",
        metavar="PATH",
        help="The path for the output .svg file.",
        required=True,
    )
    return parser.parse_args()


class Assoc(Enum):
    """Associativity of operators, with shapes as values."""

    LEFT = "rarrow"
    RIGHT = "larrow"
    NONE = "ellipse"


class Graph(object):

    _counter = 0

    def __init__(self) -> None:
        comment = __copyright__.strip()
        comment += "\nAUTO-GENERATED: use precedence.sh."
        # graphviz comments the first line but not later ones, so replace
        # newlines with comment markers.
        comment = comment.replace("\n", "\n// ")
        self._dot = graphviz.Digraph(
            comment=comment,
            graph_attr={
                "layout": "dot",
                "rankdir": "TB",
                "rank": "min",
            },
            node_attr={
                "fontsize": "12",
                "height": "0",
                "fontname": "Segoe UI,Helvetica,Arial,sans-serif",
            },
            edge_attr={"dir": "none"},
        )

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

    def _make_op(self, ops: List[str], assoc: Assoc) -> str:
        """Makes a node for an operator, or cluster of operators."""
        self._counter = self._counter + 1
        name = f"op{self._counter}"
        label = "<br/>".join([html.escape(op) for op in ops])
        self._dot.node(
            name,
            f"<{label}>",
            shape=assoc.value,
        )
        return name

    def _make_edge(self, higher: str, lower: str) -> None:
        """Makes an edge from a high predence op to a low precedence op."""
        self._dot.edge(higher, lower)

    def render(self, dot_path: str, image_path: str) -> None:
        """Renders the graph, keeping the dot source file."""
        self._dot.render(filename=dot_path, outfile=image_path)

    def source(self) -> str:
        """Returns the graph source for tests."""
        return self.source()


def main() -> None:
    parsed_args = parse_args()
    print("Generating precedence graph...")
    Graph().render(parsed_args.dot_path, parsed_args.svg_path)
    print("Done!")


if __name__ == "__main__":
    main()
